#!/usr/bin/env python3
"""
Integrated demo: Panthera arm (cartesian impedance polar) + ODrive biwheel
                 + Insight 9 VIO + viser scene + RGB stream — one web page.

Built from 8_arm_base_web_control.py with:
  • Insight 9 ROS subscribers (vio_100hz / vio_status / color compressed)
  • viser scene (frustum tracking VIO pose, floor grid, trail point cloud)
  • RGB MJPEG endpoint from Insight 9 color compressed
  • Apple-style web UI

Keyboard:
  Arm:    W/S radial  A/D orbit  Q/E up/down   I/K pitch  J/L yaw  U/O roll
          Z grip close  X grip open   Space print pose   M zero wrench   R home
  Chassis: Arrow keys (WASD freed up — Arm now exclusively uses WASD)

Run (after `source .venv/bin/activate`; ROS + Fast DDS set by `install/setup_runtime_env.sh`):
    python scripts/0_integrated_demo.py
Open http://<host>:8080 in browser.
"""

import asyncio
import json
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from scipy.spatial.transform import Rotation as Rot
import pinocchio as pin

import viser

# Panthera_lib lives under panthera_python/scripts (source-imported, not pip).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "panthera_python", "scripts"))
from Panthera_lib import Panthera

# ROS 2 (Insight 9 VIO + RGB)
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# ===================== CONFIG =====================
HOST = "0.0.0.0"
PORT = 8080            # FastAPI main page
VISER_PORT = 8084      # viser embedded iframe
TELEMETRY_HZ = 10      # halved from 20 to leave headroom for the impedance thread
IDLE_TIMEOUT = 0.3     # seconds with no key -> stop

# ── Arm — polar_web defaults; only stiffness ×1.2 (speed kept at baseline) ──
POS_STEP        = 0.002              # m/frame   (polar_web baseline)
ROT_STEP        = 0.2                # °/frame
POLAR_ANGLE_MAX = np.radians(1.5)
CTRL_FREQ_IMP   = 2000
KEY_CMD_HZ      = 200

MAX_TORQUE = [21.0, 36.0, 36.0, 30.0, 10.0, 10.0]
JOINT_VEL  = [0.5] * 6

# Mirror polar_web's CURRENT live values exactly (the [10,30,60] I used before
# was from a stale comment — polar_web has been retuned higher since).
K_POS = np.array([30.0, 30.0, 50.0])      # N/m
K_ROT = np.array([30.0, 30.0, 40.0])      # Nm/rad
K_CART = np.concatenate([K_POS, K_ROT])

B_POS = np.array([0.0, 0.0, 0.0])
B_ROT = np.array([0.0, 0.0, 0.0])
B_CART = np.concatenate([B_POS, B_ROT])

LAMBDA_DAMP = 0.05
TAU_LIMIT = np.array([20.0, 30.0, 30.0, 20.0, 10.0, 10.0])
JOINT_DAMPING = np.array([1.5, 2, 2, 2, 1, 0.8])

# Friction Fc/Fv/vel_threshold are loaded from yaml at runtime via robot.Fc / robot.Fv
# (see init_robot — the IMP_Fc/IMP_Fv module-level fallbacks below are only used
# if robot init fails or yaml is missing the friction section).
IMP_Fc         = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01])
IMP_Fv         = np.array([0.03, 0.03, 0.03, 0.03, 0.01, 0.01])
IMP_VEL_THRESH = 0.03

ENABLE_CORIOLIS = False
TOOL_OFFSET = np.array([0.165, 0.0, 0.0])

# Velocity low-pass filter cutoff (Hz). Without this, raw encoder noise gets
# multiplied by JOINT_DAMPING and excites mechanical resonance → growing
# self-oscillation. polar_web has this; 8_arm_base_web_control did NOT, which
# is why the integrated demo shook itself apart.
DQ_LPF_CUTOFF = 40.0

GRIPPER_STEP        = 0.02
GRIPPER_KP          = 8.0
GRIPPER_KD          = 0.5
GRIPPER_MIN_DEFAULT = 0.0
GRIPPER_MAX_DEFAULT = 1.6

FT_CUTOFF_FREQ = 5.0
FT_WARN_LO = 1.0
FT_WARN_HI = 5.0

JOINT_LIMIT_MARGIN = 0.1
JOINT_LIMIT_WARN_DURATION = 1.0

HOME_POS       = [0.24, 0.0, 0.15]
HOME_ROT_EULER = (0.0, np.pi / 2, 0.0)

# ── ODrive ──
LINEAR_SPEED  = 0.15
ANGULAR_SPEED = 30.0
ODRIVE_CMD_HZ = 20

WHEEL_RADIUS   = 0.05
WHEEL_BASE     = 0.25
INVERT_LEFT    = True
INVERT_RIGHT   = False
AXIS_LEFT      = 1
AXIS_RIGHT     = 0
ODRIVE_SERIAL  = None
TORQUE_CONSTANT = 8.27 / 270

BATT_CELLS   = 5
BATT_FULL_V  = 4.2 * BATT_CELLS
BATT_EMPTY_V = 3.0 * BATT_CELLS

# ── Insight 9 (ROS 2 topics + viser) ──
TOPIC_POSE   = "/insight/vio_100hz"
TOPIC_STATUS = "/insight/vio_status"
TOPIC_RGB    = "/camera/camera/color/image_rect_raw/compressed"

# RGB intrinsics (from camera_info)
RGB_FRAME_W = 1088
RGB_FRAME_H = 1920
RGB_FY      = 776.78

FRUSTUM_FOV_RAD = 2 * math.atan(RGB_FRAME_H / 2.0 / RGB_FY)
FRUSTUM_ASPECT  = RGB_FRAME_W / RGB_FRAME_H
FRUSTUM_SCALE   = 0.10

# Trail: 2Hz sample = much less GIL pressure on the impedance thread.
# Trail still grows visibly when you walk the arm around.
TRAIL_SAMPLE_HZ   = 2
TRAIL_HISTORY_SEC = 300
TRAIL_MAX_POINTS  = TRAIL_SAMPLE_HZ * TRAIL_HISTORY_SEC

# Floor grid
GRID_SIZE_M    = 10.0
GRID_CELL_M    = 0.5
GRID_SECTION_M = 2.0
GRID_Z         = -0.30

# RGB MJPEG cadence (just byte-passthrough, very cheap)
RGB_STREAM_FPS = 15

JPEG_QUALITY = 60     # only used by remaining (none) cv2 encodes; kept for legacy

# ── Arm vel-mode (replaces cartesian impedance) ──
# Velocity control via damped Jacobian pseudo-inverse + motor-level velocity
# tracking. No high-gain spring → no self-excited oscillation.
ARM_LIN_SPEED      = 0.15                                # m/s when key held (default — runtime-tunable via UI)
ARM_ANG_SPEED      = 1.0                                 # rad/s when key held (default — runtime-tunable via UI)
ARM_LIN_MIN, ARM_LIN_MAX = 0.02, 0.50
ARM_ANG_MIN, ARM_ANG_MAX = 0.1, 3.0
ARM_ACCEL_FACTOR   = 0.05                                # exponential smoothing
VEL_CTRL_HZ        = 100
ARM_JOINT_VEL_MAX  = np.array([2.0, 2.0, 2.0, 2.0, 3.0, 3.0])
ARM_DAMPING_BASE   = 0.01
ARM_MANIP_THRESH   = 0.01
SAFE_JOINT_POS     = [0.0, 0.5, 0.6, 0.0, 0.0, 0.0]
# Tool-frame velocity key map: (axis 0..5, sign ±1)
VEL_KEY_MAP_ARM = {
    "w": (0, +1), "s": (0, -1),    # tool X (forward / back)
    "a": (1, +1), "d": (1, -1),    # tool Y (left / right)
    "q": (2, +1), "e": (2, -1),    # tool Z (up / down)
    "u": (3, +1), "o": (3, -1),    # roll  (ω_x)
    "i": (4, +1), "k": (4, -1),    # pitch (ω_y)
    "j": (5, +1), "l": (5, -1),    # yaw   (ω_z)
}
# ==================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("arm_base_web")

# ── Arm key map ──
rot_step_rad = np.radians(ROT_STEP)
KEY_MAP_ARM = {
    "q": (np.array([0,  0,  POS_STEP]),   None),
    "e": (np.array([0,  0, -POS_STEP]),   None),
    "i": (np.zeros(3), ( 0,  rot_step_rad,  0)),
    "k": (np.zeros(3), ( 0, -rot_step_rad,  0)),
    "j": (np.zeros(3), ( 0,  0,  rot_step_rad)),
    "l": (np.zeros(3), ( 0,  0, -rot_step_rad)),
    "u": (np.zeros(3), ( rot_step_rad,  0,  0)),
    "o": (np.zeros(3), (-rot_step_rad,  0,  0)),
}

# ── Global state — arm ──
arm_keys: set[str] = set()
last_arm_key_time: float = 0.0

robot = None
lock = threading.Lock()
x_des_pos = None
x_des_rot = None
gripper_des = 0.0
gripper_min = GRIPPER_MIN_DEFAULT
gripper_max = GRIPPER_MAX_DEFAULT
ctrl_enabled = threading.Event()
stop_event = threading.Event()
init_joints = None

jl_lower = None
jl_upper = None
joint_limit_warn = ""
joint_limit_warn_time = 0.0
last_feasible_pos = None
last_feasible_rot = None
wrench_bias = np.zeros(6)

tele_data_raw = {}  # raw numpy from control thread
tele_data = {}      # formatted, for wrench bias read
arm_available = False  # set True only if Panthera init succeeds

# ── Global state — ODrive ──
base_keys: set[str] = set()
last_base_key_time: float = 0.0
linear_speed: float = LINEAR_SPEED
angular_speed: float = ANGULAR_SPEED
# Runtime-tunable arm EE speeds (UI sliders)
arm_lin_speed: float = ARM_LIN_SPEED
arm_ang_speed: float = ARM_ANG_SPEED
odrv = None
axis_l = None
axis_r = None

# ── Shared ──
telemetry_clients: list[WebSocket] = []


# ===================== MATH UTILS =====================

def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def orientation_error_axis_angle(R_des, R_cur):
    R_err = R_cur.T @ R_des
    rot = Rot.from_matrix(R_err)
    rotvec = rot.as_rotvec()
    return R_cur @ rotvec


def _build_pin_q(robot, joint_angles):
    q = np.zeros(robot.model.nq)
    for i, name in enumerate(robot.joint_names):
        jid = robot.model.getJointId(name)
        q[robot.model.joints[jid].idx_q] = joint_angles[i]
    return q


def compute_fk_and_jacobian(robot, data, joint_angles):
    q = _build_pin_q(robot, joint_angles)
    pin.computeJointJacobians(robot.model, data, q)

    last_jid = robot.model.getJointId(robot.joint_names[-1])
    T_last = data.oMi[last_jid]
    R_last = T_last.rotation
    p_last = T_last.translation

    r_world = R_last @ TOOL_OFFSET
    p_tcp = (p_last + r_world).copy()
    R_tcp = R_last.copy()

    J_full = pin.getJointJacobian(
        robot.model, data, last_jid,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    J_tcp = J_full.copy()
    J_tcp[:3, :] -= skew(r_world) @ J_full[3:, :]

    cols = [robot.model.joints[robot.model.getJointId(n)].idx_v
            for n in robot.joint_names]
    J6 = J_tcp[:, cols]

    return p_tcp, R_tcp, J6


def print_pose(pos, rot, label="end-effector pose"):
    euler = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
    log.info("[%s] pos=[%.4f, %.4f, %.4f] rot=[%.1f, %.1f, %.1f]",
             label, *pos, *euler)


# ===================== INIT =====================

def init_robot():
    global robot, x_des_pos, x_des_rot, gripper_des, gripper_min, gripper_max
    global ctrl_enabled, stop_event, init_joints
    global jl_lower, jl_upper, last_feasible_pos, last_feasible_rot
    global arm_available

    # Panthera library segfaults if arm is not connected — probe in subprocess first
    import multiprocessing as _mp
    def _probe_arm():
        try:
            Panthera()
        except Exception:
            pass  # exit cleanly if Python exception
    probe = _mp.Process(target=_probe_arm)
    probe.start()
    probe.join(timeout=15)
    if probe.is_alive():
        probe.kill()
        probe.join()
        log.warning("Arm probe timed out — running base-only mode")
        arm_available = False
        return
    if probe.exitcode < 0:
        # Negative = killed by signal (e.g. -11 = SIGSEGV). Positive exit codes
        # are fine — Panthera destructor often causes exit(1) on cleanup.
        log.warning("Arm probe killed by signal %d — running base-only mode", -probe.exitcode)
        arm_available = False
        return

    # Probe succeeded, now init for real in this process
    try:
        robot = Panthera()
    except Exception as e:
        log.warning("Arm init failed, running base-only mode: %s", e)
        arm_available = False
        return

    # vel mode init: just go to safe joint pose. No HOME_POS IK / moveL.
    log.info("Arm: moving to safe joint pose %s...", SAFE_JOINT_POS)
    robot.Joint_Pos_Vel(SAFE_JOINT_POS, JOINT_VEL, MAX_TORQUE, iswait=True)
    time.sleep(0.5)
    init_joints = list(SAFE_JOINT_POS)

    gripper_min = robot.gripper_limits['lower'] if robot.gripper_limits else GRIPPER_MIN_DEFAULT
    gripper_max = robot.gripper_limits['upper'] if robot.gripper_limits else GRIPPER_MAX_DEFAULT
    log.info("Gripper limits: [%.2f, %.2f] rad", gripper_min, gripper_max)

    jl_lower = np.array(robot.joint_limits['lower']) + JOINT_LIMIT_MARGIN
    jl_upper = np.array(robot.joint_limits['upper']) - JOINT_LIMIT_MARGIN

    # Keep legacy globals around (telemetry / reset code reads them) but
    # they're no longer driven by an impedance target.
    q0 = np.array(robot.get_current_pos())
    fk0 = robot.forward_kinematics()
    p0 = np.array(fk0['position'])
    R0 = np.array(fk0['rotation'])
    x_des_pos = p0.copy()
    x_des_rot = R0.copy()
    last_feasible_pos = p0.copy()
    last_feasible_rot = R0.copy()
    gripper_des = float(np.clip(robot.get_current_pos_gripper(), gripper_min, gripper_max))

    log.info("Arm ready at safe pose. Cur EE pos: [%.3f, %.3f, %.3f]",
             p0[0], p0[1], p0[2])

    arm_available = True
    ctrl_enabled.set()
    return p0, R0


def init_odrive():
    global odrv, axis_l, axis_r
    import odrive
    import odrive.enums as enums

    log.info("ODrive: searching (timeout 30s)...")
    if ODRIVE_SERIAL:
        odrv = odrive.find_any(serial_number=ODRIVE_SERIAL, timeout=30)
    else:
        odrv = odrive.find_any(timeout=30)

    axis_l = getattr(odrv, f"axis{AXIS_LEFT}")
    axis_r = getattr(odrv, f"axis{AXIS_RIGHT}")

    axis_l.clear_errors()
    axis_r.clear_errors()
    time.sleep(0.1)

    axis_l.controller.config.control_mode = enums.CONTROL_MODE_VELOCITY_CONTROL
    axis_r.controller.config.control_mode = enums.CONTROL_MODE_VELOCITY_CONTROL
    axis_l.controller.config.input_mode = enums.INPUT_MODE_PASSTHROUGH
    axis_r.controller.config.input_mode = enums.INPUT_MODE_PASSTHROUGH

    axis_l.config.enable_watchdog = False
    axis_r.config.enable_watchdog = False

    axis_l.requested_state = enums.AXIS_STATE_CLOSED_LOOP_CONTROL
    axis_r.requested_state = enums.AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.5)

    log.info("ODrive connected! Left=axis%d, Right=axis%d", AXIS_LEFT, AXIS_RIGHT)


def disconnect_odrive():
    global odrv, axis_l, axis_r
    if axis_l is None:
        return
    import odrive.enums as enums
    axis_l.controller.input_vel = 0
    axis_r.controller.input_vel = 0
    axis_l.requested_state = enums.AXIS_STATE_IDLE
    axis_r.requested_state = enums.AXIS_STATE_IDLE
    odrv = None
    axis_l = None
    axis_r = None
    log.info("ODrive disconnected.")


# ===================== INSIGHT 9 (ROS 2) + VISER =====================
# Replaces the D405/stereo V4L2 camera pipeline of 8_arm_base_web_control.py
# with Insight 9's ROS topics: VIO pose drives a viser frustum, and
# /camera/.../color/.../compressed feeds the right-rail RGB stream.

# Insight 9 shared state — written by ROS callbacks, read by viser updater
# and the MJPEG generator.
insight_lock = threading.Lock()
last_pose = None             # ((x,y,z), (qw,qx,qy,qz))
last_pose_time = 0.0
last_status = "(no vio status yet)"
last_image_jpeg = b""
last_image_time = 0.0


class InsightSubs(Node):
    """Subscribes to Insight 9 VIO + status + RGB-compressed."""

    def __init__(self):
        super().__init__("integrated_demo_subs")
        cb = ReentrantCallbackGroup()
        rel_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        sen_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=2)
        self.create_subscription(PoseStamped, TOPIC_POSE,
                                 self._on_pose, rel_qos, callback_group=cb)
        self.create_subscription(String, TOPIC_STATUS,
                                 self._on_status, rel_qos, callback_group=cb)
        self.create_subscription(CompressedImage, TOPIC_RGB,
                                 self._on_rgb, sen_qos, callback_group=cb)

    def _on_pose(self, msg: PoseStamped):
        global last_pose, last_pose_time
        p, q = msg.pose.position, msg.pose.orientation
        xyz  = (p.x, p.y, p.z)
        wxyz = (q.w, q.x, q.y, q.z)
        with insight_lock:
            last_pose = (xyz, wxyz)
            last_pose_time = time.time()
        # Push pose to viser inline — avoids waking a separate 50Hz thread.
        # viser handle property assignments are thread-safe (just enqueue).
        if _viser_frustum is not None:
            try:
                _viser_frustum.position = xyz
                _viser_frustum.wxyz = wxyz
            except Exception:
                pass

    def _on_status(self, msg: String):
        global last_status
        with insight_lock:
            last_status = msg.data

    def _on_rgb(self, msg: CompressedImage):
        global last_image_jpeg, last_image_time
        with insight_lock:
            last_image_jpeg = bytes(msg.data)
            last_image_time = time.time()


# Viser handles set up by build_viser_scene().
_viser_server = None
_viser_frustum = None


def build_viser_scene():
    """Build the viser scene: world axes, floor grid, frustum (no static mesh)."""
    global _viser_server, _viser_frustum
    vs = viser.ViserServer(host="0.0.0.0", port=VISER_PORT)
    vs.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.005)
    vs.scene.add_grid(
        "/floor", width=GRID_SIZE_M, height=GRID_SIZE_M, plane="xy",
        cell_size=GRID_CELL_M,    cell_color=(60, 60, 75),    cell_thickness=1.0,
        section_size=GRID_SECTION_M, section_color=(110, 110, 140), section_thickness=2.0,
        position=(0.0, 0.0, GRID_Z),
    )
    _viser_frustum = vs.scene.add_camera_frustum(
        "/cam", fov=FRUSTUM_FOV_RAD, aspect=FRUSTUM_ASPECT,
        scale=FRUSTUM_SCALE, color=(255, 180, 0),
    )
    _viser_server = vs
    log.info("viser server up on port %d", VISER_PORT)


def viser_updater_loop():
    """Trail-only updater at TRAIL_SAMPLE_HZ. Pose is pushed directly from
    the ROS pose callback (see InsightSubs._on_pose), so this thread is now
    a slow ticker and does NOT compete heavily with the impedance thread."""
    sample_interval = 1.0 / TRAIL_SAMPLE_HZ
    trail_pts: deque = deque(maxlen=TRAIL_MAX_POINTS)
    last_log = 0.0
    path_handle = None
    point_color = np.array([255, 120, 0], dtype=np.uint8)

    while not stop_event.is_set():
        now = time.time()
        with insight_lock:
            pose = last_pose

        if pose is not None and _viser_server is not None:
            xyz, _ = pose
            trail_pts.append(xyz)
            if len(trail_pts) >= 1:
                pts = np.asarray(trail_pts, dtype=np.float32)
                colors = np.tile(point_color, (len(pts), 1))
                try:
                    if path_handle is not None:
                        path_handle.remove()
                    path_handle = _viser_server.scene.add_point_cloud(
                        "/trail", points=pts, colors=colors,
                        point_size=0.025, point_shape="circle",
                    )
                except Exception as e:
                    log.warning("viser trail update failed: %s", e)
                    path_handle = None

        if now - last_log >= 10.0:
            log.info("viser trail: %d pts (cap %d)", len(trail_pts), TRAIL_MAX_POINTS)
            last_log = now
        time.sleep(sample_interval)


def insight_rgb_mjpeg_generator():
    """Pass-through MJPEG: yields the latest CompressedImage JPEG bytes
    direct from S.last_image_jpeg. Zero decode, zero re-encode."""
    interval = 1.0 / RGB_STREAM_FPS
    last_sent = 0.0
    while not stop_event.is_set():
        with insight_lock:
            jpeg = last_image_jpeg
            jt = last_image_time
        if jpeg and jt > last_sent:
            last_sent = jt
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(interval)


# ROS executor / node refs (set in lifespan)
_ros_node: Node | None = None
_ros_executor: MultiThreadedExecutor | None = None
_ros_thread: threading.Thread | None = None


def start_ros():
    """Initialize rclpy, create the InsightSubs node, spin in a daemon thread."""
    global _ros_node, _ros_executor, _ros_thread
    rclpy.init()
    _ros_node = InsightSubs()
    _ros_executor = MultiThreadedExecutor(num_threads=1)
    _ros_executor.add_node(_ros_node)
    _ros_thread = threading.Thread(target=_ros_executor.spin, daemon=True)
    _ros_thread.start()
    log.info("ROS 2 spin thread started; subscribed to %s, %s, %s",
             TOPIC_POSE, TOPIC_STATUS, TOPIC_RGB)


def stop_ros():
    """Shut down rclpy cleanly."""
    if _ros_executor is not None:
        try:
            _ros_executor.shutdown()
        except Exception:
            pass
    if _ros_node is not None:
        try:
            _ros_node.destroy_node()
        except Exception:
            pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


# ===================== ODRIVE CONTROL =====================

def send_velocity(x_vel: float, theta_vel: float):
    theta_rad = math.radians(theta_vel)
    half_wb = WHEEL_BASE / 2.0
    left_linear = x_vel - theta_rad * half_wb
    right_linear = x_vel + theta_rad * half_wb

    if INVERT_LEFT:
        left_linear = -left_linear
    if INVERT_RIGHT:
        right_linear = -right_linear

    circumference = 2.0 * math.pi * WHEEL_RADIUS
    axis_l.controller.input_vel = left_linear / circumference
    axis_r.controller.input_vel = right_linear / circumference


def _odrv_safe(obj, *path, default=None):
    try:
        cur = obj
        for k in path:
            cur = getattr(cur, k)
        return cur
    except Exception:
        return default


# Peak DC current tracker (since process start)
_peak_ibus = 0.0


def read_odrive_telemetry() -> dict:
    global _peak_ibus
    try:
        l_cur = axis_l.motor.current_control.Iq_measured
        r_cur = axis_r.motor.current_control.Iq_measured
        l_vel = axis_l.encoder.vel_estimate
        r_vel = axis_r.encoder.vel_estimate
        l_torque = l_cur * TORQUE_CONSTANT
        r_torque = r_cur * TORQUE_CONSTANT
        vbus = odrv.vbus_voltage
        ibus = _odrv_safe(odrv, "ibus")
        if ibus is not None:
            _peak_ibus = max(_peak_ibus, abs(float(ibus)))
        dc_power = (vbus * ibus) if ibus is not None else None
        batt_raw = max(0.0, min(100.0, (vbus - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100.0))
        batt_pct = round(batt_raw / 5.0) * 5
        return {
            "vbus": round(vbus, 2),
            "ibus": None if ibus is None else round(float(ibus), 3),
            "dc_power": None if dc_power is None else round(dc_power, 2),
            "peak_ibus": round(_peak_ibus, 3),
            "brake_armed": _odrv_safe(odrv, "brake_resistor_armed"),
            "brake_sat":   _odrv_safe(odrv, "brake_resistor_saturated"),
            "batt_pct": int(batt_pct),
            "left": {
                "current_A": round(l_cur, 3),
                "torque_Nm": round(l_torque, 4),
                "vel_rps": round(l_vel, 3),
                "vel_mps": round(l_vel * 2 * math.pi * WHEEL_RADIUS, 3),
            },
            "right": {
                "current_A": round(r_cur, 3),
                "torque_Nm": round(r_torque, 4),
                "vel_rps": round(r_vel, 3),
                "vel_mps": round(r_vel * 2 * math.pi * WHEEL_RADIUS, 3),
            },
        }
    except Exception as e:
        return {"error": str(e)}


async def odrive_control_loop():
    interval = 1.0 / ODRIVE_CMD_HZ
    while not stop_event.is_set():
        now = time.monotonic()
        if axis_l is not None and base_keys and (now - last_base_key_time < IDLE_TIMEOUT):
            x = 0.0
            theta = 0.0
            # Forward/back swapped per user preference: ↑ = backward, ↓ = forward.
            if "up" in base_keys:
                x -= linear_speed
            if "down" in base_keys:
                x += linear_speed
            if "left" in base_keys:
                theta += angular_speed
            if "right" in base_keys:
                theta -= angular_speed
            send_velocity(x, theta)
        elif axis_l is not None:
            send_velocity(0.0, 0.0)
        await asyncio.sleep(interval)


# ===================== ARM VEL CONTROL LOOP (thread) =====================
# Cartesian velocity control via damped Jacobian pseudo-inverse + motor-level
# velocity tracking. Replaces the impedance controller — much more stable on
# this hardware (no derivative-feedback amplification, no high-gain spring).
# Modeled after panthera_python/scripts/7_keyboard_cartesian_vel_control.py.

def vel_control_loop():
    global gripper_des

    if not arm_available:
        return

    dt = 1.0 / VEL_CTRL_HZ
    actual_vel = np.zeros(6)          # smoothed tool-frame velocity 6D

    _iter_count = 0
    _iter_window_t0 = time.time()
    _LOOP_REPORT_PERIOD = 2.0

    while not stop_event.is_set():
        t0 = time.time()

        if ctrl_enabled.is_set():
            try:
                # 1. Build target velocity in tool frame from currently held keys.
                target_vel = np.zeros(6)
                if arm_keys and (time.monotonic() - last_arm_key_time < IDLE_TIMEOUT):
                    for k in list(arm_keys):
                        if k in VEL_KEY_MAP_ARM:
                            idx, sign = VEL_KEY_MAP_ARM[k]
                            speed = arm_lin_speed if idx < 3 else arm_ang_speed
                            target_vel[idx] += sign * speed

                # 2. Exponential smoothing → soft accel/decel.
                actual_vel += (target_vel - actual_vel) * ARM_ACCEL_FACTOR

                # 3. Joint state + FK + Jacobian.
                q = np.array(robot.get_current_pos())
                fk = robot.forward_kinematics()
                p_cur = np.array(fk['position'])
                R_tool = np.array(fk['rotation'])
                J = robot.get_jacobian(q)

                # 4. Manipulability-based adaptive damping (handles singularities).
                try:
                    manip = float(robot.get_manipulability(q))
                except Exception:
                    manip = 1.0
                if manip < ARM_MANIP_THRESH:
                    damping = ARM_DAMPING_BASE * (1.0 + (ARM_MANIP_THRESH - manip) * 100.0)
                    vel_used = actual_vel * (manip / ARM_MANIP_THRESH)
                else:
                    damping = ARM_DAMPING_BASE
                    vel_used = actual_vel

                # 5. Tool frame → world frame.
                vel_world = np.zeros(6)
                vel_world[0:3] = R_tool @ vel_used[0:3]
                vel_world[3:6] = R_tool @ vel_used[3:6]

                # 6. Damped pseudo-inverse → joint velocity.
                J_damp = Panthera.compute_damped_pseudoinverse(J, damping)
                q_dot = J_damp @ vel_world
                q_dot = np.clip(q_dot, -ARM_JOINT_VEL_MAX, ARM_JOINT_VEL_MAX)

                # 7. Joint limit guard: zero out velocity pushing past limits.
                lower_close = q < jl_lower
                upper_close = q > jl_upper
                q_dot[lower_close & (q_dot < 0)] = 0.0
                q_dot[upper_close & (q_dot > 0)] = 0.0
                limit_violated = lower_close | upper_close
                if np.any(limit_violated):
                    parts = []
                    for i in range(len(q)):
                        if limit_violated[i]:
                            side = "lower" if q[i] < jl_lower[i] else "upper"
                            parts.append(f"J{i+1}{side}({q[i]:+.2f})")
                    with lock:
                        global joint_limit_warn, joint_limit_warn_time
                        joint_limit_warn = " ".join(parts)
                        joint_limit_warn_time = time.time()

                # 8. Gripper (Z/X) — accumulate while key held, then send as a
                #    standalone MIT-mode USB-CDC frame. We CANNOT batch this with
                #    Joint_Vel: the firmware TX buffer is single-mode, and switching
                #    modes wipes any pending slots (so Joint_Vel would overwrite the
                #    gripper command if both were queued in the same frame).
                if "z" in arm_keys or "x" in arm_keys:
                    if "z" in arm_keys:
                        gripper_des = max(gripper_min, gripper_des - GRIPPER_STEP)
                    if "x" in arm_keys:
                        gripper_des = min(gripper_max, gripper_des + GRIPPER_STEP)
                robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                    gripper_des, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)
                robot.motor_send_cmd()   # flush MIT frame to gripper

                # 9. Arm velocity (this switches buffer to MODE_VELOCITY and flushes
                #    its own frame internally).
                robot.Joint_Vel(q_dot.tolist())

                # 10. Telemetry snapshot (keep shape compatible with old code).
                tele_data_raw.update({
                    "e_pos":  np.zeros(3),       # no position error in vel mode
                    "e_rot":  np.zeros(3),
                    "p_cur":  p_cur.copy(),
                    "p_des":  p_cur.copy(),      # target == current (continuous)
                    "wrench": np.zeros(6),       # no F/T estimator in vel mode
                    "g_des":  gripper_des,
                    "g_actual": robot.get_current_pos_gripper(),
                    "q":      q.copy(),
                    "dq":     actual_vel.copy(), # show commanded EE velocity instead
                    "tor":    q_dot.copy(),      # show q_dot here for visibility
                    "manip":  manip,
                })
            except Exception as e:
                log.warning("vel_control_loop iteration error: %s", e)

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        # Loop-rate monitor (lower expected rate, much less GIL pressure)
        _iter_count += 1
        _now_w = time.time()
        if _now_w - _iter_window_t0 >= _LOOP_REPORT_PERIOD:
            hz = _iter_count / (_now_w - _iter_window_t0)
            log.info("[vel_ctrl] actual rate: %.1f Hz (nominal %d)",
                     hz, VEL_CTRL_HZ)
            _iter_count = 0
            _iter_window_t0 = _now_w

    # On loop exit: zero joint velocities so arm doesn't keep moving.
    try:
        robot.Joint_Vel([0.0] * robot.motor_count)
        log.info("vel_control_loop stopped — joint velocities zeroed")
    except Exception:
        pass


# ===================== RESET =====================

def do_reset():
    global gripper_des
    if not arm_available:
        log.warning("Reset ignored — arm not available")
        return
    ctrl_enabled.clear()
    log.info("Resetting arm to safe joint pose %s ...", SAFE_JOINT_POS)
    # vel mode: just go back to the safe joint pose
    robot.Joint_Vel([0.0] * robot.motor_count)
    robot.Joint_Pos_Vel(SAFE_JOINT_POS, JOINT_VEL, MAX_TORQUE, iswait=True)

    q_now = np.array(robot.get_current_pos())
    ctrl_data = robot.model.createData()
    p_now, R_now, _ = compute_fk_and_jacobian(robot, ctrl_data, q_now)
    with lock:
        x_des_pos[:] = p_now
        x_des_rot[:] = R_now
    print_pose(p_now, R_now, "Reset complete")
    ctrl_enabled.set()


# ===================== TELEMETRY =====================

async def telemetry_loop():
    global tele_data
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if not telemetry_clients:
            await asyncio.sleep(interval)
            continue

        combined = {}

        # Arm telemetry (format raw data on event loop)
        raw = tele_data_raw.copy()
        if raw:
            e_pos = raw["e_pos"]
            e_rot = raw["e_rot"]
            p_cur = raw["p_cur"]
            p_des = raw["p_des"]
            wrench = raw["wrench"]
            g_des = raw["g_des"]
            polar_r = np.hypot(p_des[0], p_des[1])
            polar_theta = np.degrees(np.arctan2(p_des[1], p_des[0]))
            g_pct = (g_des - gripper_min) / max(gripper_max - gripper_min, 1e-6) * 100

            arm_data = {
                "err_pos_mm": [round(e_pos[0]*1000, 1), round(e_pos[1]*1000, 1), round(e_pos[2]*1000, 1)],
                "err_rot_deg": [round(np.degrees(e_rot[0]), 1), round(np.degrees(e_rot[1]), 1), round(np.degrees(e_rot[2]), 1)],
                "err_pos_norm": round(np.linalg.norm(e_pos)*1000, 1),
                "err_rot_norm": round(np.linalg.norm(np.degrees(e_rot)), 1),
                "ee_pos": [round(p_cur[0], 4), round(p_cur[1], 4), round(p_cur[2], 4)],
                "ee_des": [round(p_des[0], 4), round(p_des[1], 4), round(p_des[2], 4)],
                "polar_r": round(polar_r, 4),
                "polar_theta": round(polar_theta, 1),
                "polar_z": round(p_des[2], 4),
                "f_ext": [round(wrench[0], 2), round(wrench[1], 2), round(wrench[2], 2)],
                "m_ext": [round(wrench[3], 2), round(wrench[4], 2), round(wrench[5], 2)],
                "f_norm": round(np.linalg.norm(wrench[:3]), 2),
                "gripper_des": round(g_des, 3),
                "gripper_act": round(raw["g_actual"], 3),
                "gripper_pct": round(g_pct, 0),
                "joint_pos": [round(x, 4) for x in raw["q"].tolist()],
                "joint_vel": [round(x, 4) for x in raw["dq"].tolist()],
                "joint_torque": [round(x, 4) for x in raw["tor"].tolist()],
                "ctrl_enabled": ctrl_enabled.is_set(),
                "joint_limit_warn": joint_limit_warn if (time.time() - joint_limit_warn_time) < JOINT_LIMIT_WARN_DURATION else "",
                "settings": {
                    "lin_speed": round(arm_lin_speed, 3),
                    "ang_speed": round(arm_ang_speed, 3),
                },
            }
            tele_data = arm_data
            combined["arm"] = arm_data

        # ODrive telemetry
        if axis_l is not None:
            combined["base"] = read_odrive_telemetry()
            combined["base"]["settings"] = {
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
            }

        # Insight 9 VIO snapshot (no heavy computation)
        with insight_lock:
            vio_status_snap = last_status
            vio_t = last_pose_time
            vio_pos = last_pose[0] if last_pose is not None else None
        age = (time.time() - vio_t) if vio_t > 0 else None
        combined["vio"] = {
            "status": vio_status_snap[:120],
            "age_s":  None if age is None else round(age, 2),
            "pos":    None if vio_pos is None else
                      [round(vio_pos[0], 4), round(vio_pos[1], 4), round(vio_pos[2], 4)],
            "has_recent_pose": age is not None and age < 0.5,
        }

        combined["arm_available"] = arm_available
        if combined:
            msg = json.dumps(combined)
            dead = []
            for ws in telemetry_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                telemetry_clients.remove(ws)

        await asyncio.sleep(interval)


# ===================== LIFECYCLE =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()

    # Insight 9: bring up ROS first so VIO frame is flowing before viser starts.
    start_ros()
    build_viser_scene()
    t_viser = threading.Thread(target=viser_updater_loop, daemon=True)
    t_viser.start()

    # Robot (Panthera) and ODrive
    await loop.run_in_executor(None, init_robot)
    await loop.run_in_executor(None, init_odrive)

    t_ctrl = None
    if arm_available:
        t_ctrl = threading.Thread(target=vel_control_loop, daemon=True)
        t_ctrl.start()
    else:
        log.info("Arm unavailable — starting base-only mode")

    t_base = asyncio.create_task(odrive_control_loop())
    t_tele = asyncio.create_task(telemetry_loop())

    yield

    stop_event.set()
    t_base.cancel()
    t_tele.cancel()
    if t_ctrl:
        t_ctrl.join(timeout=2)
    stop_ros()
    await loop.run_in_executor(None, disconnect_odrive)
    log.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)


# ===================== HTML =====================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insight 9 + Panthera + ODrive</title>
<style>
  /* ===== Apple-style design tokens ===== */
  :root {
    --bg-0: #0c0c10;
    --bg-1: #131318;
    --glass: rgba(28, 28, 32, 0.72);
    --glass-strong: rgba(38, 38, 44, 0.84);
    --hairline: rgba(255, 255, 255, 0.08);
    --hairline-strong: rgba(255, 255, 255, 0.14);

    --fg: #f2f2f7;
    --fg-2: rgba(242, 242, 247, 0.62);
    --fg-3: rgba(242, 242, 247, 0.38);

    --blue:   #0a84ff;
    --green:  #30d158;
    --red:    #ff453a;
    --orange: #ff9f0a;
    --yellow: #ffd60a;
    --purple: #bf5af2;
    --teal:   #5ac8fa;

    --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    --r-card: 14px;
    --r-pill: 999px;
    --pad: 16px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
    background: linear-gradient(180deg, #1a1a1f 0%, #0a0a0d 100%);
    color: var(--fg);
    user-select: none;
    overflow: hidden;
    font-feature-settings: "ss01", "tnum", "case";
    -webkit-font-smoothing: antialiased;
  }

  /* ===== Header ===== */
  header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 50;
    height: 56px;
    display: flex; align-items: center; gap: 16px;
    padding: 0 20px;
    background: rgba(12, 12, 16, 0.7);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border-bottom: 1px solid var(--hairline);
  }
  .title {
    font-size: 15px; font-weight: 600; letter-spacing: -0.01em;
  }
  .title .mute { color: var(--fg-3); font-weight: 400; }
  .pills {
    margin-left: auto;
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .pill {
    padding: 5px 12px;
    border-radius: var(--r-pill);
    font-size: 12px; font-weight: 500;
    background: rgba(255, 255, 255, 0.06);
    color: var(--fg-2);
    border: 1px solid var(--hairline);
    transition: background 0.2s, color 0.2s;
    white-space: nowrap;
  }
  .pill.ok    { background: rgba(48, 209, 88, 0.16);  color: var(--green); border-color: transparent; }
  .pill.warn  { background: rgba(255, 159, 10, 0.16); color: var(--orange); border-color: transparent; }
  .pill.err   { background: rgba(255, 69, 58, 0.18);  color: var(--red); border-color: transparent; }
  .pill.muted { background: rgba(255, 255, 255, 0.04); color: var(--fg-3); }

  /* ===== Main grid ===== */
  main {
    position: absolute; top: 56px; left: 0; right: 0; bottom: 0;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 14px;
    padding: 14px;
  }

  .viser-pane {
    position: relative;
    border-radius: var(--r-card);
    overflow: hidden;
    background: var(--bg-1);
    border: 1px solid var(--hairline);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
  }
  #viser-frame {
    width: 100%; height: 100%;
    border: 0; background: #fff;
  }
  .overlay {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 12px;
    background: rgba(8, 8, 10, 0.76);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--fg);
    font-size: 18px; font-weight: 600; letter-spacing: 0.02em;
    transition: opacity 0.4s;
  }
  .overlay .sub { font-size: 13px; font-weight: 400; color: var(--fg-2); }
  .overlay.hidden { opacity: 0; pointer-events: none; }
  .overlay .spinner {
    width: 28px; height: 28px;
    border: 3px solid rgba(255, 255, 255, 0.14);
    border-top-color: var(--orange);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ===== Right rail ===== */
  aside {
    overflow-y: auto;
    overscroll-behavior: contain;
    padding-right: 4px;
    display: flex; flex-direction: column; gap: 14px;
  }
  aside::-webkit-scrollbar { width: 6px; }
  aside::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 3px; }

  .card {
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border: 1px solid var(--hairline);
    border-radius: var(--r-card);
    padding: var(--pad);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.32);
  }
  .card h2 {
    display: flex; align-items: baseline; gap: 8px;
    font-size: 13px; font-weight: 600;
    color: var(--fg);
    letter-spacing: -0.005em;
    margin-bottom: 12px;
  }
  .card h2 .hint {
    margin-left: auto;
    font-size: 11px; font-weight: 500;
    color: var(--fg-3);
    font-family: var(--mono);
    letter-spacing: 0;
  }

  /* ===== RGB stream — portrait native (1088×1920), no rotation ===== */
  .rgb-stream {
    display: block;
    width: 100%;
    max-height: 540px;
    object-fit: contain;
    border-radius: 10px;
    background: #000;
  }
  .rgb-stream.off { display: none; }

  /* Header row inside cards: title left + control right */
  .card-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
  }
  .card-head h2 { margin: 0; }

  /* Apple-style toggle switch */
  .switch { position: relative; display: inline-block; width: 42px; height: 24px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .switch .slider-track {
    position: absolute; cursor: pointer; inset: 0;
    background: rgba(120, 120, 128, 0.32);
    border-radius: 12px;
    transition: background 0.2s ease;
  }
  .switch .slider-track::before {
    position: absolute; content: ""; height: 20px; width: 20px;
    left: 2px; top: 2px; background: #fff; border-radius: 50%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
  }
  .switch input:checked + .slider-track { background: #30d158; }
  .switch input:checked + .slider-track::before { transform: translateX(18px); }

  /* ===== Metrics ===== */
  .metric {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }
  .metric:last-child { border-bottom: 0; }
  .metric .label { color: var(--fg-2); }
  .metric .value { font-family: var(--mono); color: var(--fg); font-weight: 500; }
  .metric .value.green  { color: var(--green); }
  .metric .value.orange { color: var(--orange); }
  .metric .value.red    { color: var(--red); }
  .metric .peak { color: var(--fg-3); font-size: 11px; margin-left: 6px; }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-left: 4px;
  }
  .badge.drive { background: rgba(48, 209, 88, 0.2);  color: var(--green); }
  .badge.regen { background: rgba(255, 69, 58, 0.18); color: var(--red); }
  .badge.ok    { background: rgba(48, 209, 88, 0.2);  color: var(--green); }
  .badge.warn  { background: rgba(255, 159, 10, 0.18); color: var(--orange); }
  .badge.muted { background: rgba(255, 255, 255, 0.06); color: var(--fg-3); }

  /* ===== Arrow keys ===== */
  .arrows {
    display: grid;
    grid-template-columns: 56px 56px 56px;
    grid-template-rows: 56px 56px;
    grid-template-areas: ". up ." "left down right";
    gap: 6px;
    justify-content: center;
    margin-bottom: 12px;
  }
  .arrow-key[data-dir="up"]    { grid-area: up; }
  .arrow-key[data-dir="left"]  { grid-area: left; }
  .arrow-key[data-dir="down"]  { grid-area: down; }
  .arrow-key[data-dir="right"] { grid-area: right; }

  .key, .arrow-key, .action-btn {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--hairline-strong);
    border-radius: 10px;
    color: var(--fg);
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
    transition: all 0.12s cubic-bezier(0.25, 1.4, 0.5, 1);
  }
  .key:hover, .arrow-key:hover, .action-btn:hover {
    background: rgba(255, 255, 255, 0.1);
  }
  .key.active, .arrow-key.active {
    background: var(--blue);
    border-color: transparent;
    color: white;
    transform: scale(1.04);
    box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.18);
  }
  .key:active, .arrow-key:active, .action-btn:active {
    transform: scale(0.96);
  }

  /* ===== Arm keypad ===== */
  .arm-keys {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 6px;
    margin-bottom: 10px;
  }
  .arm-keys .key { height: 44px; }
  .arm-keys .group-label {
    grid-column: 1 / -1;
    font-size: 10px; color: var(--fg-3);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 4px; margin-bottom: -2px;
  }

  /* ===== Sliders ===== */
  .slider-row {
    display: grid; grid-template-columns: 56px 1fr 60px;
    align-items: center; gap: 10px;
    margin-top: 8px;
    font-size: 11px;
  }
  .slider-row label { color: var(--fg-2); }
  .slider-row .val { font-family: var(--mono); color: var(--blue); text-align: right; }
  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%; height: 4px;
    background: rgba(255, 255, 255, 0.12);
    border-radius: 2px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: white;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    cursor: pointer;
  }

  /* ===== Action buttons ===== */
  .actions {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 6px; margin-top: 12px;
  }
  .action-btn {
    height: 36px;
    font-size: 12px; font-family: -apple-system, sans-serif;
    font-weight: 500;
  }

  /* ===== Gripper bar ===== */
  .gripper-bar {
    height: 6px; background: rgba(255, 255, 255, 0.08);
    border-radius: 3px; overflow: hidden;
    margin: 6px 0;
  }
  .gripper-fill {
    height: 100%; background: linear-gradient(90deg, var(--teal), var(--blue));
    transition: width 0.12s;
  }

  /* ===== Warning ===== */
  .warn-text {
    color: var(--red); font-size: 11px;
    font-family: var(--mono);
    padding: 6px 0;
    min-height: 20px;
    animation: pulse 0.8s ease-in-out infinite alternate;
  }
  .warn-text:empty { animation: none; }
  @keyframes pulse {
    from { opacity: 0.7; }
    to   { opacity: 1.0; }
  }

  /* ===== VIO line ===== */
  .vio-line {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--teal);
    word-break: break-all;
    padding: 4px 0;
  }
</style>
</head>
<body>

<header>
  <div class="title">Integrated Demo <span class="mute">— Insight 9 · Panthera · ODrive</span></div>
  <div class="pills">
    <span class="pill" id="pill-link">connecting…</span>
    <span class="pill" id="pill-rtt">— ms</span>
    <span class="pill" id="pill-arm">arm: ?</span>
    <span class="pill" id="pill-base">base: ?</span>
    <span class="pill" id="pill-vio">vio: ?</span>
  </div>
</header>

<main>
  <section class="viser-pane">
    <iframe id="viser-frame" src="about:blank"></iframe>
    <div class="overlay" id="arm-init-overlay">
      <div class="spinner"></div>
      <div>ARM INITIALIZING</div>
      <div class="sub">stay clear of the workspace</div>
    </div>
  </section>

  <aside>
    <div class="card">
      <div class="card-head">
        <h2>Insight 9 RGB</h2>
        <label class="switch">
          <input type="checkbox" id="rgb-toggle" checked>
          <span class="slider-track"></span>
        </label>
      </div>
      <img class="rgb-stream" id="rgb-img" src="/rgb_stream" alt="Insight 9 RGB live">
    </div>

    <div class="card">
      <h2>Drive <span class="hint">↑ ↓ ← →</span></h2>
      <div class="arrows">
        <button class="arrow-key" data-dir="up">↑</button>
        <button class="arrow-key" data-dir="left">←</button>
        <button class="arrow-key" data-dir="down">↓</button>
        <button class="arrow-key" data-dir="right">→</button>
      </div>
      <div class="metric">
        <span class="label">Command</span>
        <span class="value" id="cmd-disp">x: 0.00 m/s &nbsp; θ: 0 °/s</span>
      </div>
      <div class="slider-row">
        <label>Linear</label>
        <input type="range" id="sl_lin" min="0.02" max="1.0" step="0.01" value="0.15">
        <span class="val" id="sl_lin_v">0.15</span>
      </div>
      <div class="slider-row">
        <label>Angular</label>
        <input type="range" id="sl_ang" min="5" max="180" step="1" value="30">
        <span class="val" id="sl_ang_v">30</span>
      </div>
    </div>

    <div class="card">
      <h2>Arm <span class="hint">WASD QE · IJKL UO · ZX</span></h2>
      <div class="arm-keys">
        <div class="group-label">Position (polar)</div>
        <button class="key" data-key="q">Q</button>
        <button class="key" data-key="w">W</button>
        <button class="key" data-key="e">E</button>
        <button class="key" data-key="a">A</button>
        <button class="key" data-key="s">S</button>
        <button class="key" data-key="d">D</button>
        <div class="group-label">Orientation (TCP)</div>
        <button class="key" data-key="u">U</button>
        <button class="key" data-key="i">I</button>
        <button class="key" data-key="o">O</button>
        <button class="key" data-key="j">J</button>
        <button class="key" data-key="k">K</button>
        <button class="key" data-key="l">L</button>
        <div class="group-label">Gripper</div>
        <button class="key" data-key="z">Z</button>
        <button class="key" data-key="x">X</button>
      </div>
      <div class="slider-row">
        <label>EE lin</label>
        <input type="range" id="sl_arm_lin" min="0.02" max="0.50" step="0.01" value="0.15">
        <span class="val" id="sl_arm_lin_v">0.15</span>
      </div>
      <div class="slider-row">
        <label>EE ang</label>
        <input type="range" id="sl_arm_ang" min="0.1" max="3.0" step="0.05" value="1.0">
        <span class="val" id="sl_arm_ang_v">1.00</span>
      </div>
      <div class="metric"><span class="label">EE polar</span>
        <span class="value"><span id="ee-r">--</span>m  <span id="ee-th">--</span>°  <span id="ee-z">--</span>m</span></div>
      <div class="metric"><span class="label">Pos err</span>
        <span class="value" id="err-pos-val">-- mm</span></div>
      <div class="metric"><span class="label">Rot err</span>
        <span class="value" id="err-rot-val">-- °</span></div>
      <div class="metric"><span class="label">F<sub>ext</sub></span>
        <span class="value" id="f-ext">-- N</span></div>
      <div class="metric"><span class="label">Gripper</span>
        <span class="value" id="grip-pct">--%</span></div>
      <div class="gripper-bar"><div class="gripper-fill" id="grip-fill" style="width:0%"></div></div>
      <div class="warn-text" id="warn-jl"></div>
      <div class="actions">
        <button id="btn_reset" class="action-btn">↺ Home</button>
        <button id="btn_zero"  class="action-btn">⌖ Zero F/T</button>
        <button id="btn_print" class="action-btn">⎙ Print</button>
      </div>
    </div>

    <div class="card">
      <h2>Power</h2>
      <div class="metric"><span class="label">Battery</span>
        <span class="value" id="batt-pct">--</span></div>
      <div class="metric"><span class="label">DC bus V</span>
        <span class="value" id="vbus">--</span></div>
      <div class="metric"><span class="label">DC bus I</span>
        <span class="value">
          <span id="ibus">--</span>
          <span class="peak" id="ibus-peak">(peak --)</span>
          <span id="ibus-badge"></span>
        </span></div>
      <div class="metric"><span class="label">DC power</span>
        <span class="value" id="dc-power">--</span></div>
      <div class="metric"><span class="label">Brake R</span>
        <span class="value" id="brake-state"></span></div>
    </div>

    <div class="card">
      <h2>Motors</h2>
      <div class="metric"><span class="label">L cur / vel</span>
        <span class="value" id="motor-l">--</span></div>
      <div class="metric"><span class="label">L torque</span>
        <span class="value" id="motor-l-tq">--</span></div>
      <div class="metric"><span class="label">R cur / vel</span>
        <span class="value" id="motor-r">--</span></div>
      <div class="metric"><span class="label">R torque</span>
        <span class="value" id="motor-r-tq">--</span></div>
    </div>

    <div class="card">
      <h2>VIO Status</h2>
      <div class="vio-line" id="vio-line">—</div>
      <div class="metric"><span class="label">Pose age</span>
        <span class="value" id="vio-age">—</span></div>
      <div class="metric"><span class="label">World pos</span>
        <span class="value" id="vio-pos">—</span></div>
    </div>
  </aside>
</main>

<script>
  // ===== config =====
  const VISER_PORT = __VISER_PORT__;
  document.getElementById("viser-frame").src =
    `${location.protocol}//${location.hostname}:${VISER_PORT}/`;

  const ARM_KEYS = new Set(["w","s","a","d","q","e","i","k","j","l","u","o","z","x"]);
  const BASE_MAP = {ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right"};

  let armPressed = new Set(), basePressed = new Set();
  let linSpeed = 0.15, angSpeed = 30;
  let ctrlWs = null, teleWs = null;

  // ===== pills =====
  function setPill(id, text, mood) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = "pill " + (mood || "");
  }

  // ===== WebSocket =====
  function connectCtrl() {
    ctrlWs = new WebSocket(`ws://${location.host}/ws`);
    ctrlWs.onopen    = () => {
      setPill("pill-link", "online", "ok");
      // 1Hz ping → RTT pill
      if (window._pingTimer) clearInterval(window._pingTimer);
      window._pingTimer = setInterval(() => {
        if (ctrlWs && ctrlWs.readyState === 1) {
          ctrlWs.send(JSON.stringify({type: "ping", t: performance.now()}));
        }
      }, 1000);
    };
    ctrlWs.onclose   = () => { setPill("pill-link", "offline", "err"); setTimeout(connectCtrl, 1000); };
    ctrlWs.onerror   = () => ctrlWs.close();
    ctrlWs.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.type === "pong") {
        const rtt = performance.now() - (m.t || 0);
        const mood = rtt < 30 ? "ok" : rtt < 100 ? "warn" : "err";
        setPill("pill-rtt", rtt.toFixed(0) + " ms", mood);
      }
    };
  }
  function connectTele() {
    teleWs = new WebSocket(`ws://${location.host}/ws/tele`);
    teleWs.onmessage = ev => render(JSON.parse(ev.data));
    teleWs.onclose   = () => setTimeout(connectTele, 1000);
    teleWs.onerror   = () => teleWs.close();
  }
  connectCtrl();
  connectTele();

  function sendMsg(o) { if (ctrlWs && ctrlWs.readyState === 1) ctrlWs.send(JSON.stringify(o)); }
  function sendArmKeys()  { sendMsg({type:"arm_keys",  keys:[...armPressed]});  paintActive(); updateCmdDisp(); }
  function sendBaseKeys() { sendMsg({type:"base_keys", keys:[...basePressed]}); paintActive(); updateCmdDisp(); }

  function paintActive() {
    document.querySelectorAll(".key").forEach(el =>
      el.classList.toggle("active", armPressed.has(el.dataset.key)));
    document.querySelectorAll(".arrow-key").forEach(el =>
      el.classList.toggle("active", basePressed.has(el.dataset.dir)));
  }
  function updateCmdDisp() {
    let x=0, th=0;
    if (basePressed.has("up"))    x  += linSpeed;
    if (basePressed.has("down"))  x  -= linSpeed;
    if (basePressed.has("left"))  th += angSpeed;
    if (basePressed.has("right")) th -= angSpeed;
    document.getElementById("cmd-disp").innerHTML =
      `x: ${x.toFixed(2)} m/s &nbsp; θ: ${th.toFixed(0)} °/s`;
  }

  let hbId = null;
  function startHB() {
    if (hbId) return;
    hbId = setInterval(() => {
      if (armPressed.size > 0)  sendArmKeys();
      if (basePressed.size > 0) sendBaseKeys();
      if (armPressed.size === 0 && basePressed.size === 0) {
        clearInterval(hbId); hbId = null;
      }
    }, 50);
  }

  // ===== keyboard =====
  document.addEventListener("keydown", e => {
    const k = e.key.toLowerCase();
    if (k === " ") { e.preventDefault(); sendMsg({type:"action", action:"print"});   return; }
    if (k === "r") { e.preventDefault(); sendMsg({type:"action", action:"reset"});   return; }
    if (k === "m") { e.preventDefault(); sendMsg({type:"action", action:"zero_ft"}); return; }
    const dir = BASE_MAP[e.key];
    if (dir) {
      e.preventDefault();
      if (!basePressed.has(dir)) { basePressed.add(dir); sendBaseKeys(); startHB(); }
      return;
    }
    if (ARM_KEYS.has(k)) {
      e.preventDefault();
      if (!armPressed.has(k)) { armPressed.add(k); sendArmKeys(); startHB(); }
    }
  });
  document.addEventListener("keyup", e => {
    const k = e.key.toLowerCase();
    const dir = BASE_MAP[e.key];
    if (dir) { e.preventDefault(); basePressed.delete(dir); sendBaseKeys(); return; }
    if (ARM_KEYS.has(k)) { e.preventDefault(); armPressed.delete(k); sendArmKeys(); }
  });

  // ===== touch / mouse =====
  function bindButton(el, onDown, onUp) {
    el.addEventListener("pointerdown", e => { e.preventDefault(); onDown(); });
    el.addEventListener("pointerup",   e => { e.preventDefault(); onUp(); });
    el.addEventListener("pointerleave",e => { onUp(); });
    el.addEventListener("pointercancel",e => { onUp(); });
  }
  document.querySelectorAll(".key").forEach(el => {
    const k = el.dataset.key;
    bindButton(el,
      () => { armPressed.add(k); sendArmKeys(); startHB(); },
      () => { if (armPressed.delete(k)) sendArmKeys(); });
  });
  document.querySelectorAll(".arrow-key").forEach(el => {
    const dir = el.dataset.dir;
    bindButton(el,
      () => { basePressed.add(dir); sendBaseKeys(); startHB(); },
      () => { if (basePressed.delete(dir)) sendBaseKeys(); });
  });

  document.getElementById("btn_reset").addEventListener("click", () =>
    sendMsg({type:"action", action:"reset"}));
  document.getElementById("btn_zero").addEventListener("click", () =>
    sendMsg({type:"action", action:"zero_ft"}));
  // RGB toggle: switch off → clear img.src so browser closes the MJPEG
  // connection, which lets the FastAPI generator end and stops the stream.
  const rgbToggle = document.getElementById("rgb-toggle");
  const rgbImg    = document.getElementById("rgb-img");
  rgbToggle.addEventListener("change", () => {
    if (rgbToggle.checked) {
      rgbImg.classList.remove("off");
      rgbImg.src = "/rgb_stream?" + Date.now();  // cache-bust to force new conn
    } else {
      rgbImg.classList.add("off");
      rgbImg.src = "";                            // closes the stream
    }
  });

  document.getElementById("btn_print").addEventListener("click", () =>
    sendMsg({type:"action", action:"print"}));

  // ===== sliders =====
  const slLin = document.getElementById("sl_lin");
  const slAng = document.getElementById("sl_ang");
  slLin.addEventListener("input", () => {
    linSpeed = parseFloat(slLin.value);
    document.getElementById("sl_lin_v").textContent = linSpeed.toFixed(2);
    sendMsg({type:"speed", linear:linSpeed, angular:angSpeed});
    updateCmdDisp();
  });
  slAng.addEventListener("input", () => {
    angSpeed = parseFloat(slAng.value);
    document.getElementById("sl_ang_v").textContent = angSpeed.toFixed(0);
    sendMsg({type:"speed", linear:linSpeed, angular:angSpeed});
    updateCmdDisp();
  });

  // arm EE speed sliders
  let armLin = 0.15, armAng = 1.0;
  const slArmLin = document.getElementById("sl_arm_lin");
  const slArmAng = document.getElementById("sl_arm_ang");
  slArmLin.addEventListener("input", () => {
    armLin = parseFloat(slArmLin.value);
    document.getElementById("sl_arm_lin_v").textContent = armLin.toFixed(2);
    sendMsg({type:"arm_speed", linear:armLin, angular:armAng});
  });
  slArmAng.addEventListener("input", () => {
    armAng = parseFloat(slArmAng.value);
    document.getElementById("sl_arm_ang_v").textContent = armAng.toFixed(2);
    sendMsg({type:"arm_speed", linear:armLin, angular:armAng});
  });

  // ===== telemetry render =====
  const fmt = (v, dp, unit) =>
    v == null ? "—" : (typeof v === "number" ? v.toFixed(dp) : v) + (unit ? " " + unit : "");

  function render(d) {
    // ----- arm pill + init overlay -----
    const armReady = d.arm_available && d.arm && d.arm.ctrl_enabled;
    if (!d.arm_available) {
      setPill("pill-arm", "arm: unavailable", "muted");
      document.getElementById("arm-init-overlay").classList.add("hidden");
    } else if (!armReady) {
      setPill("pill-arm", "arm: initializing", "warn");
      document.getElementById("arm-init-overlay").classList.remove("hidden");
    } else {
      setPill("pill-arm", "arm: ready", "ok");
      document.getElementById("arm-init-overlay").classList.add("hidden");
    }

    // ----- arm telemetry -----
    if (d.arm) {
      const a = d.arm;
      document.getElementById("ee-r").textContent  = fmt(a.polar_r, 3);
      document.getElementById("ee-th").textContent = fmt(a.polar_theta, 1);
      document.getElementById("ee-z").textContent  = fmt(a.polar_z, 3);

      const pe = a.err_pos_norm;
      const re = a.err_rot_norm;
      const epEl = document.getElementById("err-pos-val");
      epEl.textContent = fmt(pe, 1, "mm");
      epEl.className = "value " + (pe < 5 ? "green" : pe < 15 ? "orange" : "red");
      const erEl = document.getElementById("err-rot-val");
      erEl.textContent = fmt(re, 1, "°");
      erEl.className = "value " + (re < 3 ? "green" : re < 8 ? "orange" : "red");

      if (a.f_ext) {
        const fn = a.f_norm || 0;
        const ft = document.getElementById("f-ext");
        ft.textContent = `${fmt(fn, 2, "N")} (${fmt(a.f_ext[0],1)}, ${fmt(a.f_ext[1],1)}, ${fmt(a.f_ext[2],1)})`;
        ft.className = "value " + (fn < 1 ? "" : fn < 5 ? "orange" : "red");
      }

      const gp = a.gripper_pct || 0;
      document.getElementById("grip-pct").textContent = `${gp}%`;
      document.getElementById("grip-fill").style.width = `${gp}%`;

      const w = a.joint_limit_warn || "";
      document.getElementById("warn-jl").textContent = w;

      // sync arm-speed sliders only on first connect
      if (a.settings && !slArmLin._touched) {
        if (Math.abs(armLin - a.settings.lin_speed) > 0.001) {
          armLin = a.settings.lin_speed;
          slArmLin.value = armLin;
          document.getElementById("sl_arm_lin_v").textContent = armLin.toFixed(2);
        }
      }
      if (a.settings && !slArmAng._touched) {
        if (Math.abs(armAng - a.settings.ang_speed) > 0.01) {
          armAng = a.settings.ang_speed;
          slArmAng.value = armAng;
          document.getElementById("sl_arm_ang_v").textContent = armAng.toFixed(2);
        }
      }
    }

    // ----- base / power -----
    if (d.base && !d.base.error) {
      setPill("pill-base", "base: ok", "ok");
      const b = d.base;
      const bp = b.batt_pct;
      const bpEl = document.getElementById("batt-pct");
      bpEl.textContent = `${bp}%`;
      bpEl.className = "value " + (bp < 20 ? "red" : bp < 50 ? "orange" : "green");
      document.getElementById("vbus").textContent = fmt(b.vbus, 2, "V");
      document.getElementById("ibus").textContent = fmt(b.ibus, 3, "A");
      document.getElementById("ibus-peak").textContent = `(peak ${fmt(b.peak_ibus, 3, "A")})`;
      // flow badge
      let flow = '<span class="badge muted">idle</span>';
      if (b.ibus != null) {
        if (b.ibus > 0.05)       flow = '<span class="badge drive">DRIVE</span>';
        else if (b.ibus < -0.05) flow = '<span class="badge regen">REGEN</span>';
      }
      document.getElementById("ibus-badge").innerHTML = flow;
      document.getElementById("dc-power").textContent = fmt(b.dc_power, 2, "W");

      let brake;
      if (b.brake_armed === null || b.brake_armed === undefined) brake = '<span class="badge muted">n/a</span>';
      else if (b.brake_sat) brake = '<span class="badge warn">SATURATED</span>';
      else if (b.brake_armed) brake = '<span class="badge ok">armed</span>';
      else brake = '<span class="badge muted">disarmed</span>';
      document.getElementById("brake-state").innerHTML = brake;

      if (b.left)
        document.getElementById("motor-l").textContent =
          `${fmt(b.left.current_A, 3, "A")} / ${fmt(b.left.vel_mps, 3, "m/s")}`;
      if (b.left)
        document.getElementById("motor-l-tq").textContent = fmt(b.left.torque_Nm, 4, "Nm");
      if (b.right)
        document.getElementById("motor-r").textContent =
          `${fmt(b.right.current_A, 3, "A")} / ${fmt(b.right.vel_mps, 3, "m/s")}`;
      if (b.right)
        document.getElementById("motor-r-tq").textContent = fmt(b.right.torque_Nm, 4, "Nm");

      // sync sliders only on first connect
      if (b.settings && !slLin._touched) {
        if (Math.abs(linSpeed - b.settings.linear_speed) > 0.001) {
          linSpeed = b.settings.linear_speed;
          slLin.value = linSpeed;
          document.getElementById("sl_lin_v").textContent = linSpeed.toFixed(2);
        }
      }
      if (b.settings && !slAng._touched) {
        if (Math.abs(angSpeed - b.settings.angular_speed) > 0.1) {
          angSpeed = b.settings.angular_speed;
          slAng.value = angSpeed;
          document.getElementById("sl_ang_v").textContent = angSpeed.toFixed(0);
        }
      }
    } else if (d.base && d.base.error) {
      setPill("pill-base", "base: " + d.base.error, "warn");
    } else {
      setPill("pill-base", "base: not connected", "muted");
    }

    // ----- vio -----
    if (d.vio) {
      const v = d.vio;
      document.getElementById("vio-line").textContent = v.status || "—";
      document.getElementById("vio-age").textContent =
        v.age_s == null ? "—" : `${v.age_s}s`;
      document.getElementById("vio-pos").textContent =
        v.pos == null ? "—" :
        `(${v.pos[0].toFixed(2)}, ${v.pos[1].toFixed(2)}, ${v.pos[2].toFixed(2)})`;
      if (v.has_recent_pose) setPill("pill-vio", "vio: live",  "ok");
      else if (v.age_s != null) setPill("pill-vio", `vio: ${v.age_s.toFixed(1)}s old`, "warn");
      else setPill("pill-vio", "vio: no pose", "muted");
    }
  }
  slLin.addEventListener("change", () => slLin._touched = true);
  slAng.addEventListener("change", () => slAng._touched = true);
</script>
</body>
</html>"""


# ===================== ROUTES =====================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE.replace("__VISER_PORT__", str(VISER_PORT))


@app.get("/rgb_stream")
async def rgb_stream():
    """Insight 9 RGB MJPEG (direct passthrough of CompressedImage)."""
    return StreamingResponse(
        insight_rgb_mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws_control(ws: WebSocket):
    global last_arm_key_time, last_base_key_time, wrench_bias
    global linear_speed, angular_speed, arm_lin_speed, arm_ang_speed
    await ws.accept()
    log.info("Control WS connected")
    try:
        while True:
            data = json.loads(await ws.receive_text())
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong", "t": data.get("t", 0)}))

            elif msg_type == "arm_keys":
                if not arm_available:
                    continue
                keys = set(data.get("keys", []))
                arm_keys.clear()
                arm_keys.update(keys)
                last_arm_key_time = time.monotonic()

            elif msg_type == "base_keys":
                keys = set(data.get("keys", []))
                base_keys.clear()
                base_keys.update(keys)
                last_base_key_time = time.monotonic()

            elif msg_type == "speed":
                linear_speed = max(0.01, min(2.0, float(data.get("linear", linear_speed))))
                angular_speed = max(1.0, min(360.0, float(data.get("angular", angular_speed))))
                log.info("Base speed: linear=%.2f m/s, angular=%.1f deg/s",
                         linear_speed, angular_speed)

            elif msg_type == "arm_speed":
                arm_lin_speed = max(ARM_LIN_MIN, min(ARM_LIN_MAX,
                                                     float(data.get("linear",  arm_lin_speed))))
                arm_ang_speed = max(ARM_ANG_MIN, min(ARM_ANG_MAX,
                                                     float(data.get("angular", arm_ang_speed))))
                log.info("Arm EE speed: lin=%.2f m/s, ang=%.2f rad/s",
                         arm_lin_speed, arm_ang_speed)

            elif msg_type == "action":
                action = data.get("action", "")
                if action == "reset" and arm_available:
                    threading.Thread(target=do_reset, daemon=True).start()
                elif action == "zero_ft" and arm_available:
                    current_wrench = np.array(tele_data.get("f_ext", [0,0,0]) +
                                               tele_data.get("m_ext", [0,0,0]))
                    wrench_bias += current_wrench
                    log.info("[M] F/T sensor zeroed")
                elif action == "print" and arm_available:
                    with lock:
                        p_snap = x_des_pos.copy()
                        R_snap = x_des_rot.copy()
                    print_pose(p_snap, R_snap, "Current target pose")

    except WebSocketDisconnect:
        arm_keys.clear()
        base_keys.clear()
        log.info("Control WS disconnected")


@app.websocket("/ws/tele")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    telemetry_clients.append(ws)
    log.info("Telemetry WS connected (total: %d)", len(telemetry_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in telemetry_clients:
            telemetry_clients.remove(ws)
        log.info("Telemetry WS disconnected (total: %d)", len(telemetry_clients))


if __name__ == "__main__":
    try:
        ip = subprocess.check_output(
            "hostname -I", shell=True, text=True
        ).strip().split()[0]
        log.info("Open in browser: http://%s:%d", ip, PORT)
    except Exception:
        pass

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
