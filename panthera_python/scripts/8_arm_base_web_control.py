#!/usr/bin/env python3
"""
Arm + base combined web control
Panthera Cartesian impedance (polar) + ODrive biwheel — single web page.

Arm keys (same as 7_keyboard_ee_control_cart_impedance_polar_web.py):
  W/S: radial move    A/D: orbit around Z    Q/E: vertical
  I/K: Pitch       J/L: Yaw             U/O: Roll
  Z: close gripper     X: open gripper
  Space=print pose    R=return home    M=zero wrench

Base keys (arrow keys):
  ↑/↓: forward/back    ←/→: turn left/right

Usage:
    source <repo>/.venv/bin/activate
    cd <repo>/panthera_python/scripts
    python 8_arm_base_web_control.py

Then open http://<jetson-ip>:8083 in your PC browser.
"""

import asyncio
import json
import logging
import math
import subprocess
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from scipy.spatial.transform import Rotation as Rot
import pinocchio as pin

from Panthera_lib import Panthera

# ===================== CONFIG =====================
HOST = "0.0.0.0"
PORT = 8083
TELEMETRY_HZ = 20
IDLE_TIMEOUT = 0.3  # seconds with no key -> stop

# ── Arm ──
POS_STEP        = 0.002
ROT_STEP        = 0.2
POLAR_ANGLE_MAX = np.radians(1.5)
CTRL_FREQ_IMP   = 2000
KEY_CMD_HZ      = 200

MAX_TORQUE = [21.0, 36.0, 36.0, 30.0, 10.0, 10.0]
JOINT_VEL  = [0.5] * 6

K_POS = np.array([30.0, 30.0, 50.0])
K_ROT = np.array([30.0, 30.0, 40.0])
K_CART = np.concatenate([K_POS, K_ROT])

B_POS = np.array([0.0, 0.0, 0.0])
B_ROT = np.array([0.0, 0.0, 0.0])
B_CART = np.concatenate([B_POS, B_ROT])

LAMBDA_DAMP = 0.05
TAU_LIMIT = np.array([20.0, 30.0, 30.0, 20.0, 10.0, 10.0])
JOINT_DAMPING = np.array([1.5, 2, 2, 2, 1, 0.8])

IMP_Fc         = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01])
IMP_Fv         = np.array([0.03, 0.03, 0.03, 0.03, 0.01, 0.01])
IMP_VEL_THRESH = 0.03

ENABLE_CORIOLIS = False
TOOL_OFFSET = np.array([0.165, 0.0, 0.0])

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

# ── Camera: D405 (arm) ──
CAM1_DEV     = "/dev/video4"
CAM1_WIDTH   = 424
CAM1_HEIGHT  = 240
CAM1_FPS     = 15

# ── Camera: Stereo USB (base) — MJPG hw-encode, left eye only, rotated 180 ──
CAM2_DEV     = "/dev/video6"
CAM2_WIDTH   = 1280         # side-by-side → left eye = 640x480
CAM2_HEIGHT  = 480
CAM2_FPS     = 30

JPEG_QUALITY = 60
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

    zero_pos = [0.0] * robot.motor_count

    log.info("Arm: moving to zero position...")
    robot.Joint_Pos_Vel(zero_pos, JOINT_VEL, MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    home_rot = robot.rotation_matrix_from_euler(*HOME_ROT_EULER)
    init_joints = robot.inverse_kinematics(HOME_POS, home_rot, robot.get_current_pos())
    if init_joints is None:
        log.warning("Home IK failed, staying at zero")
    else:
        log.info("Arm: moving to home position: %s", HOME_POS)
        robot.moveJ(init_joints, duration=3.0, max_tqu=MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    fk_after_home = robot.forward_kinematics()
    p_lift = np.array(fk_after_home['position'])
    R_lift = np.array(fk_after_home['rotation'])
    p_lift[2] += 0.1

    log.info("Arm: lifting EE +0.1m along Z...")
    success = robot.moveL(
        target_position=p_lift,
        target_rotation=R_lift,
        duration=2.0,
        use_spline=True
    )
    if not success:
        log.warning("moveL lift failed, using current position as start")
    time.sleep(0.5)

    q0 = np.array(robot.get_current_pos())
    ctrl_data = robot.model.createData()
    p0, R0, _ = compute_fk_and_jacobian(robot, ctrl_data, q0)
    print_pose(p0, R0, "Cartesian impedance start pose")

    gripper_min = robot.gripper_limits['lower'] if robot.gripper_limits else GRIPPER_MIN_DEFAULT
    gripper_max = robot.gripper_limits['upper'] if robot.gripper_limits else GRIPPER_MAX_DEFAULT
    log.info("Gripper limits: [%.2f, %.2f] rad", gripper_min, gripper_max)

    jl_lower = np.array(robot.joint_limits['lower']) + JOINT_LIMIT_MARGIN
    jl_upper = np.array(robot.joint_limits['upper']) - JOINT_LIMIT_MARGIN

    x_des_pos = p0.copy()
    x_des_rot = R0.copy()
    last_feasible_pos = p0.copy()
    last_feasible_rot = R0.copy()
    gripper_des = float(np.clip(robot.get_current_pos_gripper(), gripper_min, gripper_max))

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


# ===================== CAMERA =====================

_cam_frames = {"cam1": b"", "cam2": b""}
_cam_locks = {"cam1": threading.Lock(), "cam2": threading.Lock()}
_cam_events = {"cam1": threading.Event(), "cam2": threading.Event()}
_cam_running = False


def _open_cam(dev, width, height, fps, mjpg=False):
    """Open V4L2 camera with minimal buffer for low latency."""
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize V4L2 buffer latency
    return cap


def _run_cam1():
    """D405 capture — direct encode."""
    cap = _open_cam(CAM1_DEV, CAM1_WIDTH, CAM1_HEIGHT, CAM1_FPS)
    if not cap.isOpened():
        log.warning("cam1 (%s) failed to open", CAM1_DEV)
        return
    log.info("cam1 (D405) opened: %dx%d @ %dfps", CAM1_WIDTH, CAM1_HEIGHT, CAM1_FPS)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while _cam_running:
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode(".jpg", frame, enc)
        if ok:
            with _cam_locks["cam1"]:
                _cam_frames["cam1"] = buf.tobytes()
            _cam_events["cam1"].set()
    cap.release()


def _run_cam2():
    """Stereo MJPG capture — V4L2 raw JPEG passthrough, zero CPU.
    Crop + rotate done in browser via CSS."""
    import fcntl
    import struct as _struct
    import mmap as _mmap

    # ARM64 ioctl numbers (from kernel headers, sizeof(v4l2_buffer)=88)
    VIDIOC_S_FMT     = 0xC0D05605
    VIDIOC_REQBUFS   = 0xC0145608
    VIDIOC_QUERYBUF  = 0xC0585609
    VIDIOC_QBUF      = 0xC058560F
    VIDIOC_DQBUF     = 0xC0585611
    VIDIOC_STREAMON  = 0x40045612
    VIDIOC_STREAMOFF = 0x40045613

    V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
    V4L2_MEMORY_MMAP = 1
    V4L2_PIX_FMT_MJPEG = 0x47504A4D

    # v4l2_format: type(4) + pix_format(32) + padding = 208 bytes
    FMT_SIZE = 208
    # v4l2_requestbuffers: count(4) + type(4) + memory(4) + reserved(8) = 20 bytes
    REQBUFS_SIZE = 20
    # v4l2_buffer: 88 bytes on ARM64
    BUF_SIZE = 88
    # Field offsets in v4l2_buffer
    OFF_INDEX     = 0
    OFF_TYPE      = 4
    OFF_BYTESUSED = 8
    OFF_MEMORY    = 60
    OFF_M_OFFSET  = 64   # m.offset (u32 at start of 8-byte union)
    OFF_LENGTH    = 72

    NUM_BUFS = 2

    try:
        fd = open(CAM2_DEV, "rb+", buffering=0)
    except OSError as e:
        log.warning("cam2 (%s) failed to open: %s", CAM2_DEV, e)
        return

    fileno = fd.fileno()

    # Set format: MJPG
    fmt_buf = bytearray(FMT_SIZE)
    _struct.pack_into("I", fmt_buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)  # type
    _struct.pack_into("I", fmt_buf, 4, CAM2_WIDTH)    # width
    _struct.pack_into("I", fmt_buf, 8, CAM2_HEIGHT)   # height
    _struct.pack_into("I", fmt_buf, 12, V4L2_PIX_FMT_MJPEG)  # pixelformat
    fcntl.ioctl(fileno, VIDIOC_S_FMT, fmt_buf)
    actual_w = _struct.unpack_from("I", fmt_buf, 4)[0]
    actual_h = _struct.unpack_from("I", fmt_buf, 8)[0]
    log.info("cam2 (V4L2 raw MJPG) opened: %dx%d @ %dfps — zero CPU",
             actual_w, actual_h, CAM2_FPS)

    # Request buffers
    req_buf = bytearray(REQBUFS_SIZE)
    _struct.pack_into("III", req_buf, 0, NUM_BUFS, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP)
    fcntl.ioctl(fileno, VIDIOC_REQBUFS, req_buf)
    actual_count = _struct.unpack_from("I", req_buf, 0)[0]

    # Query & mmap buffers
    buffers = []
    for i in range(actual_count):
        qbuf = bytearray(BUF_SIZE)
        _struct.pack_into("I", qbuf, OFF_INDEX, i)
        _struct.pack_into("I", qbuf, OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        _struct.pack_into("I", qbuf, OFF_MEMORY, V4L2_MEMORY_MMAP)
        fcntl.ioctl(fileno, VIDIOC_QUERYBUF, qbuf)
        length = _struct.unpack_from("I", qbuf, OFF_LENGTH)[0]
        offset = _struct.unpack_from("I", qbuf, OFF_M_OFFSET)[0]
        mm = _mmap.mmap(fileno, length, offset=offset)
        buffers.append((length, mm))

    # Queue all buffers
    for i in range(actual_count):
        qbuf = bytearray(BUF_SIZE)
        _struct.pack_into("I", qbuf, OFF_INDEX, i)
        _struct.pack_into("I", qbuf, OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        _struct.pack_into("I", qbuf, OFF_MEMORY, V4L2_MEMORY_MMAP)
        fcntl.ioctl(fileno, VIDIOC_QBUF, qbuf)

    # Start streaming
    stream_arg = _struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
    fcntl.ioctl(fileno, VIDIOC_STREAMON, stream_arg)

    try:
        while _cam_running:
            # Dequeue
            dqbuf = bytearray(BUF_SIZE)
            _struct.pack_into("I", dqbuf, OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            _struct.pack_into("I", dqbuf, OFF_MEMORY, V4L2_MEMORY_MMAP)
            fcntl.ioctl(fileno, VIDIOC_DQBUF, dqbuf)

            idx = _struct.unpack_from("I", dqbuf, OFF_INDEX)[0]
            bytesused = _struct.unpack_from("I", dqbuf, OFF_BYTESUSED)[0]

            # Read raw JPEG bytes — zero decode!
            mm = buffers[idx][1]
            mm.seek(0)
            jpeg_bytes = mm.read(bytesused)

            with _cam_locks["cam2"]:
                _cam_frames["cam2"] = jpeg_bytes
            _cam_events["cam2"].set()

            # Re-queue
            fcntl.ioctl(fileno, VIDIOC_QBUF, dqbuf)
    except Exception as e:
        log.warning("cam2 V4L2 error: %s", e)
    finally:
        fcntl.ioctl(fileno, VIDIOC_STREAMOFF, stream_arg)
        for _, mm in buffers:
            mm.close()
        fd.close()


def start_cameras():
    global _cam_running
    _cam_running = True
    threading.Thread(target=_run_cam1, daemon=True).start()
    threading.Thread(target=_run_cam2, daemon=True).start()


def stop_cameras():
    global _cam_running
    _cam_running = False
    # Unblock any waiting generators
    _cam_events["cam1"].set()
    _cam_events["cam2"].set()


def mjpeg_generator(cam_id: str):
    """Event-driven MJPEG: yields as soon as a new frame arrives, no polling."""
    evt = _cam_events[cam_id]
    lck = _cam_locks[cam_id]
    while _cam_running:
        evt.wait()           # block until new frame (no CPU spin / sleep)
        evt.clear()
        with lck:
            frame = _cam_frames[cam_id]
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )


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


def read_odrive_telemetry() -> dict:
    try:
        l_cur = axis_l.motor.current_control.Iq_measured
        r_cur = axis_r.motor.current_control.Iq_measured
        l_vel = axis_l.encoder.vel_estimate
        r_vel = axis_r.encoder.vel_estimate
        l_torque = l_cur * TORQUE_CONSTANT
        r_torque = r_cur * TORQUE_CONSTANT
        vbus = odrv.vbus_voltage
        batt_raw = max(0.0, min(100.0, (vbus - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100.0))
        batt_pct = round(batt_raw / 5.0) * 5
        return {
            "vbus": round(vbus, 2),
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
            if "up" in base_keys:
                x += linear_speed
            if "down" in base_keys:
                x -= linear_speed
            if "left" in base_keys:
                theta += angular_speed
            if "right" in base_keys:
                theta -= angular_speed
            send_velocity(x, theta)
        elif axis_l is not None:
            send_velocity(0.0, 0.0)
        await asyncio.sleep(interval)


# ===================== ARM IMPEDANCE LOOP (thread) =====================

def impedance_loop():
    global joint_limit_warn, joint_limit_warn_time
    global last_feasible_pos, last_feasible_rot
    global x_des_pos, x_des_rot

    _z6 = [0.0] * 6
    dt = 1.0 / CTRL_FREQ_IMP
    ctrl_data = robot.model.createData()

    F_ext_filtered = np.zeros(6)
    alpha_ft = (2 * np.pi * FT_CUTOFF_FREQ * dt) / (1 + 2 * np.pi * FT_CUTOFF_FREQ * dt)

    tele_counter = 0

    while not stop_event.is_set():
        t0 = time.time()

        if ctrl_enabled.is_set():
            q = np.array(robot.get_current_pos())
            dq = np.array(robot.get_current_vel())

            # Joint limit check
            violated = (q < jl_lower) | (q > jl_upper)
            if np.any(violated):
                parts = []
                for i in range(len(q)):
                    if violated[i]:
                        side = "lower" if q[i] < jl_lower[i] else "upper"
                        parts.append(f"J{i+1}{side}({q[i]:+.2f})")
                with lock:
                    joint_limit_warn = " ".join(parts)
                    joint_limit_warn_time = time.time()
                    x_des_pos[:] = last_feasible_pos
                    x_des_rot[:] = last_feasible_rot
            else:
                with lock:
                    last_feasible_pos[:] = x_des_pos
                    last_feasible_rot[:] = x_des_rot

            # FK + Jacobian
            p_cur, R_cur, J = compute_fk_and_jacobian(robot, ctrl_data, q)

            with lock:
                p_des = x_des_pos.copy()
                R_des = x_des_rot.copy()
                g_des = gripper_des

            # Cartesian error
            e_pos = p_des - p_cur
            e_rot = orientation_error_axis_angle(R_des, R_cur)
            e_x = np.concatenate([e_pos, e_rot])

            dx = J @ dq
            F = K_CART * e_x - B_CART * dx

            # DLS
            JJT = J @ J.T
            alpha_dls = np.linalg.solve(JJT + LAMBDA_DAMP**2 * np.eye(6), F)
            tor_cart = J.T @ alpha_dls

            tor_joint_damp = -JOINT_DAMPING * dq

            tor_gra = np.array(robot.get_Gravity())
            tor_cor = np.array(robot.get_Coriolis_vector(q, dq)) if ENABLE_CORIOLIS else np.zeros(6)
            tor_fri = np.array(robot.get_friction_compensation(
                dq, IMP_Fc, IMP_Fv, IMP_VEL_THRESH))

            tor = np.clip(tor_cart + tor_joint_damp + tor_gra + tor_cor + tor_fri,
                          -TAU_LIMIT, TAU_LIMIT)

            robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                g_des, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)
            robot.pos_vel_tqe_kp_kd(_z6, _z6, tor.tolist(), _z6, _z6)

            # External force estimation
            tau_measured = np.array(robot.get_current_torque())
            tau_model = tor_gra + tor_cor + tor_fri
            tau_ext = tau_measured - tau_model

            JJT_ft = J @ J.T
            F_ext_raw = J @ np.linalg.solve(
                JJT_ft + LAMBDA_DAMP**2 * np.eye(6), tau_ext)

            F_ext_filtered[:] = alpha_ft * F_ext_raw + (1 - alpha_ft) * F_ext_filtered

            # Raw telemetry every 10th iteration (~200Hz)
            tele_counter += 1
            if tele_counter >= 10:
                tele_counter = 0
                g_actual = robot.get_current_pos_gripper()
                wrench_disp = F_ext_filtered - wrench_bias
                tele_data_raw.update({
                    "e_pos": e_pos.copy(),
                    "e_rot": e_rot.copy(),
                    "p_cur": p_cur.copy(),
                    "p_des": p_des.copy(),
                    "wrench": wrench_disp.copy(),
                    "g_des": g_des,
                    "g_actual": g_actual,
                    "q": q.copy(),
                    "dq": dq.copy(),
                    "tor": tor.copy(),
                })

        elapsed = time.time() - t0
        remaining = dt - elapsed
        time.sleep(max(0.00005, remaining))

    # Release motors on exit
    try:
        robot.set_stop()
        log.info("Impedance loop stopped — motors released")
    except Exception:
        pass


# ===================== ARM KEY COMMAND (async) =====================

async def arm_key_command_loop():
    global gripper_des, x_des_pos, x_des_rot
    dt = 1.0 / KEY_CMD_HZ

    while not stop_event.is_set():
        now = time.monotonic()

        if ctrl_enabled.is_set() and arm_keys and (now - last_arm_key_time < IDLE_TIMEOUT):
            combined_pos = np.zeros(3)
            first_rot_euler = None
            any_pressed = False

            for k in list(arm_keys):
                if k in KEY_MAP_ARM:
                    dp, rot_euler = KEY_MAP_ARM[k]
                    combined_pos += dp
                    if rot_euler is not None and first_rot_euler is None:
                        first_rot_euler = rot_euler
                    any_pressed = True

            # Polar coordinates (W/S=radial, A/D=rotation)
            with lock:
                px, py = x_des_pos[0], x_des_pos[1]
            r_horiz = np.hypot(px, py)

            if "a" in arm_keys or "d" in arm_keys:
                dtheta = POS_STEP / max(r_horiz, 0.01)
                dtheta = min(dtheta, POLAR_ANGLE_MAX)
                if "d" in arm_keys:
                    dtheta = -dtheta
                c, s = np.cos(dtheta), np.sin(dtheta)
                Rz = np.array([[c, -s, 0.0],
                               [s,  c, 0.0],
                               [0.0, 0.0, 1.0]])
                with lock:
                    x_des_pos[:] = Rz @ x_des_pos
                    x_des_rot[:] = Rz @ x_des_rot
                any_pressed = True

            if "w" in arm_keys or "s" in arm_keys:
                if r_horiz > 0.001:
                    radial_dir = np.array([px, py, 0.0]) / r_horiz
                else:
                    radial_dir = np.array([1.0, 0.0, 0.0])
                dr = POS_STEP if "w" in arm_keys else -POS_STEP
                with lock:
                    x_des_pos += dr * radial_dir
                any_pressed = True

            if any_pressed:
                with lock:
                    x_des_pos += combined_pos
                    if first_rot_euler is not None:
                        dR = robot.rotation_matrix_from_euler(*first_rot_euler)
                        new_rot = x_des_rot @ dR
                        U, _, Vt = np.linalg.svd(new_rot)
                        x_des_rot[:] = U @ Vt

            # Gripper
            if "z" in arm_keys or "x" in arm_keys:
                with lock:
                    if "z" in arm_keys:
                        gripper_des = max(gripper_min, gripper_des - GRIPPER_STEP)
                    if "x" in arm_keys:
                        gripper_des = min(gripper_max, gripper_des + GRIPPER_STEP)

        await asyncio.sleep(dt)


# ===================== RESET =====================

def do_reset():
    global gripper_des
    if not arm_available:
        log.warning("Reset ignored — arm not available")
        return
    ctrl_enabled.clear()
    log.info("Resetting arm to home position...")
    home_q = init_joints if init_joints is not None else [0.0] * robot.motor_count
    robot.moveJ(home_q, duration=2.0, max_tqu=MAX_TORQUE, iswait=True)

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

    # Init robot (may fail) and ODrive — before cameras to avoid fd inheritance in probe subprocess
    await loop.run_in_executor(None, init_robot)
    await loop.run_in_executor(None, init_odrive)

    # Camera threads (after init to avoid probe subprocess inheriting V4L2 fds)
    start_cameras()

    t_ctrl = None
    t_arm_keys = None
    if arm_available:
        t_ctrl = threading.Thread(target=impedance_loop, daemon=True)
        t_ctrl.start()
        t_arm_keys = asyncio.create_task(arm_key_command_loop())
    else:
        log.info("Arm unavailable — starting base-only mode")

    t_base = asyncio.create_task(odrive_control_loop())
    t_tele = asyncio.create_task(telemetry_loop())

    yield

    stop_event.set()
    if t_arm_keys:
        t_arm_keys.cancel()
    t_base.cancel()
    t_tele.cancel()
    if t_ctrl:
        t_ctrl.join(timeout=2)
    stop_cameras()
    await loop.run_in_executor(None, disconnect_odrive)
    log.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)


# ===================== HTML =====================

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arm + Base Control</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee;
    display: flex; flex-direction: column;
    align-items: center; padding: 1rem;
    min-height: 100vh; user-select: none;
  }
  h1 { font-size: 1.2rem; margin-bottom: 0.3rem; color: #e94560; }
  .status-bar { display: flex; gap: 1.5rem; font-size: 0.82rem; color: #888; margin-bottom: 0.6rem; }
  .status-bar .connected { color: #0f0; }

  .top-row { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; justify-content: center; margin-bottom: 0.8rem; }

  .camera { text-align: center; }
  .camera h3 { font-size: 0.8rem; color: #e94560; margin-bottom: 0.3rem; }
  .camera img { border-radius: 8px; border: 2px solid #333; background: #000; max-width: 640px; height: auto; display: block; }
  .cam-wrap { display: inline-block; border-radius: 8px; overflow: hidden; }
  .cam-wrap.stereo { width: 640px; height: 480px; }
  .cam-wrap.stereo img { max-width: none; width: 1280px; height: 480px; transform: rotate(180deg) translateX(50%); }
  .cam-select { display: flex; gap: 6px; justify-content: center; margin-bottom: 0.3rem; }
  .cam-btn {
    padding: 3px 12px; border: 2px solid #444; border-radius: 7px;
    background: #16213e; color: #aaa; font-size: 0.72rem; cursor: pointer;
  }
  .cam-btn.active { border-color: #e94560; color: #eee; background: #e94560; }

  .main { display: flex; gap: 1.2rem; align-items: flex-start; flex-wrap: wrap; justify-content: center; }

  .panel { display: flex; flex-direction: column; align-items: center; }
  .panel h3 { font-size: 0.8rem; color: #e94560; margin-bottom: 0.4rem; }

  .key-section { margin-bottom: 0.5rem; text-align: center; }
  .key-section h4 { font-size: 0.7rem; color: #888; margin-bottom: 0.2rem; }
  .key-row { display: flex; gap: 4px; justify-content: center; margin-bottom: 3px; }
  .key {
    width: 44px; height: 38px;
    border: 2px solid #444; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: bold; background: #16213e;
    transition: all 0.08s; cursor: pointer;
  }
  .key.active { background: #e94560; border-color: #e94560; transform: scale(1.05); }
  .key.wide { width: 58px; font-size: 0.68rem; }

  /* Arrow keys for base */
  .arrow-keys {
    display: grid;
    grid-template-areas: ". up ." "left down right";
    gap: 5px;
  }
  .arrow-key {
    width: 52px; height: 48px;
    border: 2px solid #444; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; background: #16213e;
    transition: all 0.08s; cursor: pointer;
  }
  .arrow-key.active { background: #0a84ff; border-color: #0a84ff; transform: scale(1.05); }
  .arrow-key[data-dir="up"]    { grid-area: up; }
  .arrow-key[data-dir="left"]  { grid-area: left; }
  .arrow-key[data-dir="down"]  { grid-area: down; }
  .arrow-key[data-dir="right"] { grid-area: right; }

  .vel { margin-top: 0.4rem; font-family: monospace; font-size: 0.75rem; color: #aaa; text-align: center; }

  .sliders { margin-top: 0.6rem; width: 100%; }
  .slider-row {
    display: flex; align-items: center; gap: 0.4rem;
    margin-bottom: 0.4rem; font-size: 0.78rem;
  }
  .slider-row label { width: 60px; text-align: right; color: #aaa; }
  .slider-row input[type=range] { flex: 1; accent-color: #0a84ff; }
  .slider-row .val { width: 60px; font-family: monospace; color: #0a84ff; font-size: 0.72rem; }

  .actions { display: flex; gap: 5px; margin-top: 0.4rem; }
  .btn {
    padding: 4px 12px; border: 2px solid #444; border-radius: 7px;
    background: #16213e; color: #eee; font-size: 0.72rem; cursor: pointer;
  }
  .btn:hover { border-color: #e94560; }
  .btn:active { background: #e94560; }

  .telemetry {
    background: #16213e; border-radius: 10px; padding: 0.6rem 0.8rem;
    min-width: 360px; font-family: monospace; font-size: 0.72rem; line-height: 1.5;
  }
  .telemetry h3 { font-size: 0.8rem; color: #e94560; margin-bottom: 0.3rem; font-family: sans-serif; }
  .trow { display: flex; justify-content: space-between; gap: 0.6rem; }
  .trow .label { color: #888; white-space: nowrap; }
  .trow .value { color: #eee; text-align: right; }
  .tsep { border-top: 1px solid #333; margin: 0.2rem 0; }

  .warn { color: #ff4444; font-weight: bold; animation: blink 0.5s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  .ok { color: #4caf50; }
  .caution { color: #ddc640; }
  .danger { color: #e94560; }

  .joint-table { width: 100%; border-collapse: collapse; font-size: 0.68rem; }
  .joint-table th { color: #888; font-weight: normal; text-align: right; padding: 1px 3px; }
  .joint-table td { color: #eee; text-align: right; padding: 1px 3px; }
  .joint-table .jlabel { color: #e94560; text-align: left; }

  .base-tele { margin-top: 0.3rem; }
  .base-tele h4 { font-size: 0.75rem; color: #0a84ff; margin-bottom: 0.2rem; font-family: sans-serif; }
</style>
</head>
<body>
<h1>Arm + Base Control</h1>
<div class="status-bar">
  <span id="status">Connecting...</span>
  <span id="latency">Latency: -- ms</span>
  <span id="ctrl_status">CTRL: --</span>
</div>

<div class="top-row">
  <div class="camera">
    <div class="cam-select">
      <button class="cam-btn active" data-cam="cam1">D405 (Arm)</button>
      <button class="cam-btn" data-cam="cam2">Stereo (Base)</button>
      <button class="cam-btn" data-cam="off">Off</button>
    </div>
    <div class="cam-wrap" id="cam_wrap">
      <img id="cam" src="/video?cam=cam1" alt="camera stream">
    </div>
  </div>
</div>

<div class="main">
  <!-- Arm controls -->
  <div class="panel" id="arm_panel">
    <h3>Arm (WASD + UIOJKL)</h3>
    <div id="arm_unavailable" style="display:none; color:#f55; font-size:0.8rem; margin-bottom:0.5rem; text-align:center; padding:0.5rem; border:1px solid #f55; border-radius:7px;">Arm not detected</div>
    <div class="key-section">
      <h4>Position (Polar)</h4>
      <div class="key-row">
        <div class="key" data-key="q">Q<br><span style="font-size:0.5rem;color:#888">Z+</span></div>
        <div class="key" data-key="w">W<br><span style="font-size:0.5rem;color:#888">Rad+</span></div>
        <div class="key" data-key="e">E<br><span style="font-size:0.5rem;color:#888">Z-</span></div>
      </div>
      <div class="key-row">
        <div class="key" data-key="a">A<br><span style="font-size:0.5rem;color:#888">CCW</span></div>
        <div class="key" data-key="s">S<br><span style="font-size:0.5rem;color:#888">Rad-</span></div>
        <div class="key" data-key="d">D<br><span style="font-size:0.5rem;color:#888">CW</span></div>
      </div>
    </div>
    <div class="key-section">
      <h4>Orientation (EE)</h4>
      <div class="key-row">
        <div class="key" data-key="u">U<br><span style="font-size:0.5rem;color:#888">Ro+</span></div>
        <div class="key" data-key="i">I<br><span style="font-size:0.5rem;color:#888">Pi+</span></div>
        <div class="key" data-key="o">O<br><span style="font-size:0.5rem;color:#888">Ro-</span></div>
      </div>
      <div class="key-row">
        <div class="key" data-key="j">J<br><span style="font-size:0.5rem;color:#888">Ya+</span></div>
        <div class="key" data-key="k">K<br><span style="font-size:0.5rem;color:#888">Pi-</span></div>
        <div class="key" data-key="l">L<br><span style="font-size:0.5rem;color:#888">Ya-</span></div>
      </div>
    </div>
    <div class="key-section">
      <h4>Gripper</h4>
      <div class="key-row">
        <div class="key wide" data-key="z">Z Close</div>
        <div class="key wide" data-key="x">X Open</div>
      </div>
    </div>
    <div class="actions">
      <button class="btn" id="btn_reset">R Reset</button>
      <button class="btn" id="btn_zero">M Zero F/T</button>
      <button class="btn" id="btn_print">Print Pose</button>
    </div>
  </div>

  <!-- Base controls -->
  <div class="panel">
    <h3>Base (Arrow Keys)</h3>
    <div class="arrow-keys">
      <div class="arrow-key" data-dir="up">&uarr;</div>
      <div class="arrow-key" data-dir="left">&larr;</div>
      <div class="arrow-key" data-dir="down">&darr;</div>
      <div class="arrow-key" data-dir="right">&rarr;</div>
    </div>
    <div class="vel" id="vel">x: 0.00 m/s &nbsp; &theta;: 0.00 &deg;/s</div>
    <div class="sliders">
      <div class="slider-row">
        <label>Linear</label>
        <input type="range" id="sl_lin" min="0.02" max="1.0" step="0.01" value="0.15">
        <span class="val" id="sl_lin_v">0.15 m/s</span>
      </div>
      <div class="slider-row">
        <label>Angular</label>
        <input type="range" id="sl_ang" min="5" max="180" step="1" value="30">
        <span class="val" id="sl_ang_v">30 &deg;/s</span>
      </div>
    </div>
  </div>

  <!-- Telemetry -->
  <div class="telemetry">
    <h3>Telemetry</h3>
    <div id="tele_body">Waiting for data...</div>
  </div>
</div>

<script>
const ARM_KEYS = new Set(["w","a","s","d","q","e","i","k","j","l","u","o","z","x"]);
const BASE_MAP = {
  ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right"
};
const armPressed = new Set();
const basePressed = new Set();

// ── Control WebSocket ──
let wsCtrl = null;
let pingId = null;
function connectCtrl() {
  if (pingId) { clearInterval(pingId); pingId = null; }
  wsCtrl = new WebSocket(`ws://${location.host}/ws`);
  wsCtrl.onopen = () => {
    document.getElementById("status").textContent = "Connected";
    document.getElementById("status").className = "connected";
    pingId = setInterval(() => {
      if (wsCtrl.readyState === 1)
        wsCtrl.send(JSON.stringify({type: "ping", t: performance.now()}));
    }, 1000);
  };
  wsCtrl.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "pong") {
      const rtt = performance.now() - msg.t;
      document.getElementById("latency").textContent = `Latency: ${rtt.toFixed(1)} ms`;
    }
  };
  wsCtrl.onclose = () => {
    document.getElementById("status").textContent = "Disconnected";
    document.getElementById("status").className = "";
    setTimeout(connectCtrl, 1000);
  };
  wsCtrl.onerror = () => wsCtrl.close();
}
connectCtrl();

// ── Telemetry WebSocket ──
let wsTele = null;
function connectTele() {
  wsTele = new WebSocket(`ws://${location.host}/ws/tele`);
  wsTele.onmessage = (ev) => { renderTelemetry(JSON.parse(ev.data)); };
  wsTele.onclose = () => { setTimeout(connectTele, 2000); };
  wsTele.onerror = () => wsTele.close();
}
connectTele();

function sendMsg(obj) { if (wsCtrl && wsCtrl.readyState === 1) wsCtrl.send(JSON.stringify(obj)); }

function sendArmKeys() {
  sendMsg({type: "arm_keys", keys: [...armPressed]});
  document.querySelectorAll(".key").forEach(el => {
    el.classList.toggle("active", armPressed.has(el.dataset.key));
  });
}

function sendBaseKeys() {
  sendMsg({type: "base_keys", keys: [...basePressed]});
  document.querySelectorAll(".arrow-key").forEach(el => {
    el.classList.toggle("active", basePressed.has(el.dataset.dir));
  });
  let x = 0, th = 0;
  let linSpeed = parseFloat(document.getElementById("sl_lin").value);
  let angSpeed = parseFloat(document.getElementById("sl_ang").value);
  if (basePressed.has("up"))    x  += linSpeed;
  if (basePressed.has("down"))  x  -= linSpeed;
  if (basePressed.has("left"))  th -= angSpeed;
  if (basePressed.has("right")) th += angSpeed;
  document.getElementById("vel").textContent =
    `x: ${x.toFixed(2)} m/s   \\u03b8: ${th.toFixed(1)} \\u00b0/s`;
}

// Heartbeat
let hbId = null;
function startHB() {
  if (hbId) return;
  hbId = setInterval(() => {
    if (armPressed.size > 0) sendArmKeys();
    if (basePressed.size > 0) sendBaseKeys();
    if (armPressed.size === 0 && basePressed.size === 0) { clearInterval(hbId); hbId = null; }
  }, 50);
}

document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  // Actions
  if (k === " ") { e.preventDefault(); sendMsg({type: "action", action: "print"}); return; }
  if (k === "r") { e.preventDefault(); sendMsg({type: "action", action: "reset"}); return; }
  if (k === "m") { e.preventDefault(); sendMsg({type: "action", action: "zero_ft"}); return; }
  // Base (arrow keys)
  const dir = BASE_MAP[e.key];
  if (dir) { e.preventDefault(); basePressed.add(dir); sendBaseKeys(); startHB(); return; }
  // Arm
  if (ARM_KEYS.has(k)) { e.preventDefault(); armPressed.add(k); sendArmKeys(); startHB(); }
});

document.addEventListener("keyup", e => {
  const k = e.key.toLowerCase();
  const dir = BASE_MAP[e.key];
  if (dir) { e.preventDefault(); basePressed.delete(dir); sendBaseKeys(); return; }
  if (ARM_KEYS.has(k)) { e.preventDefault(); armPressed.delete(k); sendArmKeys(); }
});

// Touch/mouse for arm keys
document.querySelectorAll(".key").forEach(el => {
  const k = el.dataset.key;
  el.addEventListener("mousedown",  () => { armPressed.add(k); sendArmKeys(); startHB(); });
  el.addEventListener("mouseup",    () => { armPressed.delete(k); sendArmKeys(); });
  el.addEventListener("mouseleave", () => { armPressed.delete(k); sendArmKeys(); });
  el.addEventListener("touchstart", e => { e.preventDefault(); armPressed.add(k); sendArmKeys(); startHB(); });
  el.addEventListener("touchend",   e => { e.preventDefault(); armPressed.delete(k); sendArmKeys(); });
});

// Touch/mouse for base keys
document.querySelectorAll(".arrow-key").forEach(el => {
  const dir = el.dataset.dir;
  el.addEventListener("mousedown",  () => { basePressed.add(dir); sendBaseKeys(); startHB(); });
  el.addEventListener("mouseup",    () => { basePressed.delete(dir); sendBaseKeys(); });
  el.addEventListener("mouseleave", () => { basePressed.delete(dir); sendBaseKeys(); });
  el.addEventListener("touchstart", e => { e.preventDefault(); basePressed.add(dir); sendBaseKeys(); startHB(); });
  el.addEventListener("touchend",   e => { e.preventDefault(); basePressed.delete(dir); sendBaseKeys(); });
});

// Buttons
document.getElementById("btn_reset").addEventListener("click", () => sendMsg({type: "action", action: "reset"}));
document.getElementById("btn_zero").addEventListener("click", () => sendMsg({type: "action", action: "zero_ft"}));
document.getElementById("btn_print").addEventListener("click", () => sendMsg({type: "action", action: "print"}));

// Camera switching
document.querySelectorAll(".cam-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".cam-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const cam = btn.dataset.cam;
    const img = document.getElementById("cam");
    const wrap = document.getElementById("cam_wrap");
    if (cam === "off") {
      img.src = "";
      wrap.style.display = "none";
    } else {
      wrap.style.display = "";
      img.src = "/video?cam=" + cam;
      // Stereo: CSS crops left eye + rotates 180
      wrap.classList.toggle("stereo", cam === "cam2");
    }
  });
});

// Sliders
const slLin = document.getElementById("sl_lin");
const slAng = document.getElementById("sl_ang");
slLin.addEventListener("input", () => {
  document.getElementById("sl_lin_v").textContent = parseFloat(slLin.value).toFixed(2) + " m/s";
  sendMsg({type: "speed", linear: parseFloat(slLin.value), angular: parseFloat(slAng.value)});
});
slAng.addEventListener("input", () => {
  document.getElementById("sl_ang_v").innerHTML = parseFloat(slAng.value).toFixed(0) + " &deg;/s";
  sendMsg({type: "speed", linear: parseFloat(slLin.value), angular: parseFloat(slAng.value)});
});

function errColor(val, lo, hi) {
  if (val < lo) return "ok";
  if (val < hi) return "caution";
  return "danger";
}

function renderTelemetry(data) {
  let html = "";

  // Arm availability
  const armOk = data.arm_available !== false;
  const armPanel = document.getElementById("arm_panel");
  const armUnavail = document.getElementById("arm_unavailable");
  if (!armOk) {
    armUnavail.style.display = "block";
    armPanel.querySelectorAll(".key, .btn").forEach(el => { el.style.opacity = "0.3"; el.style.pointerEvents = "none"; });
    document.getElementById("ctrl_status").textContent = "ARM: N/A";
    document.getElementById("ctrl_status").style.color = "#f55";
  } else {
    armUnavail.style.display = "none";
    armPanel.querySelectorAll(".key, .btn").forEach(el => { el.style.opacity = ""; el.style.pointerEvents = ""; });
  }

  // Arm telemetry
  const d = data.arm;
  if (d) {
    document.getElementById("ctrl_status").textContent =
      "CTRL: " + (d.ctrl_enabled ? "ON" : "OFF");
    document.getElementById("ctrl_status").style.color = d.ctrl_enabled ? "#0f0" : "#f55";

    const ep = d.err_pos_mm || [0,0,0], er = d.err_rot_deg || [0,0,0];
    const epn = d.err_pos_norm || 0, ern = d.err_rot_norm || 0;
    const fe = d.f_ext || [0,0,0], me = d.m_ext || [0,0,0], fn = d.f_norm || 0;
    const ee = d.ee_pos || [0,0,0];
    const jp = d.joint_pos || [], jv = d.joint_vel || [], jt = d.joint_torque || [];
    const warn = d.joint_limit_warn || "";

    let jrows = "";
    for (let i = 0; i < jp.length; i++) {
      jrows += `<tr>
        <td class="jlabel">J${i+1}</td>
        <td>${(jp[i]*180/Math.PI).toFixed(1)}&deg;</td>
        <td>${jv[i].toFixed(2)}</td>
        <td>${jt[i].toFixed(2)}</td>
      </tr>`;
    }

    if (warn) html += `<div class="warn">JOINT LIMIT: ${warn}</div>`;

    html += `
      <div class="trow"><span class="label">Pos err (mm)</span><span class="value ${errColor(epn,3,15)}">X:${ep[0].toFixed(1)} Y:${ep[1].toFixed(1)} Z:${ep[2].toFixed(1)} |e|:${epn.toFixed(1)}</span></div>
      <div class="trow"><span class="label">Rot err (&deg;)</span><span class="value ${errColor(ern,2,8)}">X:${er[0].toFixed(1)} Y:${er[1].toFixed(1)} Z:${er[2].toFixed(1)} |e|:${ern.toFixed(1)}</span></div>
      <div class="tsep"></div>
      <div class="trow"><span class="label">EE pos (m)</span><span class="value">X:${ee[0].toFixed(3)} Y:${ee[1].toFixed(3)} Z:${ee[2].toFixed(3)}</span></div>
      <div class="trow"><span class="label">Polar</span><span class="value">r=${d.polar_r.toFixed(3)}m &theta;=${d.polar_theta.toFixed(1)}&deg; z=${d.polar_z.toFixed(3)}m</span></div>
      <div class="tsep"></div>
      <div class="trow"><span class="label">F_ext (N)</span><span class="value ${errColor(fn,1,5)}">X:${fe[0].toFixed(2)} Y:${fe[1].toFixed(2)} Z:${fe[2].toFixed(2)} |F|:${fn.toFixed(2)}</span></div>
      <div class="trow"><span class="label">M_ext (Nm)</span><span class="value">X:${me[0].toFixed(2)} Y:${me[1].toFixed(2)} Z:${me[2].toFixed(2)}</span></div>
      <div class="tsep"></div>
      <div class="trow"><span class="label">Gripper</span><span class="value">des:${d.gripper_des.toFixed(3)} act:${d.gripper_act.toFixed(3)} ${d.gripper_pct.toFixed(0)}%</span></div>
      <div class="tsep"></div>
      <table class="joint-table">
        <tr><th></th><th>pos</th><th>vel</th><th>torque</th></tr>
        ${jrows}
      </table>
    `;
  }

  if (!armOk && !d) {
    html += `<div class="trow"><span class="value" style="color:#f55">Arm not connected</span></div><div class="tsep"></div>`;
  }

  // Base telemetry
  const b = data.base;
  if (b && !b.error) {
    const L = b.left, R = b.right;
    html += `
      <div class="base-tele">
        <h4>Base</h4>
        <div class="trow"><span class="label">Battery</span><span class="value" style="color:${b.batt_pct<20?'#f55':b.batt_pct<50?'#fa0':'#0f0'}">${b.batt_pct}% (${b.vbus}V)</span></div>
        <div class="trow"><span class="label">L</span><span class="value">${L.current_A}A ${L.torque_Nm}Nm ${L.vel_mps}m/s</span></div>
        <div class="trow"><span class="label">R</span><span class="value">${R.current_A}A ${R.torque_Nm}Nm ${R.vel_mps}m/s</span></div>
      </div>
    `;

    // Sync slider values
    if (b.settings) {
      const sl = document.getElementById("sl_lin");
      const sa = document.getElementById("sl_ang");
      if (!sl._userTouched && Math.abs(parseFloat(sl.value) - b.settings.linear_speed) > 0.001) {
        sl.value = b.settings.linear_speed;
        document.getElementById("sl_lin_v").textContent = b.settings.linear_speed.toFixed(2) + " m/s";
      }
      if (!sa._userTouched && Math.abs(parseFloat(sa.value) - b.settings.angular_speed) > 0.1) {
        sa.value = b.settings.angular_speed;
        document.getElementById("sl_ang_v").innerHTML = b.settings.angular_speed.toFixed(0) + " &deg;/s";
      }
    }
  } else if (b && b.error) {
    html += `<div class="base-tele"><h4>Base</h4><div class="trow"><span class="value" style="color:#f55">Error: ${b.error}</span></div></div>`;
  }

  document.getElementById("tele_body").innerHTML = html || "Waiting for data...";
}
</script>
</body>
</html>"""


# ===================== ROUTES =====================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/video")
async def video_feed(cam: str = "cam1"):
    if cam not in ("cam1", "cam2"):
        cam = "cam1"
    return StreamingResponse(
        mjpeg_generator(cam),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws_control(ws: WebSocket):
    global last_arm_key_time, last_base_key_time, wrench_bias
    global linear_speed, angular_speed
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
