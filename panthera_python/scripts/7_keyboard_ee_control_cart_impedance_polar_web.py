#!/usr/bin/env python3
"""
Keyboard Cartesian impedance control -- Web version (Polar Cartesian Impedance Control + FastAPI/WebSocket)

Based on 7_keyboard_ee_control_cart_impedance_polar.py, with pygame replaced by a web UI.
Remote control via browser, no X11 forwarding required.

Architecture: control thread + asyncio event loop (mimics the ODrive web version's two-thread structure)
  Control thread (2000Hz) -- FK + analytical Jacobian + Cartesian impedance -> joint torques + external wrench estimation
  async task (250Hz)      -- WebSocket key state, updates Cartesian target pose x_des in polar coordinates

Usage:
    source <repo>/.venv/bin/activate
    cd <repo>/panthera_python/scripts
    python 7_keyboard_ee_control_cart_impedance_polar_web.py

Then open http://<jetson-ip>:8082 in your PC browser.

Horizontal polar keyboard mapping:
  W/S: Radial motion (away/towards base center), orientation unchanged
  A/D: Rotate about base Z axis; position and orientation rotate together
  Q/E: Vertical

Orientation control (EE frame):
  I/K: Pitch +/-    J/L: Yaw +/-    U/O: Roll +/-

Gripper: Z=close  X=open
Other:   Space=print pose  R=return home  M=zero wrench
"""

import asyncio
import json
import logging
import subprocess
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from scipy.spatial.transform import Rotation as Rot
import pinocchio as pin

from Panthera_lib import Panthera

# --------------- CONFIG ---------------
HOST = "0.0.0.0"
PORT = 8082
TELEMETRY_HZ = 20
IDLE_TIMEOUT = 0.3  # seconds with no key -> stop updating x_des

POS_STEP        = 0.002             # Per-frame position step (m) - 3x original
ROT_STEP        = 0.2               # Per-frame rotation step (degrees) - 3x original (was 0.2 in original)
POLAR_ANGLE_MAX = np.radians(1.5)   # Max polar rotation step (rad/frame)
CTRL_FREQ_IMP   = 2000              # Control thread frequency (Hz)
KEY_CMD_HZ      = 200               # Keyboard thread frequency (Hz)

MAX_TORQUE = [21.0, 36.0, 36.0, 30.0, 10.0, 10.0]
JOINT_VEL  = [0.5] * 6

# Cartesian space stiffness
K_POS = np.array([30.0, 30.0, 50.0])     # N/m
K_ROT = np.array([30.0, 30.0, 40.0])     # Nm/rad
K_CART = np.concatenate([K_POS, K_ROT])

# Cartesian space damping
B_POS = np.array([0.0, 0.0, 0.0])
B_ROT = np.array([0.0, 0.0, 0.0])
B_CART = np.concatenate([B_POS, B_ROT])

# DLS singularity regularization parameter
LAMBDA_DAMP = 0.05

# Joint torque cap
TAU_LIMIT = np.array([20.0, 30.0, 30.0, 20.0, 10.0, 10.0])

# Joint-space damping
JOINT_DAMPING = np.array([1.5, 2, 2, 2, 1, 0.8])

# Joint friction compensation
IMP_Fc         = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01])
IMP_Fv         = np.array([0.03, 0.03, 0.03, 0.03, 0.01, 0.01])
IMP_VEL_THRESH = 0.03

# Velocity low-pass cutoff frequency (Hz)
DQ_LPF_CUTOFF = 40.0

# Whether to enable Coriolis compensation
ENABLE_CORIOLIS = False

# tool offset
TOOL_OFFSET = np.array([0.165, 0.0, 0.0])

# gripper control parameters
GRIPPER_STEP        = 0.02
GRIPPER_KP          = 8.0
GRIPPER_KD          = 0.5
GRIPPER_MIN_DEFAULT = 0.0
GRIPPER_MAX_DEFAULT = 1.6

# External-force estimation low-pass cutoff frequency
FT_CUTOFF_FREQ = 5.0   # Hz
# External-force magnitude warning thresholds (N)
FT_WARN_LO = 1.0
FT_WARN_HI = 5.0

# Joint limit safety margin
JOINT_LIMIT_MARGIN = 0.1
JOINT_LIMIT_WARN_DURATION = 1.0

# Initial end-effector pose
HOME_POS       = [0.24, 0.0, 0.15]
HOME_ROT_EULER = (0.0, np.pi / 2, 0.0)
# --------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("panthera_cart_web")

# --- Key mapping (browser key -> (delta_pos, rot_euler)) ---
rot_step_rad = np.radians(ROT_STEP)
# W/S/A/D handled separately as polar coordinates
KEY_MAP = {
    "q": (np.array([0,  0,  POS_STEP]),   None),
    "e": (np.array([0,  0, -POS_STEP]),   None),
    "i": (np.zeros(3), ( 0,  rot_step_rad,  0)),
    "k": (np.zeros(3), ( 0, -rot_step_rad,  0)),
    "j": (np.zeros(3), ( 0,  0,  rot_step_rad)),
    "l": (np.zeros(3), ( 0,  0, -rot_step_rad)),
    "u": (np.zeros(3), ( rot_step_rad,  0,  0)),
    "o": (np.zeros(3), (-rot_step_rad,  0,  0)),
}

# ── Global state ──
current_keys: set[str] = set()
last_key_time: float = 0.0
telemetry_clients: list[WebSocket] = []

# Robot state (set during init)
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

# Joint limits
jl_lower = None
jl_upper = None

# Joint limit warning
joint_limit_warn = ""
joint_limit_warn_time = 0.0

# Feasible target rollback
last_feasible_pos = None
last_feasible_rot = None

# Wrench bias (M key)
wrench_bias = np.zeros(6)

# Raw telemetry from control thread (numpy arrays, minimal formatting)
tele_data_raw = {}
# Formatted telemetry for WebSocket broadcast (built by async telemetry_loop)
tele_data = {}


# ─── Math utilities ────────────────────────────────────────────────

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


def print_pose(pos, rot, label="End-effector pose"):
    euler = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
    log.info("[%s] pos=[%.4f, %.4f, %.4f] rot=[%.1f, %.1f, %.1f]",
             label, *pos, *euler)


# ─── Init ────────────────────────────────────────────────────

def init_robot():
    global robot, x_des_pos, x_des_rot, gripper_des, gripper_min, gripper_max
    global ctrl_enabled, stop_event, init_joints
    global jl_lower, jl_upper, last_feasible_pos, last_feasible_rot

    robot = Panthera()
    zero_pos = [0.0] * robot.motor_count

    log.info("Initializing: moving to zero position...")
    robot.Joint_Pos_Vel(zero_pos, JOINT_VEL, MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    home_rot = robot.rotation_matrix_from_euler(*HOME_ROT_EULER)
    init_joints = robot.inverse_kinematics(HOME_POS, home_rot, robot.get_current_pos())
    if init_joints is None:
        log.warning("Home IK failed, staying at zero")
    else:
        log.info("Moving to home position: %s", HOME_POS)
        robot.moveJ(init_joints, duration=3.0, max_tqu=MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    # Raise the end-effector 0.1 m along Z
    fk_after_home = robot.forward_kinematics()
    p_lift = np.array(fk_after_home['position'])
    R_lift = np.array(fk_after_home['rotation'])
    p_lift[2] += 0.1

    log.info("Lifting EE +0.1m along Z, target Z=%.3f m ...", p_lift[2])
    success = robot.moveL(
        target_position=p_lift,
        target_rotation=R_lift,
        duration=2.0,
        use_spline=True
    )
    if not success:
        log.warning("moveL lift failed, using current position as start")
    time.sleep(0.5)

    # Read current pose as the initial Cartesian target
    q0 = np.array(robot.get_current_pos())
    ctrl_data = robot.model.createData()
    p0, R0, _ = compute_fk_and_jacobian(robot, ctrl_data, q0)
    print_pose(p0, R0, "Cartesian impedance start pose")

    gripper_min = robot.gripper_limits['lower'] if robot.gripper_limits else GRIPPER_MIN_DEFAULT
    gripper_max = robot.gripper_limits['upper'] if robot.gripper_limits else GRIPPER_MAX_DEFAULT
    log.info("Gripper limits: [%.2f, %.2f] rad", gripper_min, gripper_max)

    # Joint limits
    jl_lower = np.array(robot.joint_limits['lower']) + JOINT_LIMIT_MARGIN
    jl_upper = np.array(robot.joint_limits['upper']) - JOINT_LIMIT_MARGIN
    for i in range(len(jl_lower)):
        log.info("  J%d: [%+.2f, %+.2f] rad", i+1, jl_lower[i], jl_upper[i])

    x_des_pos = p0.copy()
    x_des_rot = R0.copy()
    last_feasible_pos = p0.copy()
    last_feasible_rot = R0.copy()
    gripper_des = float(np.clip(robot.get_current_pos_gripper(), gripper_min, gripper_max))

    ctrl_enabled.set()

    return p0, R0


# ─── Impedance control thread ───────────────────────────────

def impedance_loop():
    global joint_limit_warn, joint_limit_warn_time
    global last_feasible_pos, last_feasible_rot
    global x_des_pos, x_des_rot

    _z6 = [0.0] * 6
    dt = 1.0 / CTRL_FREQ_IMP
    ctrl_data = robot.model.createData()

    # External-force estimation low-pass filter state
    F_ext_filtered = np.zeros(6)
    alpha_ft = (2 * np.pi * FT_CUTOFF_FREQ * dt) / (1 + 2 * np.pi * FT_CUTOFF_FREQ * dt)

    tele_counter = 0

    while not stop_event.is_set():
        t0 = time.time()

        if ctrl_enabled.is_set():
            q = np.array(robot.get_current_pos())
            dq = np.array(robot.get_current_vel())

            # --- Joint limit check ---
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

            # 6D Cartesian error
            e_pos = p_des - p_cur
            e_rot = orientation_error_axis_angle(R_des, R_cur)
            e_x = np.concatenate([e_pos, e_rot])

            # Cartesian velocity
            dx = J @ dq

            # Cartesian spring-damper force
            F = K_CART * e_x - B_CART * dx

            # Damped least-squares mapping
            JJT = J @ J.T
            alpha_dls = np.linalg.solve(JJT + LAMBDA_DAMP**2 * np.eye(6), F)
            tor_cart = J.T @ alpha_dls

            # Joint-space damping
            tor_joint_damp = -JOINT_DAMPING * dq

            # Gravity + Coriolis (optional) + friction compensation
            tor_gra = np.array(robot.get_Gravity())
            tor_cor = np.array(robot.get_Coriolis_vector(q, dq)) if ENABLE_CORIOLIS else np.zeros(6)
            tor_fri = np.array(robot.get_friction_compensation(
                dq, IMP_Fc, IMP_Fv, IMP_VEL_THRESH))

            tor = np.clip(tor_cart + tor_joint_damp + tor_gra + tor_cor + tor_fri,
                          -TAU_LIMIT, TAU_LIMIT)

            # gripper + arm torque commands
            robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                g_des, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)
            robot.pos_vel_tqe_kp_kd(_z6, _z6, tor.tolist(), _z6, _z6)

            # --- External wrench estimation ---
            tau_measured = np.array(robot.get_current_torque())
            tau_model = tor_gra + tor_cor + tor_fri
            tau_ext = tau_measured - tau_model

            JJT_ft = J @ J.T
            F_ext_raw = J @ np.linalg.solve(
                JJT_ft + LAMBDA_DAMP**2 * np.eye(6), tau_ext)

            F_ext_filtered[:] = alpha_ft * F_ext_raw + (1 - alpha_ft) * F_ext_filtered

            # Store raw telemetry data every 10th iteration (~200Hz)
            # Only store raw values here — formatting (round, tolist) is done
            # in the async telemetry_loop to minimize GIL hold time.
            tele_counter += 1
            if tele_counter >= 10:
                tele_counter = 0
                g_actual = robot.get_current_pos_gripper()
                wrench_disp = F_ext_filtered - wrench_bias
                # Store raw numpy arrays — single dict assignment is GIL-atomic
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
        # Minimum 50μs sleep guarantees a real kernel context switch and GIL release.
        # time.sleep(0) on Linux ARM can return instantly without yielding.
        # 50μs is only 10% of the 500μs budget — negligible impact on control frequency.
        time.sleep(max(0.00005, remaining))


# ─── Key command async task (runs on event loop, no GIL competition) ──

async def key_command_loop():
    global gripper_des, x_des_pos, x_des_rot
    dt = 1.0 / KEY_CMD_HZ

    while not stop_event.is_set():
        now = time.monotonic()

        if ctrl_enabled.is_set() and current_keys and (now - last_key_time < IDLE_TIMEOUT):
            combined_pos = np.zeros(3)
            first_rot_euler = None
            any_pressed = False

            for k in list(current_keys):
                if k in KEY_MAP:
                    dp, rot_euler = KEY_MAP[k]
                    combined_pos += dp
                    if rot_euler is not None and first_rot_euler is None:
                        first_rot_euler = rot_euler
                    any_pressed = True

            # --- Polar handling (W/S=radial, A/D=rotate about Z) ---
            with lock:
                px, py = x_des_pos[0], x_des_pos[1]
            r_horiz = np.hypot(px, py)

            if "a" in current_keys or "d" in current_keys:
                dtheta = POS_STEP / max(r_horiz, 0.01)
                dtheta = min(dtheta, POLAR_ANGLE_MAX)
                if "d" in current_keys:
                    dtheta = -dtheta
                c, s = np.cos(dtheta), np.sin(dtheta)
                Rz = np.array([[c, -s, 0.0],
                               [s,  c, 0.0],
                               [0.0, 0.0, 1.0]])
                with lock:
                    x_des_pos[:] = Rz @ x_des_pos
                    x_des_rot[:] = Rz @ x_des_rot
                any_pressed = True

            if "w" in current_keys or "s" in current_keys:
                if r_horiz > 0.001:
                    radial_dir = np.array([px, py, 0.0]) / r_horiz
                else:
                    radial_dir = np.array([1.0, 0.0, 0.0])
                dr = POS_STEP if "w" in current_keys else -POS_STEP
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

            # gripper control
            if "z" in current_keys or "x" in current_keys:
                with lock:
                    if "z" in current_keys:
                        gripper_des = max(gripper_min, gripper_des - GRIPPER_STEP)
                    if "x" in current_keys:
                        gripper_des = min(gripper_max, gripper_des + GRIPPER_STEP)

        await asyncio.sleep(dt)


# ─── Reset ───────────────────────────────────────────────────

def do_reset():
    global gripper_des
    ctrl_enabled.clear()
    log.info("Resetting to home position...")
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


# ── Telemetry broadcast (separate from control WS) ──
async def telemetry_loop():
    global tele_data
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if telemetry_clients:
            raw = tele_data_raw.copy()  # GIL-safe snapshot
            if raw:
                # Format raw numpy arrays into JSON-friendly dicts
                # This runs on the event loop — no GIL competition with control thread
                e_pos = raw["e_pos"]
                e_rot = raw["e_rot"]
                p_cur = raw["p_cur"]
                p_des = raw["p_des"]
                wrench = raw["wrench"]
                g_des = raw["g_des"]
                polar_r = np.hypot(p_des[0], p_des[1])
                polar_theta = np.degrees(np.arctan2(p_des[1], p_des[0]))
                g_pct = (g_des - gripper_min) / max(gripper_max - gripper_min, 1e-6) * 100

                data = {
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
                tele_data = data
                msg = json.dumps(data)
                dead = []
                for ws in telemetry_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    telemetry_clients.remove(ws)
        await asyncio.sleep(interval)


# ── App lifecycle ──
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init_robot)

    # Only impedance_loop is a thread (2000Hz real-time control needs dedicated thread)
    # key_command_loop + telemetry_loop are async tasks on the event loop (like ODrive)
    t_ctrl = threading.Thread(target=impedance_loop, daemon=True)
    t_ctrl.start()

    t_keys = asyncio.create_task(key_command_loop())
    t_tele = asyncio.create_task(telemetry_loop())
    yield
    stop_event.set()
    t_keys.cancel()
    t_tele.cancel()
    t_ctrl.join(timeout=2)
    log.info("Shutdown complete (motors hold position until timeout)")


app = FastAPI(lifespan=lifespan)


# ── HTML ──
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panthera Cartesian Impedance (Polar)</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee;
    display: flex; flex-direction: column;
    align-items: center; padding: 1.2rem 1rem;
    min-height: 100vh; user-select: none;
  }
  h1 { font-size: 1.2rem; margin-bottom: 0.3rem; color: #e94560; }
  .status-bar { display: flex; gap: 1.5rem; font-size: 0.82rem; color: #888; margin-bottom: 0.8rem; }
  .status-bar .connected { color: #0f0; }

  .main { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; justify-content: center; }

  .control-panel { display: flex; flex-direction: column; align-items: center; min-width: 260px; }

  .key-section { margin-bottom: 0.7rem; text-align: center; }
  .key-section h3 { font-size: 0.75rem; color: #888; margin-bottom: 0.3rem; }
  .key-row { display: flex; gap: 5px; justify-content: center; margin-bottom: 3px; }
  .key {
    width: 46px; height: 40px;
    border: 2px solid #444; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: bold; background: #16213e;
    transition: all 0.08s; cursor: pointer;
  }
  .key.active { background: #e94560; border-color: #e94560; transform: scale(1.05); }
  .key.wide { width: 62px; font-size: 0.72rem; }

  .actions { display: flex; gap: 6px; margin-top: 0.5rem; }
  .btn {
    padding: 5px 14px; border: 2px solid #444; border-radius: 7px;
    background: #16213e; color: #eee; font-size: 0.78rem; cursor: pointer;
  }
  .btn:hover { border-color: #e94560; }
  .btn:active { background: #e94560; }

  .telemetry {
    background: #16213e; border-radius: 10px; padding: 0.8rem 1rem;
    min-width: 380px; font-family: monospace; font-size: 0.75rem; line-height: 1.55;
  }
  .telemetry h2 { font-size: 0.85rem; color: #e94560; margin-bottom: 0.4rem; font-family: sans-serif; }
  .trow { display: flex; justify-content: space-between; gap: 0.8rem; }
  .trow .label { color: #888; white-space: nowrap; }
  .trow .value { color: #eee; text-align: right; }
  .tsep { border-top: 1px solid #333; margin: 0.25rem 0; }

  .warn { color: #ff4444; font-weight: bold; animation: blink 0.5s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  .ok { color: #4caf50; }
  .caution { color: #ddc640; }
  .danger { color: #e94560; }

  .joint-table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  .joint-table th { color: #888; font-weight: normal; text-align: right; padding: 1px 3px; }
  .joint-table td { color: #eee; text-align: right; padding: 1px 3px; }
  .joint-table .jlabel { color: #e94560; text-align: left; }
</style>
</head>
<body>
<h1>Cartesian Impedance Control (Polar)</h1>
<div class="status-bar">
  <span id="status">Connecting...</span>
  <span id="latency">Latency: -- ms</span>
  <span id="ctrl_status">CTRL: --</span>
</div>

<div class="main">
  <div class="control-panel">
    <div class="key-section">
      <h3>Position (Polar)</h3>
      <div class="key-row">
        <div class="key" data-key="q">Q<br><span style="font-size:0.55rem;color:#888">Z+</span></div>
        <div class="key" data-key="w">W<br><span style="font-size:0.55rem;color:#888">Rad+</span></div>
        <div class="key" data-key="e">E<br><span style="font-size:0.55rem;color:#888">Z-</span></div>
      </div>
      <div class="key-row">
        <div class="key" data-key="a">A<br><span style="font-size:0.55rem;color:#888">CCW</span></div>
        <div class="key" data-key="s">S<br><span style="font-size:0.55rem;color:#888">Rad-</span></div>
        <div class="key" data-key="d">D<br><span style="font-size:0.55rem;color:#888">CW</span></div>
      </div>
    </div>

    <div class="key-section">
      <h3>Orientation (EE frame)</h3>
      <div class="key-row">
        <div class="key" data-key="u">U<br><span style="font-size:0.55rem;color:#888">Ro+</span></div>
        <div class="key" data-key="i">I<br><span style="font-size:0.55rem;color:#888">Pi+</span></div>
        <div class="key" data-key="o">O<br><span style="font-size:0.55rem;color:#888">Ro-</span></div>
      </div>
      <div class="key-row">
        <div class="key" data-key="j">J<br><span style="font-size:0.55rem;color:#888">Ya+</span></div>
        <div class="key" data-key="k">K<br><span style="font-size:0.55rem;color:#888">Pi-</span></div>
        <div class="key" data-key="l">L<br><span style="font-size:0.55rem;color:#888">Ya-</span></div>
      </div>
    </div>

    <div class="key-section">
      <h3>Gripper</h3>
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

  <div class="telemetry">
    <h2>Telemetry</h2>
    <div id="tele_body">Waiting for data...</div>
  </div>
</div>

<script>
const ALL_KEYS = new Set(["w","a","s","d","q","e","i","k","j","l","u","o","z","x"]);
const pressed = new Set();

// ── Control WebSocket (keys + ping/pong) — lightweight, latency-critical ──
let wsCtrl = null;
let pingId = null;
function connectCtrl() {
  if (pingId) { clearInterval(pingId); pingId = null; }
  wsCtrl = new WebSocket(`ws://${location.host}/ws`);
  wsCtrl.onopen = () => {
    document.getElementById("status").textContent = "Control: OK";
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
      document.getElementById("latency").textContent = `Ctrl latency: ${rtt.toFixed(1)} ms`;
    }
  };
  wsCtrl.onclose = () => {
    document.getElementById("status").textContent = "Control: disconnected";
    document.getElementById("status").className = "";
    setTimeout(connectCtrl, 1000);
  };
  wsCtrl.onerror = () => wsCtrl.close();
}
connectCtrl();

// ── Telemetry WebSocket (display data) — separate channel, no impact on control ──
let wsTele = null;
function connectTele() {
  wsTele = new WebSocket(`ws://${location.host}/ws/tele`);
  wsTele.onmessage = (ev) => { renderTelemetry(JSON.parse(ev.data)); };
  wsTele.onclose = () => { setTimeout(connectTele, 2000); };
  wsTele.onerror = () => wsTele.close();
}
connectTele();

function sendMsg(obj) { if (wsCtrl && wsCtrl.readyState === 1) wsCtrl.send(JSON.stringify(obj)); }

function sendKeys() {
  sendMsg({type: "keys", keys: [...pressed]});
  document.querySelectorAll(".key").forEach(el => {
    el.classList.toggle("active", pressed.has(el.dataset.key));
  });
}

let hbId = null;
function startHB() {
  if (hbId) return;
  hbId = setInterval(() => {
    if (pressed.size > 0) sendKeys();
    else { clearInterval(hbId); hbId = null; }
  }, 50);
}

document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); sendMsg({type: "action", action: "print"}); return; }
  if (k === "r") { e.preventDefault(); sendMsg({type: "action", action: "reset"}); return; }
  if (k === "m") { e.preventDefault(); sendMsg({type: "action", action: "zero_ft"}); return; }
  if (!ALL_KEYS.has(k)) return;
  e.preventDefault();
  pressed.add(k); sendKeys(); startHB();
});
document.addEventListener("keyup", e => {
  const k = e.key.toLowerCase();
  if (!ALL_KEYS.has(k)) return;
  e.preventDefault();
  pressed.delete(k); sendKeys();
});

document.querySelectorAll(".key").forEach(el => {
  const k = el.dataset.key;
  el.addEventListener("mousedown",  () => { pressed.add(k); sendKeys(); startHB(); });
  el.addEventListener("mouseup",    () => { pressed.delete(k); sendKeys(); });
  el.addEventListener("mouseleave", () => { pressed.delete(k); sendKeys(); });
  el.addEventListener("touchstart", e => { e.preventDefault(); pressed.add(k); sendKeys(); startHB(); });
  el.addEventListener("touchend",   e => { e.preventDefault(); pressed.delete(k); sendKeys(); });
});

document.getElementById("btn_reset").addEventListener("click", () => sendMsg({type: "action", action: "reset"}));
document.getElementById("btn_zero").addEventListener("click", () => sendMsg({type: "action", action: "zero_ft"}));
document.getElementById("btn_print").addEventListener("click", () => sendMsg({type: "action", action: "print"}));

function errColor(val, lo, hi) {
  if (val < lo) return "ok";
  if (val < hi) return "caution";
  return "danger";
}

function renderTelemetry(d) {
  document.getElementById("ctrl_status").textContent =
    "CTRL: " + (d.ctrl_enabled ? "ON" : "OFF (resetting)");
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

  let warnHtml = "";
  if (warn) {
    warnHtml = `<div class="warn">JOINT LIMIT: ${warn}</div>`;
  }

  document.getElementById("tele_body").innerHTML = `
    ${warnHtml}
    <div class="trow"><span class="label">Pos err (mm)</span><span class="value ${errColor(epn,3,15)}">X:${ep[0].toFixed(1)} Y:${ep[1].toFixed(1)} Z:${ep[2].toFixed(1)} |e|:${epn.toFixed(1)}</span></div>
    <div class="trow"><span class="label">Rot err (&deg;)</span><span class="value ${errColor(ern,2,8)}">X:${er[0].toFixed(1)} Y:${er[1].toFixed(1)} Z:${er[2].toFixed(1)} |e|:${ern.toFixed(1)}</span></div>
    <div class="tsep"></div>
    <div class="trow"><span class="label">EE pos (m)</span><span class="value">X:${ee[0].toFixed(3)} Y:${ee[1].toFixed(3)} Z:${ee[2].toFixed(3)}</span></div>
    <div class="trow"><span class="label">Polar</span><span class="value">r=${d.polar_r.toFixed(3)}m  &theta;=${d.polar_theta.toFixed(1)}&deg;  z=${d.polar_z.toFixed(3)}m</span></div>
    <div class="tsep"></div>
    <div class="trow"><span class="label">F_ext (N)</span><span class="value ${errColor(fn, 1, 5)}">X:${fe[0].toFixed(2)} Y:${fe[1].toFixed(2)} Z:${fe[2].toFixed(2)} |F|:${fn.toFixed(2)}</span></div>
    <div class="trow"><span class="label">M_ext (Nm)</span><span class="value">X:${me[0].toFixed(2)} Y:${me[1].toFixed(2)} Z:${me[2].toFixed(2)}</span></div>
    <div class="tsep"></div>
    <div class="trow"><span class="label">Gripper</span><span class="value">des:${d.gripper_des.toFixed(3)} act:${d.gripper_act.toFixed(3)} ${d.gripper_pct.toFixed(0)}%</span></div>
    <div class="trow"><span class="label">Params</span><span class="value">K_p=[${""" + ",".join(str(int(x)) for x in K_POS) + """}] K_r=[${""" + ",".join(str(int(x)) for x in K_ROT) + """}]</span></div>
    <div class="tsep"></div>
    <table class="joint-table">
      <tr><th></th><th>pos</th><th>vel</th><th>torque</th></tr>
      ${jrows}
    </table>
  `;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.websocket("/ws")
async def ws_control(ws: WebSocket):
    """Control-only WebSocket: keys + ping/pong + actions. Lightweight."""
    global last_key_time, wrench_bias
    await ws.accept()
    log.info("Control WS connected")
    try:
        while True:
            data = json.loads(await ws.receive_text())
            msg_type = data.get("type", "keys")

            if msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong", "t": data.get("t", 0)}))

            elif msg_type == "keys":
                keys = set(data.get("keys", []))
                current_keys.clear()
                current_keys.update(keys)
                last_key_time = time.monotonic()

            elif msg_type == "action":
                action = data.get("action", "")
                if action == "reset":
                    threading.Thread(target=do_reset, daemon=True).start()
                elif action == "zero_ft":
                    current_wrench = np.array(tele_data.get("f_ext", [0,0,0]) +
                                               tele_data.get("m_ext", [0,0,0]))
                    wrench_bias += current_wrench
                    log.info("[M] F/T sensor zeroed")
                elif action == "print":
                    with lock:
                        p_snap = x_des_pos.copy()
                        R_snap = x_des_rot.copy()
                    print_pose(p_snap, R_snap, "Current target pose")

    except WebSocketDisconnect:
        current_keys.clear()
        log.info("Control WS disconnected")


@app.websocket("/ws/tele")
async def ws_telemetry(ws: WebSocket):
    """Telemetry-only WebSocket: server pushes display data. Separate from control."""
    await ws.accept()
    telemetry_clients.append(ws)
    log.info("Telemetry WS connected (total: %d)", len(telemetry_clients))
    try:
        while True:
            # Keep connection alive by reading (client won't send much)
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
