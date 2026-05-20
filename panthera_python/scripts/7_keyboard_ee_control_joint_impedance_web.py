#!/usr/bin/env python3
"""
Keyboard end-effector control - Web version (Joint Impedance + FastAPI/WebSocket)

Based on 7_keyboard_ee_control_joint_impedance.py, with pygame replaced by a web UI.
Remote control via browser, no X11 forwarding required.

Architecture:
  Web keyboard thread  -- WebSocket receives key state, pushes incremental commands to ik_queue
  IK thread            -- consumes ik_queue, computes inverse kinematics, updates shared q_des
  Control thread (1500Hz) -- Joint impedance control with strict timing, unaffected by IK latency
  Telemetry thread (20Hz) -- broadcasts joint/EE state to all browsers

Usage:
    source <repo>/.venv/bin/activate
    cd <repo>/panthera_python/scripts
    python 7_keyboard_ee_control_joint_impedance_web.py

Then open http://<jetson-ip>:8081 in your PC browser.

Position control (base frame):
  W / S  ->  X axis forward / backward
  A / D  ->  Y axis left / right
  Q / E  ->  Z axis up / down

Orientation control (EE frame):
  I / K  ->  Pitch about Y axis (+/-)
  J / L  ->  Yaw about Z axis (+/-)
  U / O  ->  Roll about X axis (+/-)

Gripper:  Z=close  X=open
Other:    Space=print pose  R=return home
"""

import asyncio
import json
import logging
import math
import queue
import subprocess
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from scipy.spatial.transform import Rotation as Rot

from Panthera_lib import Panthera

# --------------- CONFIG ---------------
HOST = "0.0.0.0"
PORT = 8081
TELEMETRY_HZ = 20
IDLE_TIMEOUT = 0.3  # seconds with no key -> stop sending IK cmds

POS_STEP      = 0.003   # Per-frame position step (m) - 3x faster than original
ROT_STEP      = 0.3     # Per-frame rotation step (degrees) - 3x faster than original
CTRL_FREQ_IMP = 1500    # Impedance control thread frequency (Hz)
KEY_CMD_HZ    = 200      # Key -> IK command frequency (Hz)

MAX_TORQUE = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
JOINT_VEL  = [0.5] * 6

# Joint impedance control parameters
IMP_K         = np.array([20.0,  25.0, 25.0,  20.0,  10.0, 5.0])
IMP_B         = np.array([1.0,   1.0,  1.0,   0.8,   0.4,  0.2])
IMP_TAU_LIMIT = np.array([10.0, 20.0, 20.0,  10.0,   5.0,  5.0])
IMP_Fc        = np.array([0.05, 0.05, 0.05,  0.05,   0.0,  0.0])
IMP_Fv        = np.array([0.03, 0.03, 0.03,  0.03,   0.0,  0.0])
IMP_VEL_THRESH = 0.02

# gripper control parameters
GRIPPER_STEP        = 0.02
GRIPPER_KP          = 8.0
GRIPPER_KD          = 0.5
GRIPPER_MIN_DEFAULT = 0.0
GRIPPER_MAX_DEFAULT = 1.6

# Initial end-effector pose
HOME_POS       = [0.24, 0.0, 0.15]
HOME_ROT_EULER = (0.0, np.pi / 2, 0.0)
# --------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("panthera_web")

# --- Key mapping (browser key name -> (delta_pos, rot_euler)) ---
rot_step_rad = np.radians(ROT_STEP)
KEY_MAP = {
    "w": (np.array([ POS_STEP,  0,  0]),  None),
    "s": (np.array([-POS_STEP,  0,  0]),  None),
    "a": (np.array([0,  POS_STEP,  0]),   None),
    "d": (np.array([0, -POS_STEP,  0]),   None),
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
q_des = None
q_cur_cache = None
gripper_des = 0.0
gripper_min = GRIPPER_MIN_DEFAULT
gripper_max = GRIPPER_MAX_DEFAULT
ctrl_enabled = threading.Event()
stop_event = threading.Event()
ik_queue = None
init_joints = None

# Telemetry snapshot
tele_lock = threading.Lock()
tele_data = {}


def init_robot():
    """Initialize robot, move to home, set up shared state."""
    global robot, q_des, q_cur_cache, gripper_des, gripper_min, gripper_max
    global ctrl_enabled, stop_event, ik_queue, init_joints

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

    fk = robot.forward_kinematics()
    target_pos = np.array(fk['position'])
    target_rot = np.array(fk['rotation'])
    euler = Rot.from_matrix(target_rot).as_euler('xyz', degrees=True)
    log.info("Home pose: pos=[%.3f, %.3f, %.3f] rot=[%.1f, %.1f, %.1f]",
             *target_pos, *euler)

    gripper_min = robot.gripper_limits['lower'] if robot.gripper_limits else GRIPPER_MIN_DEFAULT
    gripper_max = robot.gripper_limits['upper'] if robot.gripper_limits else GRIPPER_MAX_DEFAULT

    q_des = np.array(robot.get_current_pos())
    q_cur_cache = q_des.copy()
    gripper_des = float(np.clip(robot.get_current_pos_gripper(), gripper_min, gripper_max))

    ctrl_enabled.set()
    ik_queue = queue.Queue(maxsize=1)

    return target_pos, target_rot


def impedance_loop():
    """Joint impedance control at CTRL_FREQ_IMP Hz."""
    _z6 = [0.0] * 6
    dt = 1.0 / CTRL_FREQ_IMP
    while not stop_event.is_set():
        t0 = time.time()
        if ctrl_enabled.is_set():
            q  = np.array(robot.get_current_pos())
            dq = np.array(robot.get_current_vel())
            with lock:
                q_cur_cache[:] = q
                target = q_des.copy()
                g_des = gripper_des

            tor_imp = IMP_K * (target - q) + IMP_B * (-dq)
            tor_gra = np.array(robot.get_Gravity())
            tor_fri = np.array(robot.get_friction_compensation(
                dq, IMP_Fc, IMP_Fv, IMP_VEL_THRESH))
            tor = np.clip(tor_imp + tor_gra + tor_fri, -IMP_TAU_LIMIT, IMP_TAU_LIMIT)

            robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                g_des, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)
            robot.pos_vel_tqe_kp_kd(_z6, _z6, tor.tolist(), _z6, _z6)

            # Update telemetry
            fk = robot.forward_kinematics(q)
            ee_pos = fk['position']
            ee_rot = fk['rotation']
            ee_euler = Rot.from_matrix(ee_rot).as_euler('xyz', degrees=True)
            g_act = robot.get_current_pos_gripper()

            with tele_lock:
                tele_data.update({
                    "joint_pos": [round(x, 4) for x in q.tolist()],
                    "joint_vel": [round(x, 4) for x in dq.tolist()],
                    "joint_torque": [round(x, 4) for x in tor.tolist()],
                    "joint_des": [round(x, 4) for x in target.tolist()],
                    "ee_pos": [round(x, 4) for x in ee_pos],
                    "ee_euler": [round(x, 1) for x in ee_euler],
                    "gripper_des": round(g_des, 3),
                    "gripper_act": round(g_act, 3),
                    "gripper_pct": round(
                        (g_des - gripper_min) / max(gripper_max - gripper_min, 1e-6) * 100, 0),
                    "ctrl_enabled": ctrl_enabled.is_set(),
                })

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def ik_worker(ik_state):
    """Consume IK commands, update q_des."""
    while not stop_event.is_set():
        try:
            cmd = ik_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if cmd[0] == 'move':
            _, delta_pos, rot_euler = cmd
            new_pos = ik_state['pos'] + delta_pos

            if rot_euler is not None:
                new_rot = ik_state['rot'] @ robot.rotation_matrix_from_euler(*rot_euler)
                U, _, Vt = np.linalg.svd(new_rot)
                new_rot = U @ Vt
            else:
                new_rot = ik_state['rot']

            with lock:
                q_init = q_des.copy()
            joint_angles = robot.inverse_kinematics(new_pos, new_rot, q_init)

            if joint_angles is not None:
                ik_state['pos'] = new_pos
                ik_state['rot'] = new_rot
                with lock:
                    q_des[:] = joint_angles
            else:
                log.warning("IK failed for [%.3f, %.3f, %.3f]", *new_pos)

        elif cmd[0] == 'reset':
            _, new_pos, new_rot = cmd
            ik_state['pos'] = new_pos.copy()
            ik_state['rot'] = new_rot.copy()


def key_command_loop(ik_state):
    """Convert current_keys to IK commands at KEY_CMD_HZ."""
    global gripper_des
    dt = 1.0 / KEY_CMD_HZ
    while not stop_event.is_set():
        t0 = time.time()
        now = time.monotonic()

        if ctrl_enabled.is_set() and current_keys and (now - last_key_time < IDLE_TIMEOUT):
            combined_pos = np.zeros(3)
            first_rot_euler = None
            any_motion = False

            for k in current_keys:
                if k in KEY_MAP:
                    dp, rot_euler = KEY_MAP[k]
                    combined_pos += dp
                    if rot_euler is not None and first_rot_euler is None:
                        first_rot_euler = rot_euler
                    any_motion = True

            if any_motion:
                try:
                    ik_queue.put_nowait(('move', combined_pos, first_rot_euler))
                except queue.Full:
                    pass

            # Gripper
            if "z" in current_keys or "x" in current_keys:
                with lock:
                    if "z" in current_keys:
                        gripper_des = max(gripper_min, gripper_des - GRIPPER_STEP)
                    if "x" in current_keys:
                        gripper_des = min(gripper_max, gripper_des + GRIPPER_STEP)

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def do_reset(ik_state):
    """Reset to home position (blocking)."""
    global gripper_des
    ctrl_enabled.clear()
    log.info("Resetting to home position...")
    home_q = init_joints if init_joints is not None else [0.0] * robot.motor_count
    robot.moveJ(home_q, duration=2.0, max_tqu=MAX_TORQUE, iswait=True)
    with lock:
        q_des[:] = robot.get_current_pos()
    fk = robot.forward_kinematics()
    new_pos = np.array(fk['position'])
    new_rot = np.array(fk['rotation'])
    # Clear queue and sync IK state
    while not ik_queue.empty():
        try:
            ik_queue.get_nowait()
        except queue.Empty:
            break
    ik_queue.put(('reset', new_pos, new_rot))
    log.info("Reset complete")
    ctrl_enabled.set()


# ── Telemetry broadcast ──
async def telemetry_loop():
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if telemetry_clients:
            with tele_lock:
                data = tele_data.copy()
            if data:
                msg = json.dumps({"type": "telemetry", **data})
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

# We need ik_state accessible for reset
_ik_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    target_pos, target_rot = await loop.run_in_executor(None, init_robot)
    _ik_state['pos'] = target_pos.copy()
    _ik_state['rot'] = target_rot.copy()

    t_ctrl = threading.Thread(target=impedance_loop, daemon=True)
    t_ik = threading.Thread(target=ik_worker, args=(_ik_state,), daemon=True)
    t_keys = threading.Thread(target=key_command_loop, args=(_ik_state,), daemon=True)
    t_ctrl.start()
    t_ik.start()
    t_keys.start()

    t_tele = asyncio.create_task(telemetry_loop())
    yield
    stop_event.set()
    t_tele.cancel()
    t_ctrl.join(timeout=2)
    t_ik.join(timeout=2)
    t_keys.join(timeout=2)
    log.info("Shutdown complete (motors hold position until timeout)")


app = FastAPI(lifespan=lifespan)


# ── HTML ──
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panthera Joint Impedance Control</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee;
    display: flex; flex-direction: column;
    align-items: center; padding: 1.5rem 1rem;
    min-height: 100vh; user-select: none;
  }
  h1 { font-size: 1.3rem; margin-bottom: 0.3rem; color: #e94560; }
  .status-bar { display: flex; gap: 1.5rem; font-size: 0.82rem; color: #888; margin-bottom: 1rem; }
  .status-bar .connected { color: #0f0; }

  .main { display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap; justify-content: center; }

  /* Control panel */
  .control-panel { display: flex; flex-direction: column; align-items: center; min-width: 260px; }

  .key-section { margin-bottom: 0.8rem; text-align: center; }
  .key-section h3 { font-size: 0.8rem; color: #888; margin-bottom: 0.4rem; }
  .key-row { display: flex; gap: 6px; justify-content: center; margin-bottom: 4px; }
  .key {
    width: 48px; height: 42px;
    border: 2px solid #444; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: bold; background: #16213e;
    transition: all 0.08s; cursor: pointer;
  }
  .key.active { background: #e94560; border-color: #e94560; transform: scale(1.05); }
  .key.wide { width: 64px; font-size: 0.75rem; }

  .actions { display: flex; gap: 8px; margin-top: 0.6rem; }
  .btn {
    padding: 6px 16px; border: 2px solid #444; border-radius: 8px;
    background: #16213e; color: #eee; font-size: 0.8rem; cursor: pointer;
  }
  .btn:hover { border-color: #e94560; }
  .btn:active { background: #e94560; }

  /* Telemetry */
  .telemetry {
    background: #16213e; border-radius: 12px; padding: 1rem 1.2rem;
    min-width: 340px; font-family: monospace; font-size: 0.78rem; line-height: 1.6;
  }
  .telemetry h2 { font-size: 0.9rem; color: #e94560; margin-bottom: 0.5rem; font-family: sans-serif; }
  .trow { display: flex; justify-content: space-between; gap: 1rem; }
  .trow .label { color: #888; white-space: nowrap; }
  .trow .value { color: #eee; text-align: right; }
  .tsep { border-top: 1px solid #333; margin: 0.3rem 0; }

  .joint-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
  .joint-table th { color: #888; font-weight: normal; text-align: right; padding: 1px 4px; }
  .joint-table td { color: #eee; text-align: right; padding: 1px 4px; }
  .joint-table .jlabel { color: #e94560; text-align: left; }
</style>
</head>
<body>
<h1>Panthera Joint Impedance Control</h1>
<div class="status-bar">
  <span id="status">Connecting...</span>
  <span id="latency">Latency: -- ms</span>
  <span id="ctrl_status">CTRL: --</span>
</div>

<div class="main">
  <div class="control-panel">
    <div class="key-section">
      <h3>Position (base frame)</h3>
      <div class="key-row">
        <div class="key" data-key="q">Q<br><span style="font-size:0.6rem;color:#888">Z+</span></div>
        <div class="key" data-key="w">W<br><span style="font-size:0.6rem;color:#888">X+</span></div>
        <div class="key" data-key="e">E<br><span style="font-size:0.6rem;color:#888">Z-</span></div>
      </div>
      <div class="key-row">
        <div class="key" data-key="a">A<br><span style="font-size:0.6rem;color:#888">Y+</span></div>
        <div class="key" data-key="s">S<br><span style="font-size:0.6rem;color:#888">X-</span></div>
        <div class="key" data-key="d">D<br><span style="font-size:0.6rem;color:#888">Y-</span></div>
      </div>
    </div>

    <div class="key-section">
      <h3>Orientation (EE frame)</h3>
      <div class="key-row">
        <div class="key" data-key="u">U<br><span style="font-size:0.6rem;color:#888">Ro+</span></div>
        <div class="key" data-key="i">I<br><span style="font-size:0.6rem;color:#888">Pi+</span></div>
        <div class="key" data-key="o">O<br><span style="font-size:0.6rem;color:#888">Ro-</span></div>
      </div>
      <div class="key-row">
        <div class="key" data-key="j">J<br><span style="font-size:0.6rem;color:#888">Ya+</span></div>
        <div class="key" data-key="k">K<br><span style="font-size:0.6rem;color:#888">Pi-</span></div>
        <div class="key" data-key="l">L<br><span style="font-size:0.6rem;color:#888">Ya-</span></div>
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
      <button class="btn" id="btn_print">Space Print</button>
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
let ws = null;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    document.getElementById("status").textContent = "Connected";
    document.getElementById("status").className = "connected";
    setInterval(() => {
      if (ws.readyState === 1)
        ws.send(JSON.stringify({type: "ping", t: performance.now()}));
    }, 1000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "telemetry") renderTelemetry(msg);
    else if (msg.type === "pong") {
      const rtt = performance.now() - msg.t;
      document.getElementById("latency").textContent = `Latency: ${rtt.toFixed(1)} ms`;
    }
  };
  ws.onclose = () => {
    document.getElementById("status").textContent = "Disconnected - reconnecting...";
    document.getElementById("status").className = "";
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}
connect();

function sendMsg(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function sendKeys() {
  sendMsg({type: "keys", keys: [...pressed]});
  document.querySelectorAll(".key").forEach(el => {
    el.classList.toggle("active", pressed.has(el.dataset.key));
  });
}

// Heartbeat while keys held
let hbId = null;
function startHB() {
  if (hbId) return;
  hbId = setInterval(() => {
    if (pressed.size > 0) sendKeys();
    else { clearInterval(hbId); hbId = null; }
  }, 50);  // 20Hz heartbeat = matches KEY_CMD_HZ enough
}

document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); sendMsg({type: "action", action: "print"}); return; }
  if (k === "r") { e.preventDefault(); sendMsg({type: "action", action: "reset"}); return; }
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

// Touch/mouse on key buttons
document.querySelectorAll(".key").forEach(el => {
  const k = el.dataset.key;
  el.addEventListener("mousedown",  () => { pressed.add(k); sendKeys(); startHB(); });
  el.addEventListener("mouseup",    () => { pressed.delete(k); sendKeys(); });
  el.addEventListener("mouseleave", () => { pressed.delete(k); sendKeys(); });
  el.addEventListener("touchstart", e => { e.preventDefault(); pressed.add(k); sendKeys(); startHB(); });
  el.addEventListener("touchend",   e => { e.preventDefault(); pressed.delete(k); sendKeys(); });
});

document.getElementById("btn_reset").addEventListener("click", () => sendMsg({type: "action", action: "reset"}));
document.getElementById("btn_print").addEventListener("click", () => sendMsg({type: "action", action: "print"}));

function renderTelemetry(d) {
  document.getElementById("ctrl_status").textContent =
    "CTRL: " + (d.ctrl_enabled ? "ON" : "OFF (resetting)");
  document.getElementById("ctrl_status").style.color = d.ctrl_enabled ? "#0f0" : "#f55";

  const jp = d.joint_pos || [], jv = d.joint_vel || [], jt = d.joint_torque || [], jd = d.joint_des || [];
  const ep = d.ee_pos || [0,0,0], er = d.ee_euler || [0,0,0];

  let jrows = "";
  for (let i = 0; i < jp.length; i++) {
    jrows += `<tr>
      <td class="jlabel">J${i+1}</td>
      <td>${(jp[i]*180/Math.PI).toFixed(1)}&deg;</td>
      <td>${(jd[i]*180/Math.PI).toFixed(1)}&deg;</td>
      <td>${(jv[i]).toFixed(2)}</td>
      <td>${(jt[i]).toFixed(2)}</td>
    </tr>`;
  }

  document.getElementById("tele_body").innerHTML = `
    <div class="trow"><span class="label">EE pos (m)</span><span class="value">X:${ep[0].toFixed(3)} Y:${ep[1].toFixed(3)} Z:${ep[2].toFixed(3)}</span></div>
    <div class="trow"><span class="label">EE rot (&deg;)</span><span class="value">R:${er[0].toFixed(1)} P:${er[1].toFixed(1)} Y:${er[2].toFixed(1)}</span></div>
    <div class="trow"><span class="label">Gripper</span><span class="value">des:${d.gripper_des.toFixed(3)} act:${d.gripper_act.toFixed(3)} ${d.gripper_pct.toFixed(0)}%</span></div>
    <div class="tsep"></div>
    <table class="joint-table">
      <tr><th></th><th>pos</th><th>des</th><th>vel</th><th>torque</th></tr>
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
async def websocket_endpoint(ws: WebSocket):
    global last_key_time
    await ws.accept()
    telemetry_clients.append(ws)
    log.info("Browser connected (total: %d)", len(telemetry_clients))
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
                    threading.Thread(
                        target=do_reset, args=(_ik_state,), daemon=True
                    ).start()
                elif action == "print":
                    with lock:
                        q_snap = q_cur_cache.copy()
                    fk = robot.forward_kinematics(q_snap)
                    pos = fk['position']
                    rot = fk['rotation']
                    euler = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
                    log.info("EE Pose: pos=[%.4f, %.4f, %.4f] rot=[%.1f, %.1f, %.1f]",
                             *pos, *euler)

    except WebSocketDisconnect:
        current_keys.clear()
        if ws in telemetry_clients:
            telemetry_clients.remove(ws)
        log.info("Browser disconnected (total: %d)", len(telemetry_clients))


if __name__ == "__main__":
    try:
        ip = subprocess.check_output(
            "hostname -I", shell=True, text=True
        ).strip().split()[0]
        log.info("Open in browser: http://%s:%d", ip, PORT)
    except Exception:
        pass

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
