#!/usr/bin/env python3
"""Insight 9 + ODrive biwheel chassis web demo.

One process, two ports:
  - FastAPI on 8080: HTML page (keyboard + telemetry + viser iframe), /ws
  - viser   on 8082: 3D scene with VIO frustum (RGB rendered inside the cone)
                     + trail line.

Run (after `source .venv/bin/activate`; ROS + Fast DDS set by `install/setup_runtime_env.sh`):
    python scripts/2_insight_chassis_demo.py
"""

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import viser

# --------- CONFIG ---------
HTTP_PORT = 8080
VISER_PORT = 8082

LINEAR_SPEED = 0.15
ANGULAR_SPEED = 30.0
COMMAND_HZ = 20
TELEMETRY_HZ = 10
IDLE_TIMEOUT = 0.3

WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.25
INVERT_LEFT = True
INVERT_RIGHT = False
AXIS_LEFT = 1
AXIS_RIGHT = 0
TORQUE_CONSTANT = 8.27 / 270
BATT_CELLS = 5
BATT_FULL_V = 4.2 * BATT_CELLS
BATT_EMPTY_V = 3.0 * BATT_CELLS

TOPIC_POSE = "/insight/vio_100hz"
TOPIC_STATUS = "/insight/vio_status"
TOPIC_RGB = "/camera/camera/color/image_rect_raw/compressed"

RGB_FRAME_W = 1088
RGB_FRAME_H = 1920
RGB_FY = 776.78
RGB_STREAM_FPS = 15           # MJPEG cadence for the right panel
FRUSTUM_FOV_RAD = 2 * math.atan(RGB_FRAME_H / 2.0 / RGB_FY)
FRUSTUM_ASPECT = RGB_FRAME_W / RGB_FRAME_H
FRUSTUM_SCALE = 0.10

TRAIL_SAMPLE_HZ = 10         # how often viser_updater samples S.last_pose into the trail
TRAIL_HISTORY_SEC = 300      # 5 minutes of history
TRAIL_MAX_POINTS = TRAIL_SAMPLE_HZ * TRAIL_HISTORY_SEC   # = 3000

GRID_SIZE_M = 10.0           # ground grid extent (square)
GRID_CELL_M = 0.5            # minor cell edge length
GRID_SECTION_M = 2.0         # major division edge length
GRID_Z = -0.30               # ground plane height relative to VIO origin (meters)

ODRIVE_CONNECT_TIMEOUT = 8

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("chassis_demo")


# --------- shared state ---------
class S:
    lock = threading.Lock()
    keys: set = set()
    last_key_time: float = 0.0
    linear_speed: float = LINEAR_SPEED
    angular_speed: float = ANGULAR_SPEED

    odrv = None
    axis_l = None
    axis_r = None
    odrive_err: str = ""

    last_pose: tuple | None = None         # ((x,y,z), (qw,qx,qy,qz))
    last_pose_time: float = 0.0
    last_status: str = "(no vio status yet)"
    last_image_jpeg: bytes = b""
    last_image_time: float = 0.0

    # Peak DC bus current observed (absolute value); updated by read_telemetry on each tick.
    peak_ibus: float = 0.0

    # path_pts now owned and grown by viser_updater (10 Hz sampling), not on_pose


telemetry_clients: list[WebSocket] = []


# --------- ROS node ---------
class InsightSubs(Node):
    def __init__(self):
        super().__init__("insight_chassis_subs")
        cb = ReentrantCallbackGroup()
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=2)

        self.create_subscription(PoseStamped, TOPIC_POSE,
                                 self._on_pose, reliable_qos, callback_group=cb)
        self.create_subscription(String, TOPIC_STATUS,
                                 self._on_status, reliable_qos, callback_group=cb)
        self.create_subscription(CompressedImage, TOPIC_RGB,
                                 self._on_rgb, sensor_qos, callback_group=cb)

    def _on_pose(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        with S.lock:
            S.last_pose = ((p.x, p.y, p.z), (q.w, q.x, q.y, q.z))
            S.last_pose_time = time.time()

    def _on_status(self, msg: String):
        with S.lock:
            S.last_status = msg.data

    def _on_rgb(self, msg: CompressedImage):
        with S.lock:
            S.last_image_jpeg = bytes(msg.data)
            S.last_image_time = time.time()


# --------- ODrive ---------
def connect_odrive():
    try:
        import odrive
        import odrive.enums as enums
    except ImportError as e:
        S.odrive_err = f"odrive lib import failed: {e}"
        log.warning(S.odrive_err)
        return

    log.info("Searching for ODrive (timeout %ds)...", ODRIVE_CONNECT_TIMEOUT)
    try:
        odrv = odrive.find_any(timeout=ODRIVE_CONNECT_TIMEOUT)
    except Exception as e:
        S.odrive_err = f"ODrive not found: {e}"
        log.warning("%s — chassis control will be disabled, viz still works.", S.odrive_err)
        return

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
    S.odrv, S.axis_l, S.axis_r = odrv, axis_l, axis_r
    S.odrive_err = ""
    log.info("ODrive connected: left=axis%d right=axis%d", AXIS_LEFT, AXIS_RIGHT)


def disconnect_odrive():
    if S.axis_l is None:
        return
    try:
        import odrive.enums as enums
        S.axis_l.controller.input_vel = 0
        S.axis_r.controller.input_vel = 0
        S.axis_l.requested_state = enums.AXIS_STATE_IDLE
        S.axis_r.requested_state = enums.AXIS_STATE_IDLE
    except Exception as e:
        log.warning("ODrive disconnect quirk: %s", e)
    S.odrv = S.axis_l = S.axis_r = None


def send_velocity(x_vel: float, theta_vel: float):
    if S.axis_l is None:
        return
    th = math.radians(theta_vel)
    half = WHEEL_BASE / 2.0
    L = x_vel - th * half
    R = x_vel + th * half
    if INVERT_LEFT:
        L = -L
    if INVERT_RIGHT:
        R = -R
    circ = 2.0 * math.pi * WHEEL_RADIUS
    try:
        S.axis_l.controller.input_vel = L / circ
        S.axis_r.controller.input_vel = R / circ
    except Exception as e:
        log.warning("ODrive write failed: %s", e)


def _safe_get(obj, *path, default=None):
    """Walk an attribute path; return default if any step is missing or errors.
    Lets us read optional ODrive fw fields without per-call try/except clutter."""
    try:
        cur = obj
        for k in path:
            cur = getattr(cur, k)
        return cur
    except Exception:
        return default


def _zero_axis():
    return {
        "iq_A": 0.0, "id_A": None,
        "torque_Nm": 0.0, "vel_rps": 0.0, "vel_mps": 0.0,
        "axis_err": "0x0", "motor_err": "0x0",
        "encoder_err": "0x0", "controller_err": "0x0",
    }


def _axis_data(ax) -> dict:
    iq  = _safe_get(ax, "motor", "current_control", "Iq_measured", default=0.0)
    id_ = _safe_get(ax, "motor", "current_control", "Id_measured")
    vel = _safe_get(ax, "encoder", "vel_estimate", default=0.0)
    return {
        "iq_A":           round(float(iq), 3),
        "id_A":           None if id_ is None else round(float(id_), 3),
        "torque_Nm":      round(iq * TORQUE_CONSTANT, 4),
        "vel_rps":        round(vel, 3),
        "vel_mps":        round(vel * 2 * math.pi * WHEEL_RADIUS, 3),
        "axis_err":       hex(_safe_get(ax, "error", default=0)),
        "motor_err":      hex(_safe_get(ax, "motor", "error", default=0)),
        "encoder_err":    hex(_safe_get(ax, "encoder", "error", default=0)),
        "controller_err": hex(_safe_get(ax, "controller", "error", default=0)),
    }


def read_telemetry() -> dict:
    if S.axis_l is None:
        return {
            "vbus": 0.0, "ibus": None, "dc_power": None,
            "batt_pct": 0, "brake_armed": None, "brake_sat": None,
            "left": _zero_axis(), "right": _zero_axis(),
            "peak_ibus": 0.0,
            "vio_status": S.last_status[:80],
            "settings": {"linear_speed": S.linear_speed, "angular_speed": S.angular_speed},
            "odrive_err": S.odrive_err or "(no odrive)",
        }
    try:
        vbus     = S.odrv.vbus_voltage
        ibus     = _safe_get(S.odrv, "ibus")
        dc_power = (vbus * ibus) if ibus is not None else None
        pct = max(0.0, min(100.0, (vbus - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100.0))
        pct = round(pct / 5.0) * 5

        # Track peak |ibus| (absolute, so regen counts too).
        if ibus is not None:
            with S.lock:
                S.peak_ibus = max(S.peak_ibus, abs(float(ibus)))
        peak_ibus = S.peak_ibus

        return {
            "vbus":         round(vbus, 2),
            "ibus":         None if ibus is None else round(float(ibus), 3),
            "dc_power":     None if dc_power is None else round(dc_power, 2),
            "batt_pct":     int(pct),
            "brake_armed":  _safe_get(S.odrv, "brake_resistor_armed"),
            "brake_sat":    _safe_get(S.odrv, "brake_resistor_saturated"),
            "left":  _axis_data(S.axis_l),
            "right": _axis_data(S.axis_r),
            "peak_ibus":    round(peak_ibus, 3),
            "vio_status":   S.last_status[:80],
            "settings": {"linear_speed": S.linear_speed, "angular_speed": S.angular_speed},
            "odrive_err": "",
        }
    except Exception as e:
        return {"error": str(e)}


# --------- viser scene ---------
def build_viser():
    vs = viser.ViserServer(host="0.0.0.0", port=VISER_PORT)
    vs.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.005)
    vs.scene.add_grid(
        "/floor",
        width=GRID_SIZE_M,
        height=GRID_SIZE_M,
        plane="xy",
        cell_size=GRID_CELL_M,
        cell_color=(60, 60, 75),
        cell_thickness=1.0,
        section_size=GRID_SECTION_M,
        section_color=(110, 110, 140),
        section_thickness=2.0,
        position=(0.0, 0.0, GRID_Z),
    )
    frustum = vs.scene.add_camera_frustum(
        "/cam",
        fov=FRUSTUM_FOV_RAD,
        aspect=FRUSTUM_ASPECT,
        scale=FRUSTUM_SCALE,
        color=(255, 180, 0),
    )
    return vs, frustum


def viser_updater(vs: viser.ViserServer, frustum):
    """Pose: pushed every iteration.
    Trail: sample S.last_pose at TRAIL_SAMPLE_HZ, append to deque (5 min),
    render as a point cloud. PointCloudHandle has no in-place setter so we
    remove+add each redraw — but for points (vs lines) the flicker is
    visually unnoticeable."""
    pose_interval = 0.02
    sample_interval = 1.0 / TRAIL_SAMPLE_HZ
    trail_pts: deque = deque(maxlen=TRAIL_MAX_POINTS)
    last_sample = 0.0
    last_log = 0.0
    path_handle = None
    point_color = np.array([255, 120, 0], dtype=np.uint8)

    while True:
        now = time.time()
        with S.lock:
            pose = S.last_pose

        if pose is not None:
            xyz, wxyz = pose
            frustum.position = xyz
            frustum.wxyz = wxyz

            if now - last_sample >= sample_interval:
                last_sample = now
                trail_pts.append(xyz)
                if len(trail_pts) >= 1:
                    pts = np.asarray(trail_pts, dtype=np.float32)
                    colors = np.tile(point_color, (len(pts), 1))
                    try:
                        if path_handle is not None:
                            path_handle.remove()
                        path_handle = vs.scene.add_point_cloud(
                            "/trail",
                            points=pts,
                            colors=colors,
                            point_size=0.025,
                            point_shape="circle",
                        )
                    except Exception as e:
                        log.warning("trail update failed: %s (pts.shape=%s)", e, pts.shape)
                        path_handle = None

        if now - last_log >= 5.0:
            log.info("trail: %d pts (cap %d)", len(trail_pts), TRAIL_MAX_POINTS)
            last_log = now

        time.sleep(pose_interval)


# --------- async loops ---------
async def control_loop():
    interval = 1.0 / COMMAND_HZ
    while True:
        now = time.monotonic()
        with S.lock:
            keys_now = set(S.keys)
            last_t = S.last_key_time
            lin = S.linear_speed
            ang = S.angular_speed
        if keys_now and (now - last_t) < IDLE_TIMEOUT:
            x = 0.0
            th = 0.0
            # ↑ = backward, ↓ = forward (per user preference)
            if "up" in keys_now:
                x -= lin
            if "down" in keys_now:
                x += lin
            if "left" in keys_now:
                th += ang
            if "right" in keys_now:
                th -= ang
            send_velocity(x, th)
        else:
            send_velocity(0.0, 0.0)
        await asyncio.sleep(interval)


async def telemetry_loop():
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if telemetry_clients:
            data = read_telemetry()
            msg = json.dumps({"type": "telemetry", **data})
            dead = []
            for ws in telemetry_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in telemetry_clients:
                    telemetry_clients.remove(ws)
        await asyncio.sleep(interval)


# --------- FastAPI ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, connect_odrive)
    t1 = asyncio.create_task(control_loop())
    t2 = asyncio.create_task(telemetry_loop())
    yield
    t1.cancel()
    t2.cancel()
    await loop.run_in_executor(None, disconnect_odrive)


app = FastAPI(lifespan=lifespan)


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insight 9 + ODrive Chassis Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee;
    user-select: none;
    display: flex; flex-direction: column;
  }
  header {
    padding: 0.6rem 1rem; background: #0f0f1f;
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
    border-bottom: 1px solid #222;
  }
  header h1 { font-size: 1.05rem; color: #e94560; }
  .pill { font-size: 0.78rem; padding: 0.2rem 0.6rem; border-radius: 999px; background: #222; color: #888; }
  .pill.connected { background: #0c5; color: #fff; }
  .pill.warn { background: #d93; color: #fff; }

  main {
    flex: 1; display: flex; min-height: 0;
  }
  #viser-frame {
    flex: 1; min-width: 0; min-height: 0;
    border: none; background: #000;
  }
  aside {
    width: 360px; padding: 1rem; overflow-y: auto;
    background: #16213e; border-left: 1px solid #222;
    display: flex; flex-direction: column; gap: 1rem;
  }
  .card { background: #1f2b4a; border-radius: 10px; padding: 0.9rem; }
  .card h2 { font-size: 0.9rem; color: #e94560; margin-bottom: 0.6rem; }

  /* keys */
  .keys { display: grid; grid-template-areas: ". up ." "left down right"; gap: 8px; justify-content: center; }
  .key {
    width: 64px; height: 64px;
    border: 2px solid #444; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; background: #0f1730; cursor: pointer;
    transition: all 0.08s;
  }
  .key.active { background: #e94560; border-color: #e94560; transform: scale(1.08); }
  .key[data-dir="up"]    { grid-area: up; }
  .key[data-dir="left"]  { grid-area: left; }
  .key[data-dir="down"]  { grid-area: down; }
  .key[data-dir="right"] { grid-area: right; }
  .vel { margin-top: 0.6rem; font-family: monospace; font-size: 0.82rem; color: #aaa; text-align: center; }
  .hint { margin-top: 0.3rem; font-size: 0.72rem; color: #555; text-align: center; }

  /* sliders */
  .slider-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; font-size: 0.82rem; }
  .slider-row label { width: 70px; text-align: right; color: #aaa; }
  .slider-row input[type=range] { flex: 1; accent-color: #e94560; }
  .slider-row .val { width: 70px; font-family: monospace; color: #e94560; font-size: 0.78rem; }

  /* telemetry */
  .telemetry { font-family: monospace; font-size: 0.78rem; line-height: 1.6; }
  .trow { display: flex; justify-content: space-between; }
  .trow .label { color: #888; }
  .trow .value { color: #eee; }
  .trow .value.err { color: #f55; }
  .tsep { border-top: 1px solid #2a3858; margin: 0.4rem 0; }
  .section-title { color: #fa8; font-family: sans-serif; font-size: 0.78rem;
                   margin: 0.4rem 0 0.3rem 0; font-weight: bold; }
  .badge { display: inline-block; padding: 0.05rem 0.45rem; border-radius: 4px;
           font-size: 0.7rem; font-family: monospace; margin-left: 0.3rem; }
  .badge.drive { background: #2a4a30; color: #6f6; }
  .badge.regen { background: #4a3030; color: #f88; }
  .badge.warn  { background: #503510; color: #fa0; }
  .badge.ok    { background: #2a4a30; color: #6f6; }
  .badge.muted { background: #333;    color: #888; }
  .errgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.05rem 0.5rem; }
  .peak { color: #888; font-size: 0.72rem; }
  .vio-line { font-family: monospace; font-size: 0.75rem; color: #9cf; word-break: break-all; }
  .rgb-stream { width: 100%; max-height: 540px; object-fit: contain;
                border-radius: 6px; background: #000; display: block; }
</style>
</head>
<body>
<header>
  <h1>Insight 9 + ODrive Chassis</h1>
  <span class="pill" id="status">Connecting…</span>
  <span class="pill" id="latency">— ms</span>
  <span class="pill" id="odrive-pill">odrive: ?</span>
</header>

<main>
  <iframe id="viser-frame" src=""></iframe>
  <aside>
    <div class="card">
      <h2>Insight 9 RGB</h2>
      <img class="rgb-stream" src="/rgb_stream" alt="Insight 9 color stream">
    </div>

    <div class="card">
      <h2>Drive (Arrow keys / WASD)</h2>
      <div class="keys">
        <div class="key" data-dir="up">&uarr;</div>
        <div class="key" data-dir="left">&larr;</div>
        <div class="key" data-dir="down">&darr;</div>
        <div class="key" data-dir="right">&rarr;</div>
      </div>
      <div class="vel" id="vel">x: 0.00 m/s &nbsp; &theta;: 0 &deg;/s</div>
      <div class="hint">click viser pane first if keyboard does nothing — focus issue</div>
    </div>

    <div class="card">
      <h2>Speed</h2>
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

    <div class="card">
      <h2>Power</h2>
      <div class="telemetry" id="power_body">Waiting for data…</div>
    </div>

    <div class="card">
      <h2>Motors</h2>
      <div class="telemetry" id="motors_body">Waiting for data…</div>
    </div>

    <div class="card">
      <h2>VIO status</h2>
      <div class="vio-line" id="vio_line">—</div>
    </div>
  </aside>
</main>

<script>
const VISER_PORT = __VISER_PORT__;
document.getElementById('viser-frame').src = `${location.protocol}//${location.hostname}:${VISER_PORT}/`;

const KEY_MAP = {
  ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right",
  w:"up", W:"up", s:"down", S:"down", a:"left", A:"left", d:"right", D:"right",
};
const pressed = new Set();
let ws = null;
let linSpeed = 0.15, angSpeed = 30;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    setPill("status", "Connected", "connected");
    setInterval(() => { if (ws.readyState===1) ws.send(JSON.stringify({type:"ping",t:performance.now()})); }, 1000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "telemetry") renderTelemetry(msg);
    else if (msg.type === "pong") {
      const rtt = performance.now() - msg.t;
      document.getElementById("latency").textContent = `${rtt.toFixed(0)} ms`;
    }
  };
  ws.onclose = () => {
    setPill("status", "Disconnected — retry", "");
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}
function setPill(id, txt, cls) {
  const el = document.getElementById(id);
  el.textContent = txt;
  el.className = "pill " + (cls || "");
}
connect();

function sendMsg(o) { if (ws && ws.readyState===1) ws.send(JSON.stringify(o)); }
function sendKeys() {
  sendMsg({type:"keys", keys:[...pressed]});
  document.querySelectorAll(".key").forEach(el => {
    el.classList.toggle("active", pressed.has(el.dataset.dir));
  });
  let x=0, th=0;
  // ↑ = backward, ↓ = forward (matches server-side control_loop)
  if (pressed.has("up")) x -= linSpeed;
  if (pressed.has("down")) x += linSpeed;
  if (pressed.has("left")) th += angSpeed;
  if (pressed.has("right")) th -= angSpeed;
  document.getElementById("vel").textContent =
    `x: ${x.toFixed(2)} m/s   θ: ${th.toFixed(0)} °/s`;
}

let heartbeatId = null;
function startHB() { if (!heartbeatId) heartbeatId = setInterval(() => { if (pressed.size>0) sendKeys(); else stopHB(); }, 100); }
function stopHB()  { if (heartbeatId) { clearInterval(heartbeatId); heartbeatId = null; } }

document.addEventListener("keydown", e => {
  const dir = KEY_MAP[e.key]; if (!dir) return;
  e.preventDefault();
  if (!pressed.has(dir)) { pressed.add(dir); sendKeys(); startHB(); }
});
document.addEventListener("keyup", e => {
  const dir = KEY_MAP[e.key]; if (!dir) return;
  e.preventDefault();
  pressed.delete(dir); sendKeys();
  if (pressed.size===0) stopHB();
});
document.querySelectorAll(".key").forEach(el => {
  const dir = el.dataset.dir;
  el.addEventListener("touchstart", e => { e.preventDefault(); pressed.add(dir); sendKeys(); });
  el.addEventListener("touchend",   e => { e.preventDefault(); pressed.delete(dir); sendKeys(); });
  el.addEventListener("mousedown",  e => { pressed.add(dir); sendKeys(); });
  el.addEventListener("mouseup",    e => { pressed.delete(dir); sendKeys(); });
  el.addEventListener("mouseleave", e => { pressed.delete(dir); sendKeys(); });
});

const slLin = document.getElementById("sl_lin");
const slAng = document.getElementById("sl_ang");
slLin.addEventListener("input", () => {
  linSpeed = parseFloat(slLin.value);
  document.getElementById("sl_lin_v").textContent = linSpeed.toFixed(2) + " m/s";
  sendMsg({type:"speed", linear:linSpeed, angular:angSpeed});
});
slAng.addEventListener("input", () => {
  angSpeed = parseFloat(slAng.value);
  document.getElementById("sl_ang_v").innerHTML = angSpeed.toFixed(0) + " &deg;/s";
  sendMsg({type:"speed", linear:linSpeed, angular:angSpeed});
});

function renderTelemetry(d) {
  if (d.odrive_err) setPill("odrive-pill", "odrive: " + d.odrive_err, "warn");
  else setPill("odrive-pill", "odrive ok", "connected");

  if (d.error && !d.left) {
    document.getElementById("power_body").textContent = "Error: " + d.error;
    document.getElementById("motors_body").textContent = "";
    return;
  }
  if (d.settings && !slLin._touched) {
    if (Math.abs(linSpeed - d.settings.linear_speed) > 0.001) {
      linSpeed = d.settings.linear_speed;
      slLin.value = linSpeed;
      document.getElementById("sl_lin_v").textContent = linSpeed.toFixed(2) + " m/s";
    }
  }
  if (d.settings && !slAng._touched) {
    if (Math.abs(angSpeed - d.settings.angular_speed) > 0.1) {
      angSpeed = d.settings.angular_speed;
      slAng.value = angSpeed;
      document.getElementById("sl_ang_v").innerHTML = angSpeed.toFixed(0) + " &deg;/s";
    }
  }

  // ---------- Power card ----------
  const fmt = (v, dp, unit) => v == null ? "—" : (typeof v === 'number' ? v.toFixed(dp) : v) + (unit ? " " + unit : "");
  const battColor = d.batt_pct < 20 ? "#f55" : d.batt_pct < 50 ? "#fa0" : "#0f0";

  // DC current sign → drive vs regen
  let flowBadge = '<span class="badge muted">idle</span>';
  if (d.ibus != null) {
    if (d.ibus > 0.05)       flowBadge = '<span class="badge drive">DRIVE</span>';
    else if (d.ibus < -0.05) flowBadge = '<span class="badge regen">REGEN</span>';
  }

  // Brake resistor badge
  let brakeBadge;
  if (d.brake_armed === null || d.brake_armed === undefined) {
    brakeBadge = '<span class="badge muted">n/a</span>';
  } else if (d.brake_sat) {
    brakeBadge = '<span class="badge warn">SATURATED</span>';
  } else if (d.brake_armed) {
    brakeBadge = '<span class="badge ok">armed</span>';
  } else {
    brakeBadge = '<span class="badge muted">disarmed</span>';
  }

  document.getElementById("power_body").innerHTML = `
    <div class="trow"><span class="label">Battery</span><span class="value" style="color:${battColor}">${d.batt_pct}%</span></div>
    <div class="trow"><span class="label">DC bus V</span><span class="value">${fmt(d.vbus, 2, "V")}</span></div>
    <div class="trow"><span class="label">DC bus I</span><span class="value">${fmt(d.ibus, 3, "A")} <span class="peak">(peak ${fmt(d.peak_ibus, 3, "A")})</span> ${flowBadge}</span></div>
    <div class="trow"><span class="label">DC power</span><span class="value">${fmt(d.dc_power, 2, "W")}</span></div>
    <div class="trow"><span class="label">Brake R</span><span class="value">${brakeBadge}</span></div>
  `;

  // ---------- Motors card ----------
  const L = d.left || {}, R = d.right || {};
  const errCl = v => v === "0x0" ? "value" : "value err";

  const renderMotor = (m, label) => `
    <div class="section-title">${label}</div>
    <div class="trow"><span class="label">Iq / Id</span><span class="value">${fmt(m.iq_A, 3, "A")} / ${fmt(m.id_A, 3, "A")}</span></div>
    <div class="trow"><span class="label">Torque</span><span class="value">${fmt(m.torque_Nm, 4, "Nm")}</span></div>
    <div class="trow"><span class="label">Velocity</span><span class="value">${fmt(m.vel_mps, 3, "m/s")} (${fmt(m.vel_rps, 2, "rps")})</span></div>
    <div class="trow"><span class="label">Errors</span><span class="value"></span></div>
    <div class="errgrid">
      <div><span class="label">axis </span><span class="${errCl(m.axis_err)}">${m.axis_err}</span></div>
      <div><span class="label">motor</span><span class="${errCl(m.motor_err)}">${m.motor_err}</span></div>
      <div><span class="label">enc  </span><span class="${errCl(m.encoder_err)}">${m.encoder_err}</span></div>
      <div><span class="label">ctrl </span><span class="${errCl(m.controller_err)}">${m.controller_err}</span></div>
    </div>
  `;
  document.getElementById("motors_body").innerHTML =
    renderMotor(L, "Left") + '<div class="tsep"></div>' + renderMotor(R, "Right");

  document.getElementById("vio_line").textContent = d.vio_status || "—";
}
slLin.addEventListener("change", ()=>slLin._touched=true);
slAng.addEventListener("change", ()=>slAng._touched=true);
</script>
</body>
</html>
"""


def _mjpeg_generator():
    """Stream the raw JPEG bytes from /camera/.../color/.../compressed."""
    interval = 1.0 / RGB_STREAM_FPS
    last_sent_time = 0.0
    while True:
        with S.lock:
            jpeg = S.last_image_jpeg
            jpeg_time = S.last_image_time
        if jpeg and jpeg_time > last_sent_time:
            last_sent_time = jpeg_time
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(interval)


@app.get("/rgb_stream")
async def rgb_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE.replace("__VISER_PORT__", str(VISER_PORT))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    telemetry_clients.append(ws)
    log.info("client connected (n=%d)", len(telemetry_clients))
    try:
        while True:
            data = json.loads(await ws.receive_text())
            t = data.get("type", "")
            if t == "ping":
                await ws.send_text(json.dumps({"type": "pong", "t": data.get("t", 0)}))
            elif t == "keys":
                keys = set(data.get("keys", []))
                with S.lock:
                    S.keys = keys
                    S.last_key_time = time.monotonic()
            elif t == "speed":
                with S.lock:
                    S.linear_speed = max(0.01, min(2.0, float(data.get("linear", S.linear_speed))))
                    S.angular_speed = max(1.0, min(360.0, float(data.get("angular", S.angular_speed))))
    except WebSocketDisconnect:
        with S.lock:
            S.keys.clear()
        if ws in telemetry_clients:
            telemetry_clients.remove(ws)
        log.info("client disconnected (n=%d)", len(telemetry_clients))


# --------- main ---------
def main():
    rclpy.init()
    node = InsightSubs()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    vs, frustum = build_viser()
    threading.Thread(target=viser_updater, args=(vs, frustum), daemon=True).start()

    log.info("viser  : http://0.0.0.0:%d", VISER_PORT)
    log.info("control: http://0.0.0.0:%d", HTTP_PORT)
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")


if __name__ == "__main__":
    main()
