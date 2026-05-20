#!/usr/bin/env python3
"""
ODrive Biwheel Web Controller

A lightweight web server that lets you drive the ODrive biwheel base
from any browser on the same network using keyboard (WASD / arrow keys).
Streams real-time ODrive telemetry (current, torque, velocity) and
allows adjusting speed parameters from the browser.

Usage:
    source <repo>/.venv/bin/activate
    python <repo>/scripts/1_odrive_web_control.py

Then open http://<jetson-ip>:8080 in your PC browser.
"""

import asyncio
import json
import logging
import math
import time
from contextlib import asynccontextmanager

import threading

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

# --------------- CONFIG ---------------
HOST = "0.0.0.0"
PORT = 8080
LINEAR_SPEED = 0.15        # m/s  (forward/backward) — adjustable from UI
ANGULAR_SPEED = 30.0       # deg/s (left/right turn) — adjustable from UI
COMMAND_HZ = 20            # how often we push velocity to ODrive
TELEMETRY_HZ = 10          # how often we push telemetry to browser
IDLE_TIMEOUT = 0.3         # seconds with no key -> stop

# ODrive hardware mapping
WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.25
INVERT_LEFT = True
INVERT_RIGHT = False
AXIS_LEFT = 1
AXIS_RIGHT = 0
ODRIVE_SERIAL = None       # None = auto-detect first ODrive
TORQUE_CONSTANT = 8.27 / 270  # Nm/A — adjust for your motor (8.27 / KV)

# Battery: 5S Li-ion (20V drill battery)
BATT_CELLS = 5
BATT_FULL_V = 4.2 * BATT_CELLS   # 21.0V
BATT_EMPTY_V = 3.0 * BATT_CELLS  # 15.0V

# Camera (RealSense D405 via UVC)
CAMERA_DEV = "/dev/video4"     # YUYV RGB stream
CAM_WIDTH = 424
CAM_HEIGHT = 240
CAM_FPS = 30
JPEG_QUALITY = 70
# --------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("odrive_web")

# Global mutable state
current_keys: set[str] = set()
last_key_time: float = 0.0
linear_speed: float = LINEAR_SPEED
angular_speed: float = ANGULAR_SPEED
odrv = None
axis_l = None
axis_r = None
telemetry_clients: list[WebSocket] = []


def connect_odrive():
    global odrv, axis_l, axis_r
    import odrive
    import odrive.enums as enums

    log.info("Searching for ODrive (timeout 30s)...")
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


# ---- Camera capture (background thread) ----
_latest_frame: bytes = b""
_frame_lock = threading.Lock()
_cam_running = False


def camera_thread():
    global _latest_frame, _cam_running
    cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    if not cap.isOpened():
        log.warning("Camera %s failed to open — video stream disabled", CAMERA_DEV)
        return
    log.info("Camera opened: %s (%dx%d @ %dfps)", CAMERA_DEV, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
    _cam_running = True
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while _cam_running:
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if ok:
            with _frame_lock:
                _latest_frame = buf.tobytes()
    cap.release()
    log.info("Camera thread stopped")


def stop_camera():
    global _cam_running
    _cam_running = False


def mjpeg_generator():
    """Yield MJPEG multipart frames."""
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(1.0 / CAM_FPS)


def send_velocity(x_vel: float, theta_vel: float):
    """Convert body velocity to wheel RPS and send to ODrive."""
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


def read_telemetry() -> dict:
    """Read ODrive telemetry — all reads are cheap USB polls."""
    try:
        l_cur = axis_l.motor.current_control.Iq_measured
        r_cur = axis_r.motor.current_control.Iq_measured
        l_vel = axis_l.encoder.vel_estimate  # turns/s
        r_vel = axis_r.encoder.vel_estimate
        l_torque = l_cur * TORQUE_CONSTANT
        r_torque = r_cur * TORQUE_CONSTANT
        vbus = odrv.vbus_voltage
        batt_raw = max(0.0, min(100.0, (vbus - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100.0))
        batt_pct = round(batt_raw / 5.0) * 5  # snap to 5% steps
        l_err = axis_l.error
        r_err = axis_r.error
        return {
            "vbus": round(vbus, 2),
            "batt_pct": int(batt_pct),
            "left": {
                "current_A": round(l_cur, 3),
                "torque_Nm": round(l_torque, 4),
                "vel_rps": round(l_vel, 3),
                "vel_mps": round(l_vel * 2 * math.pi * WHEEL_RADIUS, 3),
                "error": hex(l_err),
            },
            "right": {
                "current_A": round(r_cur, 3),
                "torque_Nm": round(r_torque, 4),
                "vel_rps": round(r_vel, 3),
                "vel_mps": round(r_vel * 2 * math.pi * WHEEL_RADIUS, 3),
                "error": hex(r_err),
            },
            "settings": {
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
            },
        }
    except Exception as e:
        return {"error": str(e)}


async def control_loop():
    """Background task: read current_keys and push velocity at fixed rate."""
    interval = 1.0 / COMMAND_HZ
    while True:
        now = time.monotonic()
        if current_keys and (now - last_key_time < IDLE_TIMEOUT):
            x = 0.0
            theta = 0.0
            if "up" in current_keys:
                x -= linear_speed
            if "down" in current_keys:
                x += linear_speed
            if "left" in current_keys:
                theta += angular_speed
            if "right" in current_keys:
                theta -= angular_speed
            send_velocity(x, theta)
        else:
            send_velocity(0.0, 0.0)
        await asyncio.sleep(interval)


async def telemetry_loop():
    """Background task: broadcast telemetry to all connected browsers."""
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if telemetry_clients and axis_l is not None:
            data = read_telemetry()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    await loop.run_in_executor(None, connect_odrive)
    t1 = asyncio.create_task(control_loop())
    t2 = asyncio.create_task(telemetry_loop())
    yield
    t1.cancel()
    t2.cancel()
    stop_camera()
    await loop.run_in_executor(None, disconnect_odrive)


app = FastAPI(lifespan=lifespan)


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ODrive Biwheel Control</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee;
    display: flex; flex-direction: column;
    align-items: center; padding: 2rem 1rem;
    min-height: 100vh; user-select: none;
  }
  h1 { font-size: 1.4rem; margin-bottom: 0.5rem; color: #e94560; }
  .status { font-size: 0.85rem; margin-bottom: 1rem; color: #888; }
  .status.connected { color: #0f0; }

  .main { display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap; justify-content: center; }

  /* Camera */
  .camera { text-align: center; }
  .camera h2 { font-size: 0.95rem; color: #e94560; margin-bottom: 0.5rem; }
  .camera img { border-radius: 8px; border: 2px solid #333; background: #000; width: 424px; height: 240px; }

  /* Keys panel */
  .control-panel { display: flex; flex-direction: column; align-items: center; }
  .keys {
    display: grid;
    grid-template-areas: ". up ." "left down right";
    gap: 10px;
  }
  .key {
    width: 72px; height: 72px;
    border: 2px solid #444; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; background: #16213e;
    transition: all 0.1s; cursor: pointer;
  }
  .key.active { background: #e94560; border-color: #e94560; transform: scale(1.08); }
  .key[data-dir="up"]    { grid-area: up; }
  .key[data-dir="left"]  { grid-area: left; }
  .key[data-dir="down"]  { grid-area: down; }
  .key[data-dir="right"] { grid-area: right; }
  .vel { margin-top: 1rem; font-family: monospace; font-size: 0.9rem; color: #aaa; text-align: center; }
  .hint { margin-top: 0.5rem; font-size: 0.75rem; color: #555; }

  /* Sliders */
  .sliders { margin-top: 1.2rem; width: 100%; }
  .slider-row {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.6rem; font-size: 0.85rem;
  }
  .slider-row label { width: 80px; text-align: right; color: #aaa; }
  .slider-row input[type=range] { flex: 1; accent-color: #e94560; }
  .slider-row .val { width: 65px; font-family: monospace; color: #e94560; }

  /* Telemetry panel */
  .telemetry {
    background: #16213e; border-radius: 12px; padding: 1rem 1.2rem;
    min-width: 280px; font-family: monospace; font-size: 0.82rem; line-height: 1.7;
  }
  .telemetry h2 { font-size: 0.95rem; color: #e94560; margin-bottom: 0.6rem; font-family: sans-serif; }
  .trow { display: flex; justify-content: space-between; }
  .trow .label { color: #888; }
  .trow .value { color: #eee; }
  .trow .value.err { color: #f55; }
  .tsep { border-top: 1px solid #333; margin: 0.4rem 0; }
</style>
</head>
<body>
<h1>ODrive Biwheel Control</h1>
<div class="status" id="status">Connecting...</div>
<div class="status" id="latency">Latency: -- ms</div>

<div class="camera">
  <h2>D405 RGB</h2>
  <img id="cam" src="/video" alt="camera stream">
</div>

<div class="main">
  <!-- Left: controls -->
  <div class="control-panel">
    <div class="keys">
      <div class="key" data-dir="up">&uarr;</div>
      <div class="key" data-dir="left">&larr;</div>
      <div class="key" data-dir="down">&darr;</div>
      <div class="key" data-dir="right">&rarr;</div>
    </div>
    <div class="vel" id="vel">x: 0.00 m/s &nbsp; &theta;: 0.00 &deg;/s</div>
    <div class="hint">Arrow keys or W/A/S/D</div>

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

  <!-- Right: telemetry -->
  <div class="telemetry" id="tele">
    <h2>Telemetry</h2>
    <div id="tele_body">Waiting for data...</div>
  </div>
</div>

<script>
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
    document.getElementById("status").textContent = "Connected";
    document.getElementById("status").className = "status connected";
    setInterval(() => { if (ws.readyState===1) ws.send(JSON.stringify({type:"ping",t:performance.now()})); }, 1000);
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
    document.getElementById("status").className = "status";
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}
connect();

function sendMsg(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

function sendKeys() {
  sendMsg({type: "keys", keys: [...pressed]});
  document.querySelectorAll(".key").forEach(el => {
    el.classList.toggle("active", pressed.has(el.dataset.dir));
  });
  let x = 0, th = 0;
  if (pressed.has("up"))    x  += linSpeed;
  if (pressed.has("down"))  x  -= linSpeed;
  if (pressed.has("left"))  th -= angSpeed;
  if (pressed.has("right")) th += angSpeed;
  document.getElementById("vel").textContent =
    `x: ${x.toFixed(2)} m/s   \\u03b8: ${th.toFixed(1)} \\u00b0/s`;
}

// Heartbeat: while any key is held, re-send every 100ms so server doesn't time out
let heartbeatId = null;
function startHeartbeat() {
  if (heartbeatId) return;
  heartbeatId = setInterval(() => {
    if (pressed.size > 0) sendKeys();
    else stopHeartbeat();
  }, 100);
}
function stopHeartbeat() {
  if (heartbeatId) { clearInterval(heartbeatId); heartbeatId = null; }
}

document.addEventListener("keydown", e => {
  const dir = KEY_MAP[e.key]; if (!dir) return;
  e.preventDefault();
  pressed.add(dir); sendKeys(); startHeartbeat();
});
document.addEventListener("keyup", e => {
  const dir = KEY_MAP[e.key]; if (!dir) return;
  e.preventDefault();
  pressed.delete(dir); sendKeys();
  if (pressed.size === 0) stopHeartbeat();
});
document.querySelectorAll(".key").forEach(el => {
  el.addEventListener("touchstart", e => { e.preventDefault(); pressed.add(el.dataset.dir); sendKeys(); });
  el.addEventListener("touchend",   e => { e.preventDefault(); pressed.delete(el.dataset.dir); sendKeys(); });
  el.addEventListener("mousedown",  e => { pressed.add(el.dataset.dir); sendKeys(); });
  el.addEventListener("mouseup",    e => { pressed.delete(el.dataset.dir); sendKeys(); });
  el.addEventListener("mouseleave", e => { pressed.delete(el.dataset.dir); sendKeys(); });
});

// Sliders
const slLin = document.getElementById("sl_lin");
const slAng = document.getElementById("sl_ang");
slLin.addEventListener("input", () => {
  linSpeed = parseFloat(slLin.value);
  document.getElementById("sl_lin_v").textContent = linSpeed.toFixed(2) + " m/s";
  sendMsg({type: "speed", linear: linSpeed, angular: angSpeed});
});
slAng.addEventListener("input", () => {
  angSpeed = parseFloat(slAng.value);
  document.getElementById("sl_ang_v").innerHTML = angSpeed.toFixed(0) + " &deg;/s";
  sendMsg({type: "speed", linear: linSpeed, angular: angSpeed});
});

function renderTelemetry(d) {
  if (d.error && !d.left) {
    document.getElementById("tele_body").textContent = "Error: " + d.error;
    return;
  }
  // Sync slider values from server on first load
  if (d.settings) {
    if (Math.abs(linSpeed - d.settings.linear_speed) > 0.001 && !slLin._userTouched) {
      linSpeed = d.settings.linear_speed;
      slLin.value = linSpeed;
      document.getElementById("sl_lin_v").textContent = linSpeed.toFixed(2) + " m/s";
    }
    if (Math.abs(angSpeed - d.settings.angular_speed) > 0.1 && !slAng._userTouched) {
      angSpeed = d.settings.angular_speed;
      slAng.value = angSpeed;
      document.getElementById("sl_ang_v").innerHTML = angSpeed.toFixed(0) + " &deg;/s";
    }
  }
  const L = d.left, R = d.right;
  const errClass = (v) => v === "0x0" ? "value" : "value err";
  document.getElementById("tele_body").innerHTML = `
    <div class="trow"><span class="label">Battery</span><span class="value" style="color:${d.batt_pct<20?'#f55':d.batt_pct<50?'#fa0':'#0f0'}">${d.batt_pct}% (${d.vbus}V)</span></div>
    <div class="tsep"></div>
    <div class="trow"><span class="label">L current</span><span class="value">${L.current_A} A</span></div>
    <div class="trow"><span class="label">L torque</span><span class="value">${L.torque_Nm} Nm</span></div>
    <div class="trow"><span class="label">L vel</span><span class="value">${L.vel_rps} rps / ${L.vel_mps} m/s</span></div>
    <div class="trow"><span class="label">L error</span><span class="${errClass(L.error)}">${L.error}</span></div>
    <div class="tsep"></div>
    <div class="trow"><span class="label">R current</span><span class="value">${R.current_A} A</span></div>
    <div class="trow"><span class="label">R torque</span><span class="value">${R.torque_Nm} Nm</span></div>
    <div class="trow"><span class="label">R vel</span><span class="value">${R.vel_rps} rps / ${R.vel_mps} m/s</span></div>
    <div class="trow"><span class="label">R error</span><span class="${errClass(R.error)}">${R.error}</span></div>
  `;
}
</script>
</body>
</html>"""


@app.get("/video")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global last_key_time, linear_speed, angular_speed
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

            elif msg_type == "speed":
                linear_speed = max(0.01, min(2.0, float(data.get("linear", linear_speed))))
                angular_speed = max(1.0, min(360.0, float(data.get("angular", angular_speed))))
                log.info("Speed updated: linear=%.2f m/s, angular=%.1f deg/s",
                         linear_speed, angular_speed)

    except WebSocketDisconnect:
        current_keys.clear()
        if ws in telemetry_clients:
            telemetry_clients.remove(ws)
        log.info("Browser disconnected (total: %d)", len(telemetry_clients))


if __name__ == "__main__":
    import subprocess
    try:
        ip = subprocess.check_output(
            "hostname -I", shell=True, text=True
        ).strip().split()[0]
        log.info("Open in browser: http://%s:%d", ip, PORT)
    except Exception:
        pass

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
