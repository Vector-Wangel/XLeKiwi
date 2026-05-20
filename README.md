# XLeKiwi

Control, perception and visualization stack for the **XLeKiwi** mobile manipulation platform. Runs on Jetson Nano (Ubuntu 22.04 + ROS 2 Humble + Python 3.10) or Jetson Thor (Ubuntu 24.04 + ROS 2 Jazzy + Python 3.12). x86_64 Linux works for most paths except hardware-specific bits.

The repo has two top-level Python components:

1. [scripts/](scripts/) — standalone demos (ODrive base web control, the integrated Insight 9 + Panthera + base demo)
2. [panthera_python/](panthera_python/) — Python SDK for the 6-DoF Panthera arm, with URDF/yaml configs for both the **Panthera-HT** (heavy-torque) and **XLeRobot-HT** (lightweight) arm variants

---

## 1. Install

### 1.1 System packages

```bash
sudo bash install/install_system_deps.sh
```

Auto-detects the Ubuntu codename and installs the matching ROS 2 distro:

| Ubuntu | ROS 2 | Typical hardware |
|---|---|---|
| 22.04 jammy | Humble | Jetson Nano (JetPack 6.x) |
| 24.04 noble | Jazzy | Jetson Thor (JetPack 7.x) |

Installs apt prerequisites, ROS 2 (`ros-base` + `cv-bridge` + `rmw-fastrtps` + `rosbag2-mcap`), `libserialport-dev`, `libyaml-cpp-dev`, `libeigen3-dev`. Prints `[ok] ros2 CLI works (<distro>)` on success.

### 1.2 Python environment

```bash
bash install/install_python_deps.sh
```

Creates `.venv/` in the repo root with `--system-site-packages` (so ROS 2's `rclpy` is importable), installs the Panthera motor wheel matching the local Python version and architecture, and installs demo dependencies (`fastapi`, `uvicorn`, `viser`, `odrive`).

Activate it:

```bash
source .venv/bin/activate
```

### 1.3 Runtime environment (Insight 9 / ROS)

```bash
bash install/setup_runtime_env.sh
```

Writes the Fast DDS profile to `~/fastdds_usb.xml` and appends an XLeKiwi block to `~/.bashrc` that sources the appropriate ROS distro (prefers Jazzy over Humble). After running it, **edit `~/fastdds_usb.xml`** and replace `169.254.10.2` with the host-side IP you see for the Insight 9 USB-CDC link:

```bash
ip -br addr | grep 169.254
```

If the host has multiple NICs (e.g. WiFi + Ethernet + USB-CDC), also run:

```bash
sudo bash install/add_dds_route.sh
```

This forces DDS multicast (239.0.0.0/8) onto the USB-CDC interface so discovery doesn't escape to WiFi or other networks.

---

## 2. Hardware

Skip subsections for hardware you don't have.

### 2.1 Insight 9 (VIO camera)

Plug the USB-C cable into any USB3 port, wait ~20 seconds for the camera to come up, then:

```bash
ip -br addr | grep 169.254       # find the virtual NIC, note this host's IPv4
ping -c 3 169.254.10.1            # should respond
```

After section 1.3 is done, verify topics:

```bash
source ~/.bashrc
ros2 daemon stop && ros2 daemon start && sleep 5
ros2 topic list                   # should include /insight/vio_100hz, /camera/camera/imu
ros2 topic hz /insight/vio_100hz  # ~100 Hz
ros2 topic hz /camera/camera/imu  # ~400 Hz
```

Firmware < v1.2.4 needs an OTA upgrade first — open `http://169.254.10.1` in a browser.

### 2.2 ODrive (biwheel base)

```bash
lsusb | grep -i odrive
# udev rule so non-root users can talk to it
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="0d32", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/99-odrive.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

source .venv/bin/activate
python -c "import odrive; o=odrive.find_any(timeout=10); print('vbus=',o.vbus_voltage)"
```

Motor calibration / pole-pair / encoder config lives in the ODrive's own flash — it travels with the controller, no host-side setup needed.

### 2.3 Panthera arm

Wiring: Follower → CAN1 (`/dev/ttyACM0`), Leader → CAN2 (`/dev/ttyACM1`).

```bash
sudo chmod 777 /dev/ttyACM*       # temporary; prefer a udev rule for permanent

source .venv/bin/activate
cd panthera_python/scripts
python 0_robot_get_state.py       # expect: 7 motors online, error=0x0
python 0_robot_set_zero.py        # first run / after switching arm variant
```

The repo supports two arm variants via yaml symlinks in [panthera_python/robot_param/](panthera_python/robot_param/):

```bash
cd panthera_python/robot_param

# Switch to the lightweight XLeRobot-HT arm (motors 4438_30 / 3536_32)
ln -sf Follower_xlerobot_ht.yaml Follower.yaml
ln -sf Leader_xlerobot_ht.yaml   Leader.yaml

# Switch back to the high-torque Panthera-HT arm (default; motors 5047_36 / 6056_36 / 4438_30)
ln -sf Follower_panthera.yaml Follower.yaml
ln -sf Leader_panthera.yaml   Leader.yaml

readlink Follower.yaml            # confirm what's active
```

After every hardware swap, re-run `0_robot_get_state.py` and `0_robot_set_zero.py` — the two arms have different zero positions.

### 2.4 RealSense D405 (UVC RGB, optional)

The ODrive web demo reads a D405 RGB stream from `/dev/video4` if present. UVC is in-kernel, no extra setup needed beyond plugging the camera in.

---

## 3. Demos ([scripts/](scripts/))

Everything under `scripts/` is a standalone entry point — they don't depend on each other. Open a fresh terminal (so `~/.bashrc` sources ROS + Fast DDS), then:

```bash
source .venv/bin/activate
python scripts/<N>_<name>.py
```

| # | Demo | Port(s) |
|---|---|---|
| 0 | [0_integrated_demo.py](scripts/0_integrated_demo.py) — **Insight 9 VIO + Panthera arm + ODrive base + viser, one web page** | 8080 + 8084 |
| 1 | [1_odrive_web_control.py](scripts/1_odrive_web_control.py) — ODrive biwheel base web control | 8080 |
| 2 | [2_insight_chassis_demo.py](scripts/2_insight_chassis_demo.py) — Insight 9 + ODrive combined demo (chassis + VIO + RGB) | 8080 + 8082 |

### 3.1 0_integrated_demo.py (recommended)

The most complete demo. Single web page integrating:
- Panthera arm Cartesian impedance control (polar mode, 2 kHz control loop)
- ODrive biwheel base velocity control
- Insight 9 VIO frustum + trajectory trail (viser 3D)
- Insight 9 RGB compressed → MJPEG endpoint
- Telemetry panel

```bash
python scripts/0_integrated_demo.py
# Open http://<host>:8080 in your browser.
```

Key bindings:

| Action | Keys |
|---|---|
| Arm radial in/out | `W` / `S` |
| Arm orbit around base | `A` / `D` |
| Arm vertical up/down | `Q` / `E` |
| Arm pitch | `I` / `K` |
| Arm yaw | `J` / `L` |
| Arm roll | `U` / `O` |
| Gripper close/open | `Z` / `X` |
| Print end-effector pose | `Space` |
| Zero external wrench | `M` |
| Return to home | `R` |
| Base drive | Arrow keys |

Required hardware: Insight 9, ODrive base, and the Panthera Follower (on CAN1) must all be connected.

### 3.2 1_odrive_web_control.py — ODrive web control

Drive the biwheel base with WASD / arrow keys from a browser. Telemetry shows current / torque / speed in real time, plus a video feed from `/dev/video4` (D405 UVC RGB) if connected.

Hardware mapping (top of file, editable): `WHEEL_RADIUS=0.05`, `WHEEL_BASE=0.25`, `AXIS_LEFT=1`, `AXIS_RIGHT=0`, `INVERT_LEFT=True`. Default 0.15 m/s linear, 30 °/s angular (sliders in the UI). 5S Li-ion battery voltage estimation (`BATT_FULL_V=21.0V`).

---

## 4. Panthera SDK ([panthera_python/](panthera_python/))

6-DoF arm (with gripper). Lower layer is the `hightorque_robot` wheel (CAN / serial) plus Pinocchio dynamics. Detailed API reference in [panthera_python/README.md](panthera_python/README.md).

### 4.1 Scripts ([panthera_python/scripts/](panthera_python/scripts/))

All scripts must be run from `panthera_python/scripts/` (`Panthera_lib` is imported from source, not via pip).

| # | Script | Purpose |
|---|---|---|
| 0 | [0_robot_get_state.py](panthera_python/scripts/0_robot_get_state.py) | Print each joint's position / velocity / torque — **always run first** |
| 0 | [0_robot_set_zero.py](panthera_python/scripts/0_robot_set_zero.py) | Set the zero position |
| 1 | `1_Joint_PosVel_control.py` etc. | Joint-level control, FK/IK tests |
| 2 | `2_*_compensation_control.py` / `2_Jointimpendence_*` | Gravity / friction compensation, joint impedance |
| 3 | `3_interpolation_control_*` / `3_sin_trajectory_control.py` | Quintic / septic interpolation, sinusoidal trajectories |
| 4 | [4_impedance_trajectory_control_with_gra_pd.py](panthera_python/scripts/4_impedance_trajectory_control_with_gra_pd.py) | Trajectory + impedance control |
| 5 | [5_teleop_control.py](panthera_python/scripts/5_teleop_control.py) | **Leader/follower teleop** (CAN1=Follower, CAN2=Leader) |
| 5 | [5_record_trajectory.py](panthera_python/scripts/5_record_trajectory.py) | Record trajectories from teleop → jsonl |
| 5 | [5_replay_trajectory.py](panthera_python/scripts/5_replay_trajectory.py) | Replay a recorded trajectory |
| 6 | `6_moveL_pos_control.py` / `6_moveL_rotate_control.py` | Cartesian straight-line / rotation |
| 7 | `7_keyboard_ee_control_*` | Keyboard end-effector control (Cartesian / joint impedance, terminal + web variants) |
| 8 | [8_arm_base_web_control.py](panthera_python/scripts/8_arm_base_web_control.py) | Arm + ODrive base joint web control, port 8083 |

### 4.2 Panthera_lib

[panthera_python/scripts/Panthera_lib/Panthera.py](panthera_python/scripts/Panthera_lib/Panthera.py) inherits from `htr.Robot` and wraps position / velocity / MIT control, FK/IK, dynamics (M/C/G + friction), quintic/septic interpolation, recording/replay. Three additions on top of the upstream SDK:

- `end_effector_frame_id` reads `urdf.end_effector_link` (typically `tool_link`) from the yaml; FK uses `pin.oMf[end_effector_frame_id]`. **The 0.165m X offset is no longer hard-coded** — it's encoded in the URDF's `tool_joint`. Change the tool tip by editing `tool_joint origin xyz`.
- Joint-limit velocity safety guard: when a joint approaches its limit, velocity into the limit is zeroed automatically (motion away from the limit is unaffected).
- `get_jacobian()` returns the 6×6 Jacobian in the `pin.LOCAL_WORLD_ALIGNED` frame.

---

## 5. Port reference

| Port | Service | Launcher |
|---|---|---|
| 8080 | ODrive web / Insight chassis demo / **integrated_demo main page** | one of |
| 8082 | Panthera EE keyboard web / Insight chassis viser iframe | one of |
| 8083 | Panthera arm + ODrive base web | [8_arm_base_web_control.py](panthera_python/scripts/8_arm_base_web_control.py) |
| 8084 | **integrated_demo viser iframe** | [0_integrated_demo.py](scripts/0_integrated_demo.py) |

⚠️ Ports 8080 and 8082 are reused across services — **you can only run one of the conflicting demos at a time**. The recommended workflow is just `0_integrated_demo.py` (covers the most ground).

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ros2 topic list` is empty | 1) Verify `~/fastdds_usb.xml` `<address>` matches `ip -br addr` output; 2) on multi-NIC hosts run `sudo bash install/add_dds_route.sh`; 3) `ros2 daemon stop && ros2 daemon start` |
| `rclpy` import fails | `.venv` must be created with `--system-site-packages`; source ROS before activating venv |
| `cv_bridge` numpy ABI error `_ARRAY_API not found` | numpy 2.x in venv clashes with apt cv_bridge (built against numpy 1.x). Workaround: use `CompressedImage` topics + `cv2.imdecode` — see [scripts/2_insight_chassis_demo.py](scripts/2_insight_chassis_demo.py) |
| `ImportError: pinocchio` | `pip install pin` (note: **`pin`**, not `pinocchio` — different packages) |
| `libyaml-cpp.so.0.6 not found` | System ships 0.7/0.8; build 0.6.1 from source — see end of [panthera_python/README.md](panthera_python/README.md) |
| `/dev/ttyACM*` permission denied | `sudo chmod 777 /dev/ttyACM*` (temporary) or write a udev rule (permanent) |
| ODrive `find_any` times out | USB *data* cable issue (not a charging cable); try a different USB3 port |
| Insight 9 IMU < 400 Hz | USB bandwidth: use a USB3 cable into a USB3 port (`lsusb -t` to confirm); enable high power mode with `sudo nvpmodel -m <max>` |
| viser browser feels laggy | Lower `RGB_STREAM_FPS` to 5–10, or reduce point cloud size in the demo source |

---

## Repository layout

```
xlekiwi/
├── install/                Install scripts (system / Python / runtime / DDS route) + Fast DDS profile template
├── scripts/                Standalone demos (numbered, plain `python scripts/<N>_<name>.py`)
└── panthera_python/        Panthera / XLeRobot-HT arm SDK
    ├── scripts/            Numbered example/control scripts
    ├── robot_param/        Per-arm yaml configs (symlink picks the active variant)
    ├── xlerobot_ht_description/          XLeRobot-HT arm URDF
    └── xlerobot_ht_description_gripper/  XLeRobot-HT arm URDF with gripper
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
