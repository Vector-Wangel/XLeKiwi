# Panthera arm Python SDK

Python control SDK for the Panthera 6-DoF arm. This fork also ships configs and URDFs for the lightweight **XLeRobot-HT** arm variant — both run through the same SDK; the active variant is selected by yaml symlink (see [Two arm variants](#two-arm-variants) below).

## Features

- **Control modes**
  - Position + velocity control (with per-joint torque limit)
  - 5-parameter MIT mode (position + velocity + torque + Kp + Kd)
  - Independent gripper control
- **Kinematics**
  - Forward kinematics (FK)
  - Inverse kinematics (IK)
- **Dynamics**
  - Gravity compensation
  - Coriolis compensation
  - Mass matrix
  - Friction compensation
  - Full inverse dynamics
- **Trajectory planning**
  - Quintic polynomial interpolation (continuous velocity, acceleration)
  - Septic polynomial interpolation (continuous velocity, acceleration, jerk)
  - Septic interpolation with velocity boundary conditions
- **Safety**
  - Per-joint position limits
  - Torque limits
  - Timeout detection
  - Position-reached detection
- **Leader / follower teleoperation** with trajectory record/replay
- **YAML-based configuration**

## Base coordinate frame

![base joint](images/base_joint.png)

## Installation

### Recommended: dedicated conda env

To avoid conflicts with system Python (e.g. ROS), use an isolated env:

```bash
conda create -n panthera python=3.10
conda activate panthera
```

> When using this SDK as part of the full XLeKiwi stack, the top-level `install/install_python_deps.sh` builds a `.venv` that is shared with the demo scripts — you don't need a separate conda env in that case.

### Option 1 — prebuilt wheel (Ubuntu 22, recommended)

Pick the wheel that matches your Python version from `motor_whl/`:

```bash
# Python 3.10
pip install motor_whl/hightorque_robot-1.2.0-cp310-cp310-linux_x86_64.whl
# (also: cp39, cp311, cp312)
```

For Jetson aarch64 use the `linux_aarch64` wheel instead. The top-level installer picks this automatically.

### Option 2 — build from source (Ubuntu 20 or unusual systems)

System dependencies:

```bash
sudo apt-get install -y cmake python3-dev python3-pip \
    liblcm-dev libyaml-cpp-dev libserialport-dev
pip install pybind11
```

Build the motor C++ project:

```bash
cd ../panthera_cpp/motor_cpp
mkdir -p build && cd build
cmake .. && make
```

Build the Python binding:

```bash
cd ../../panthera_python
mkdir -p build && cd build
cmake .. && make
```

A successful build prints `Build target _hightorque_robot`.

### Python high-level deps (both options)

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install "pyyaml>=6.0" "pin>=2.6.0" "scipy>=1.9.0"
```

**Important:**
- Install `pin` (Pinocchio's pip package). **Not** `pinocchio` — that's an unrelated test framework on PyPI.
- PyYAML must be ≥ 6.0 for Python 3.12+ support.
- Inside a conda env, use `pip` (not `pip3`).

### Verify

```bash
python -c "import hightorque_robot; print('motor SDK OK')"
python -c "import pinocchio as pin; print('pin OK')"
python -c "import yaml; print('pyyaml OK')"
```

Then from `scripts/`:

```python
# Panthera_lib is provided as source — import it from the scripts directory
cd scripts
from Panthera_lib import Panthera
```

## Quick start

### Wiring reference

![board wiring](images/Board.png)

### Serial-port permissions

When all motors are connected you should see seven `/dev/ttyACM*` devices:

```bash
ls /dev/ttyACM*
sudo chmod -R 777 /dev/ttyACM*
```

For a permanent rule, see [Auto-grant serial permissions](#auto-grant-serial-permissions).

### Test sequence

a. **Always run first.** From `scripts/`, check all joints:

```bash
python 0_robot_get_state.py
```

You should see the port, motor IDs, init log, then a continuous print of each joint's position / velocity / torque.

b. Position + velocity control:

```bash
python 1_Joint_PosVel_control.py
```

The arm cycles through preset positions. If it completes the sequence, the SDK is working.

### Record and replay a trajectory

`5_record_trajectory.py` runs leader/follower teleoperation and records the trajectory. Press Ctrl+C to stop; the trajectory is saved as `trajectory_YYYYMMDD_HHMMSS.jsonl` in the current directory.

```bash
python 5_record_trajectory.py
```

Edit `TRAJECTORY_FILE` at the top of `5_replay_trajectory.py` to point at that file, then:

```bash
python 5_replay_trajectory.py
```

The leader arm replays the recorded trajectory. To replay on the follower instead, change the `config_path` inside the script from `Leader.yaml` to `Follower.yaml`.

### Leader / follower teleoperation

Wire the follower to CAN port 1, leader to CAN port 2, then:

```bash
python 5_teleop_control.py
```

## Two arm variants

The repo ships configs for two physical arms; switch between them with symlinks in `robot_param/`:

| Arm | Description | yaml |
|---|---|---|
| **Panthera-HT** (default) | Heavy-torque, motors 5047_36 / 6056_36 / 4438_30 | `Follower_panthera.yaml` / `Leader_panthera.yaml` |
| **XLeRobot-HT** | Lightweight, motors 4438_30 / 3536_32 | `Follower_xlerobot_ht.yaml` / `Leader_xlerobot_ht.yaml` |

```bash
cd panthera_python/robot_param

# Switch to XLeRobot-HT
ln -sf Follower_xlerobot_ht.yaml Follower.yaml
ln -sf Leader_xlerobot_ht.yaml   Leader.yaml

# Switch to Panthera-HT
ln -sf Follower_panthera.yaml Follower.yaml
ln -sf Leader_panthera.yaml   Leader.yaml

readlink Follower.yaml   # confirm which one is active
```

`Panthera()` with no argument loads `Follower.yaml` — scripts are arm-agnostic. After switching, **re-run** `0_robot_get_state.py` (verify motors online) and `0_robot_set_zero.py` (the zero positions differ between the two arms).

Each yaml field of interest:
- `robot.param_file` → `motor_param/6dof_{Panthera,xlerobot_ht}_params_{leader,follower}.yaml` — per-joint motor models
- `robot.max_torque` → Panthera-HT: `[21, 36, 36, 21, 10, 10]`, XLeRobot-HT: `[10, 10, 10, 10, 3.7, 3.7]` (matches URDF effort limits)
- `robot.joint_limits` → matches the corresponding URDF
- `urdf.file_path` → corresponding URDF file
- `urdf.end_effector_link` → `tool_link` (a fixed joint at the end of all four URDFs with a 0.165 m X offset as the tool tip reference; the leader yamls default to `joint6` since the leader doesn't need a tool offset)

## Usage example

```python
from Panthera_lib import Panthera
import numpy as np

robot = Panthera()                       # default loads Follower.yaml
# robot = Panthera("path/to/Leader.yaml")  # or specify

joint_pos = robot.get_current_pos()
print(f"current joint angles: {joint_pos}")

# Per-joint position + velocity + max-torque control
target_pos = [0.0, 0.5, -0.5, 0.0, 0.5, 0.0]
target_vel = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
max_torque = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]

robot.Joint_Pos_Vel(target_pos, target_vel, max_torque)

# Same call but block until reached
robot.Joint_Pos_Vel(target_pos, target_vel, max_torque,
                    iswait=True, tolerance=0.01, timeout=15.0)

# Gripper
robot.gripper_open()
# robot.gripper_close(pos=0.0)
```

## API reference

### `Panthera` class

`Panthera` extends `htr.Robot` with arm-level helpers.

#### Constructor
- `Panthera(config_path=None)` — `config_path` defaults to `robot_param/Follower.yaml`

#### State
- `get_current_state()` — list of all joint states
- `get_current_pos()` — joint angles, `np.ndarray[6]`
- `get_current_vel()` — joint velocities, `np.ndarray[6]`
- `get_current_torque()` — joint torques, `np.ndarray[6]`
- `get_current_state_gripper()`, `get_current_pos_gripper()`, `get_current_vel_gripper()`, `get_current_torque_gripper()` — gripper equivalents

#### Control commands
- `Joint_Pos_Vel(pos, vel, max_tqu=None, iswait=False, tolerance=0.1, timeout=15.0)` — per-joint position + velocity + max-torque
  - `pos` target angles `[6]` (rad), `vel` target speeds `[6]` (rad/s), `max_tqu` per-joint torque cap `[6]` (Nm; defaults from yaml), `iswait` block until reached
- `moveJ(pos, duration, max_tqu=None, iswait=False, tolerance=0.1, timeout=15.0)` — joint-space synchronized motion (all joints reach target simultaneously)
- `pos_vel_tqe_kp_kd(pos, vel, tqe, kp, kd)` — 5-parameter MIT mode
  - `tqe` feed-forward torque `[6]` (Nm), `kp` position gain `[6]`, `kd` velocity gain `[6]`
- `gripper_control(pos, vel, max_tqu)` — gripper pos/vel
- `gripper_control_MIT(pos, vel, tqe, kp, kd)` — gripper MIT mode
- `gripper_open(vel=0.5, max_tqu=0.5)` / `gripper_close(pos=0.0, vel=0.5, max_tqu=0.5)`

#### Position checking
- `check_position_reached(target_positions, tolerance=0.1)`
- `wait_for_position(target_positions, tolerance=0.01, timeout=15.0)`

#### Kinematics
- `forward_kinematics(joint_angles=None)` — returns `{'position': [x,y,z], 'rotation': R, 'transform': T, 'joint_angles': q}`
- `inverse_kinematics(target_position, target_rotation=None, init_q=None, max_iter=1000, eps=1e-4)` — returns `[6]` joint angles or `None`

#### Dynamics
- `get_Gravity(q=None)` — `G(q)`
- `get_Coriolis(q=None, v=None)` — `C(q,v)` matrix
- `get_Coriolis_vector(q=None, v=None)` — `C(q,v) * v`
- `get_Mass_Matrix(q=None)` — `M(q)`
- `get_Inertia_Terms(q=None, a=None)` — `M(q) * a`
- `get_Dynamics(q=None, v=None, a=None)` — `τ = M(q)a + C(q,v)v + G(q)`
- `get_friction_compensation(vel=None, Fc=None, Fv=None, vel_threshold=0.01)` — `τ_friction = Fc · sign(v) + Fv · v`

#### Trajectory planning
- `quintic_interpolation(start_pos, end_pos, duration, current_time)` — `(pos, vel, acc)`
- `septic_interpolation(start_pos, end_pos, duration, current_time)` — `(pos, vel, acc)`
- `septic_interpolation_with_velocity(start_pos, end_pos, start_vel, end_vel, duration, current_time)` — `(pos, vel, acc)`

#### Inherited from `htr.Robot`
- `motor_send_cmd()` — push the current command frame to motors
- `send_get_motor_state_cmd()` — request a state update
- `set_stop()` / `set_reset()` — stop / restart all motors
- `set_timeout(timeout_ms)` — communication timeout

### Motor parameter files (`motor_param/*.yaml`)

Per-joint motor IDs, CAN bus assignment, motor models.

## Project layout

```
panthera_python/
├── motor_whl/                          Prebuilt wheels (cp39/310/311/312, x86_64 / aarch64)
├── Panthera-HT_description/            Panthera-HT arm URDF
├── xlerobot_ht_description/            XLeRobot-HT arm URDF
├── xlerobot_ht_description_gripper/    XLeRobot-HT arm URDF (with gripper)
├── robot_param/
│   ├── Follower.yaml / Leader.yaml     Symlinks to the active variant
│   ├── Follower_panthera.yaml          Panthera-HT
│   ├── Leader_panthera.yaml
│   ├── Follower_xlerobot_ht.yaml       XLeRobot-HT
│   ├── Leader_xlerobot_ht.yaml
│   └── motor_param/
│       ├── 6dof_Panthera_params_{leader,follower}.yaml
│       ├── 6dof_xlerobot_ht_params_{leader,follower}.yaml
│       ├── motor_1.yaml, motor_6.yaml
│       └── robot_config.yaml
├── scripts/
│   ├── Panthera_lib/                   High-level wrapper (source-imported)
│   │   ├── Panthera.py
│   │   ├── recorder.py
│   │   └── __init__.py
│   ├── 0_robot_get_state.py / 0_robot_set_zero.py
│   ├── 1_Joint_*.py / 1_moveJ_control.py / 1_*_kinematics_test.py
│   ├── 2_*_compensation_control.py / 2_Jointimpendence_*.py / 2_inv_PosVel_control.py
│   ├── 3_interpolation_control_*.py / 3_sin_trajectory_control.py / 3_gravity_compensation_with_fk.py
│   ├── 4_impedance_trajectory_control_with_gra_pd.py
│   ├── 5_teleop_control.py / 5_record_trajectory.py / 5_replay_trajectory.py
│   ├── 6_moveL_pos_control.py / 6_moveL_rotate_control.py
│   ├── 7_keyboard_*.py                 Keyboard end-effector control (terminal + web variants)
│   ├── 8_arm_base_web_*.py             Arm + ODrive base web control
│   └── motor_example/                  Low-level motor examples (01_..09_) + motor_README.md
├── images/                             Diagrams referenced by this README
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## Troubleshooting

### URDF fails to load
Check `urdf.file_path` in the yaml — relative paths are resolved from the directory of the yaml file.

### IK doesn't converge
Verify the target pose is inside the arm's reachable workspace. Tune `max_iter` and `eps`.

### Motors don't connect
Check the board's power switch — the motor power button should be solid green. If it's lit but motors still fail, check inter-motor wiring.

### Auto-grant serial permissions

Create a udev rule:

```bash
sudo nano /etc/udev/rules.d/99-tegra-devices.rules
```

Add:

```
KERNEL=="ttyACM*", MODE="0777"
```

Reload:

```bash
sudo udevadm control --reload-rules
```

Reconnect the board for the rule to take effect.

## Common issues

### `libserialport.so.0` missing

```bash
sudo apt-get update
sudo apt-get install libserialport-dev
dpkg -L libserialport-dev | grep "\.so"
```

### `import hightorque_robot` fails: `libyaml-cpp.so.0.6: cannot open shared object file`

The motor wheel is built against `libyaml-cpp` 0.6.x, but Ubuntu 22 ships 0.7 and 24 ships 0.8. Build 0.6.1 from source:

```bash
# Check current version
ls -la /usr/local/lib/libyaml-cpp*
# Expect to see libyaml-cpp.so.0.7 or .so.0.8

# Remove existing version(s) (only the .so symlinks — leave system-installed 0.7/0.8 alone if you can)
sudo rm /usr/local/lib/libyaml-cpp.so.0.7
sudo rm /usr/local/lib/libyaml-cpp.so.0.7.0

# Build 0.6.1 from source
cd ~
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
git checkout yaml-cpp-0.6.1

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
```

## License

Apache License 2.0 — see the repo-level [LICENSE](../LICENSE).
