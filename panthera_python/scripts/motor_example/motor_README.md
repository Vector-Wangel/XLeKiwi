# Low-level motor control Python interface

Python bindings for the high-torque motor control C++ library (via pybind11). Use this when you need direct motor-level access — read this file (and skim the examples) before writing your own code.

## Features

- Supports multiple high-torque motor models (4438, 5046, 5047, 6056 series, ...)
- Control modes:
  - Position
  - Velocity
  - Torque
  - Voltage
  - Current
  - Mixed (position + velocity + torque + PID gains)
- Multi-CAN bus, multi-motor
- Safety: position limits, torque limits, timeout detection
- YAML-based configuration

## Dependencies

### System

```bash
sudo apt-get install -y \
    cmake python3-dev python3-pip \
    liblcm-dev libyaml-cpp-dev libserialport-dev
```

### Python

```bash
pip3 install pybind11 numpy
```

### C++ project (build first)

```bash
cd path/to/hightorque_robot
mkdir -p build && cd build
cmake ..
make
```

## Install

### Option 1 — pip editable

```bash
cd path/to/hightorque_robot_python
pip3 install -e .
```

### Option 2 — CMake build

```bash
cd path/to/hightorque_robot_python
mkdir -p build && cd build
cmake ..
make
```

## Examples

### Basic

```python
import hightorque_robot as htr

# Instantiate from a yaml config
robot = htr.Robot("/path/to/robot_config.yaml")

# Enable LCM publishing
robot.lcm_enable()

# Enumerate motors
motors = robot.get_motors()
print(f"motor count: {len(motors)}")

# Drive motor 0
motor = motors[0]
motor.position(0.5)   # position mode
# motor.velocity(1.0) # velocity mode
# motor.torque(5.0)   # torque mode

# Push the command frame
robot.motor_send_cmd()

# Read state
state = motor.get_current_motor_state()
print(f"motor {state.ID}: pos={state.position}, vel={state.velocity}, tqe={state.torque}")

# Stop all
robot.set_stop()
```

### Mixed control loop

```python
import hightorque_robot as htr
import time

robot = htr.Robot("/path/to/robot_config.yaml")
motors = robot.get_motors()

# Initial command: position + velocity + max-torque
for motor in motors:
    motor.pos_vel_MAXtqe(position=1.0, velocity=0.5, torque_max=10.0)
robot.motor_send_cmd()

while True:
    for i, motor in enumerate(motors):
        motor.pos_vel_MAXtqe(1.0 if i % 2 == 0 else -1.0, 0.1, 10.0)
    robot.motor_send_cmd()

    for motor in motors:
        state = motor.get_current_motor_state()
        print(f"motor {state.ID}: pos={state.position:.3f}, vel={state.velocity:.3f}")

    time.sleep(0.001)  # 1 kHz
```

### Reading the config

```python
import hightorque_robot as htr

robot = htr.Robot("path/to/hightorque_robot/robot_param/robot_config.yaml")
print(f"robot name: {robot.robot_params.robot_name}")
print(f"motor count: {len(robot.get_motors())}")
robot.set_timeout(100)   # 100 ms
robot.set_reset()
```

## API reference

### `Robot`
- `Robot()` / `Robot(config_path)` — constructor
- `get_motors()` — list of `Motor` objects
- `motor_send_cmd()` — push the current command frame to motors
- `set_stop()` / `set_reset()` — stop / restart all motors
- `set_reset_zero()` / `set_reset_zero(motor_id)` — reset motor zero
- `send_get_motor_state_cmd()` — request a state update
- `set_timeout(timeout_ms)` — comms timeout
- `lcm_enable()` — enable LCM publishing

### `Motor`
- `position(pos)`, `velocity(vel)`, `torque(tqe)` — single-quantity control
- `voltage(vol)`, `current(cur)` — open-loop voltage / closed-loop current
- `pos_vel_MAXtqe(pos, vel, tqe_max)` — position + velocity + torque cap
- `pos_vel_tqe_kp_kd(pos, vel, tqe, kp, kd)` — full 5-parameter MIT mode
- `pos_vel_kp_kd(pos, vel, kp, kd)` — position + velocity + PID
- `pos_vel_acc(pos, vel, acc)` — position + velocity + acceleration
- `get_current_motor_state()` — read state
- `get_motor_id()` — motor ID
- `stop()` / `brake()` / `reset()`

### `MotorState`
- `ID` — motor ID
- `mode` — operating mode
- `fault` — fault code
- `position` (rad), `velocity` (rad/s), `torque` (N·m)
- `Kp`, `Kd` — current PID gains

## Troubleshooting

### Can't find C++ lib
Build it: `cd path/to/hightorque_robot/build && cmake .. && make`

### Can't find pybind11
`pip3 install pybind11`

### Serial permission denied
```bash
sudo usermod -a -G dialout $USER
# log out and back in
```

### Import error
```bash
export PYTHONPATH=path/to/hightorque_robot_python:$PYTHONPATH
```

## Directory layout

```
motor_example/
├── motor_control.py            generic example
├── 01_motor_get_status.py      read motor state
├── 02_position_control.py      position
├── 03_velocity_control.py      velocity
├── 04_torque_control.py        torque
├── 05_voltage_control.py       voltage
├── 06_current_control.py       current
├── 07_pos_vel_maxtorque_control.py     pos + vel + max-torque
├── 08_pos_vel_torque_kp_kd_control.py  5-parameter MIT
└── 09_set_zero.py              set motor zero
```

## License

Apache License 2.0 — see the repo-level [LICENSE](../../../LICENSE).
