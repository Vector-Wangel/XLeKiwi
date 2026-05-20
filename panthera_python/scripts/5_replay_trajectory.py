#!/usr/bin/env python3
"""
Replay a *.jsonl trajectory file (single arm + gripper).
Supports position + velocity + gripper replay.
"""
import os,sys
import numpy as np
from Panthera_lib import Panthera, TrajectoryRecorder

# ---------------- Parameters ----------------
TRAJECTORY_FILE = "trajectory_test_1.jsonl"  # change to the actual recorded filename

# Joint PD gains
kp_play = [30.0, 40.0, 55.0, 15.0, 7.0, 5.0]        # stiffness during replay (tunable)
kd_play = [3.0, 4.0, 5.5, 1.5, 0.7, 0.5]            # damping during replay

# Gripper PD gains
gripper_kp = 5.0   # gripper stiffness
gripper_kd = 0.5   # gripper damping

# Torque limits
tau_limit = [15.0, 30.0, 30.0, 15.0, 5.0, 5.0]
# --------------------------------------------

if __name__ == "__main__":
    if not os.path.isfile(TRAJECTORY_FILE):
        print(f"File does not exist: {TRAJECTORY_FILE}")
        sys.exit(1)

    # Create the robot (use the same config as when recording)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Follower.yaml")
    robot = Panthera(config_path)

    # Friction compensation params are loaded from the friction section of Follower.yaml
    # (auto-switched to match the currently active arm).
    Fc = robot.Fc
    Fv = robot.Fv
    vel_threshold = robot.vel_threshold

    print(f"Starting replay: {TRAJECTORY_FILE}")
    print("Data format: auto-detected (supports position + velocity + gripper)")

    try:
        # One-shot replay: send position+velocity+gripper at the original recorded sampling intervals
        TrajectoryRecorder.play(
            robot=robot,
            filepath=TRAJECTORY_FILE,
            kp=kp_play,
            kd=kd_play,
            fc=Fc,
            fv=Fv,
            vel_threshold=vel_threshold,
            tau_limit=tau_limit,
            gripper_kp=gripper_kp,
            gripper_kd=gripper_kd
        )

    except KeyboardInterrupt:
        print("\nReplay interrupted")
    finally:
        # robot.set_stop()
        print("Motors stopped")
