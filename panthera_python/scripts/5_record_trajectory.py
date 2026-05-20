#!/usr/bin/env python3
"""
Single leader-arm gravity compensation program with real-time trajectory
recording (position + velocity + gripper).
"""
import os
import time
import numpy as np
from Panthera_lib import Panthera, TrajectoryRecorder

# ---------------- Parameters ----------------
DO_RECORD = True          # True=record, False=do not record
REC_FILE  = None          # None=auto-generate filename
# --------------------------------------------

def main():
    # Get current state of the leader arm
    Leader_positions = Leader.get_current_pos()
    Leader_velocity = Leader.get_current_vel()

    # Get current gripper state
    gripper_pos = Leader.get_current_pos_gripper()
    gripper_vel = Leader.get_current_vel_gripper()

    # Compute gravity compensation torque
    Leader_gra = Leader.get_Gravity()

    # Add friction compensation
    Leader_tor = np.array(Leader_gra) + Leader.get_friction_compensation(Leader_velocity, Fc, Fv, vel_threshold)

    # Torque cap (per motor spec)
    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    Leader_tor = np.clip(Leader_tor, -tau_limit, tau_limit)

    # Zero stiffness, zero damping control (pure gravity compensation mode, free to drag around)
    Leader.pos_vel_tqe_kp_kd(zero_pos, zero_vel, Leader_tor, zero_kp, zero_kd)

    # Zero stiffness, zero damping gripper control (free to drag)
    Leader.gripper_control_MIT(0.0, 0.0, 0.0, 0.0, 0.0)

    # Print 6 joints + gripper
    print("\r", end="")
    for i in range(Leader.motor_count):
        print(f"J{i+1}: {Leader_positions[i]:6.3f}rad {Leader_velocity[i]:6.3f}rad/s | ", end="")
    print(f"gripper: {gripper_pos:6.3f}rad {gripper_vel:6.3f}rad/s   ", end="", flush=True)

    time.sleep(0.001)

if __name__ == "__main__":
    # Create robot instance (during recording the Leader arm is moved, so we load Leader.yaml)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Leader.yaml")
    Leader = Panthera(config_path)

    # Create zero position/velocity arrays
    zero_pos = [0.0] * Leader.motor_count
    zero_vel = [0.0] * Leader.motor_count
    zero_kp = [0.0] * Leader.motor_count
    zero_kd = [0.0] * Leader.motor_count

    # Friction compensation params are loaded from the friction section of Leader.yaml
    Fc = Leader.Fc
    Fv = Leader.Fv
    vel_threshold = Leader.vel_threshold

    # Instantiate the recorder (if recording is enabled)
    if DO_RECORD:
        rec = TrajectoryRecorder(REC_FILE)
        print("Starting trajectory recording (position + velocity + gripper)...")

    try:
        # Before recording, send a few read commands so we don't capture stale 999 joint states
        for i in range(10):
            Leader.send_get_motor_state_cmd()
            time.sleep(0.1)
        while True:
            main()                            # gravity compensation control loop
            if DO_RECORD:
                # Log joint position/velocity + gripper position/velocity
                rec.log(
                    Leader.get_current_pos(),
                    Leader.get_current_vel(),
                    Leader.get_current_pos_gripper(),
                    Leader.get_current_vel_gripper()
                )
    except KeyboardInterrupt:
        if DO_RECORD:
            rec.close()
        print("\nProgram stopped" + (", trajectory saved" if DO_RECORD else ""))
