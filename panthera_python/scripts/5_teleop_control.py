#!/usr/bin/env python3
"""
Leader-follower arm teleoperation program.
"""
import time
import sys
import os
import numpy as np
from Panthera_lib import Panthera

def main():
    # ****** Arm control *******
    # Read the leader arm's position and velocity to drive the follower arm
    Leader_positions = Leader.get_current_pos()
    Leader_velocity = Leader.get_current_vel()
    Follower_velocity = Follower.get_current_vel()
    # Read follower-arm torque to compute force feedback
    Follower_torque = Follower.get_current_torque()
    # Compute gravity torques
    Leader_gra = Leader.get_Gravity()
    Follower_gra = Follower.get_Gravity()
    # External force experienced by the follower arm
    tor_diff = np.array(Follower_torque) - np.array(Follower_gra)
    # Per-element thresholding: zero out values below the threshold
    tor_diff[np.abs(tor_diff) < tor_threshold] = 0
    # Leader-arm torque
     # Force-feedback mode:
    # Leader_tor = np.array(Leader_gra) - tor_diff*0.8 + Leader.get_friction_compensation(Leader_velocity, Leader.Fc, Leader.Fv, Leader.vel_threshold)
     # No-force-feedback mode (smoother):
    Leader_tor = np.array(Leader_gra) + Leader.get_friction_compensation(Leader_velocity, Leader.Fc, Leader.Fv, Leader.vel_threshold)
    # Follower-arm torque
    Follower_tor = np.array(Follower_gra) + Follower.get_friction_compensation(Follower_velocity, Follower.Fc, Follower.Fv, Follower.vel_threshold)
    # Torque cap (per motor spec)
    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    Leader_tor = np.clip(Leader_tor, -tau_limit, tau_limit)
    Follower_tor = np.clip(Follower_tor, -tau_limit, tau_limit)
    # Run control
    Leader.pos_vel_tqe_kp_kd(zero_pos, zero_vel, Leader_tor, zero_kp, zero_kd)
    Follower.pos_vel_tqe_kp_kd(Leader_positions, Leader_velocity, Follower_tor, kp, kd)

    # ******* gripper control *******
    Leader_gripper_positions = Leader.get_current_pos_gripper()
    Leader_gripper_velocity = Leader.get_current_vel_gripper()
    Follower_gripper = Follower.get_current_state_gripper()
    gripper_torque = Follower.get_friction_compensation(Leader_gripper_velocity, 0.06, 0.0, 0.15) - Follower_gripper.torque*0.5
    tor_diff[np.abs(gripper_torque) < 0.2] = 0
    Leader.gripper_control_MIT(1.5, 0, gripper_torque, 0.2, 0.02)
    Follower.gripper_control_MIT(Leader_gripper_positions, Leader_gripper_velocity, 0, gripper_kp, gripper_kd)

    # Print 6 joints
    for i in range(Leader.motor_count):
        print(f"joint{i+1}: pos={Leader_positions[i]:7.3f} rad, vel={Leader_velocity[i]:7.3f} rad/s")
    print(f"feedback torque:",tor_diff)
    print(f"gripper torque: {gripper_torque:7.3f} Nm")
    print('-' * 40)

    time.sleep(0.001)
    #Motors auto-power-off on exit. Be careful.

if __name__ == "__main__":
    # Create robot instances
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Leader.yaml")
    Leader = Panthera(config_path)
    config_path = os.path.join(script_dir, "../robot_param/Follower.yaml")
    Follower = Panthera(config_path)
    # Create zero position/velocity arrays
    zero_pos = [0.0] * Leader.motor_count
    zero_vel = [0.0] * Leader.motor_count
    zero_kp = [0.0] * Leader.motor_count
    zero_kd = [0.0] * Leader.motor_count
    kp = [10.0, 21.0, 21.0, 16.0, 13.0, 1.0]
    kd = [1.0, 2.0, 2.0, 0.9, 0.8, 0.1]
    gripper_kp = 4.0
    gripper_kd = 0.4
    # Friction compensation params: Leader / Follower each load from their own yaml's friction section.
    # main() uses Leader.Fc / Leader.Fv / Leader.vel_threshold and the same-named attributes on Follower.
    tor_threshold = np.array([0.5, 1.0, 1.0, 0.5, 0.3, 0.3])

    try:
        while(1):
            main()
    except KeyboardInterrupt:
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")
