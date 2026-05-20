#!/usr/bin/env python3
"""
Gravity compensation program.
Compensates the gravity term only.
"""
import time
import numpy as np
from Panthera_lib import Panthera

def main():
    tor = robot.get_Gravity()  # call function to get gravity compensation torque
    # Torque cap (per motor spec)
    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    tor = np.clip(tor, -tau_limit, tau_limit)
    robot.pos_vel_tqe_kp_kd(zero_pos, zero_vel, tor, zero_kp, zero_kd)
    robot.gripper_control_MIT(0,0,0,0,0)
    print(f"gravity compensation torque:",tor)
    time.sleep(0.002)
    #Motors auto-power-off on exit. Be careful.

if __name__ == "__main__":
    robot = Panthera()
    # Create zero position/velocity arrays
    zero_pos = [0.0] * robot.motor_count
    zero_vel = [0.0] * robot.motor_count
    zero_kp = [0.0] * robot.motor_count
    zero_kd = [0.0] * robot.motor_count
    zero_tor = [0.0] * robot.motor_count # used for debugging
    try:
        while(1):
            main()
    except KeyboardInterrupt:
        # Without this line, motors power down when the script stops
        # robot.set_stop()
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")