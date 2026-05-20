#!/usr/bin/env python3
"""
Simple position-velocity control program for a 6-joint robot arm.
Modify the target position array directly in the code to control the robot.
"""
import time
from Panthera_lib import Panthera

def main():
    # Send position control command (using blocking mode)
    print("\nsend control command...")
    zero_success = robot.Joint_Pos_Vel(zero_pos, vel, max_torque, iswait=True)
    print(f"execution status 0: {zero_success}")
    time.sleep(1)

    robot.Joint_Pos_Vel(pos2, vel, max_torque, iswait=True)
    robot.gripper_close()
    time.sleep(2)

    robot.Joint_Pos_Vel(pos1, vel, max_torque, iswait=True)
    robot.gripper_open()
    time.sleep(2)

    robot.Joint_Pos_Vel(pos2, vel, max_torque, iswait=True)
    robot.gripper_close()
    time.sleep(2)

    zero_success = robot.Joint_Pos_Vel(zero_pos, vel, max_torque, iswait=True)
    print(f"execution status 0: {zero_success}")
    time.sleep(2)

    # Hold position for 2 seconds
    print("\nholding position for 2 seconds...")
    time.sleep(2)
    # Motors auto-power-off on exit. Be careful.

if __name__ == "__main__":
    robot = Panthera()
    zero_pos = [0.0] * robot.motor_count
    pos1 = [0.0, 0.8, 0.8, 0.3, 0.0, 0.0] 
    pos2 = [0.0, 1.2, 1.2, 0.4, 0.0, 0.0] 
    pos3 = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0] 
    vel = [0.5] * robot.motor_count
    max_torque = robot.max_torque.tolist()
    try:
        main()
    except KeyboardInterrupt:
        print("\n\ninterrupted")
    except Exception as e:
        print(f"\nerror: {e}")
    finally:
        print("\n\nall motors stopped")