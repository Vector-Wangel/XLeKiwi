#!/usr/bin/env python3
"""
Simple inverse-kinematics motion program.
Specify the end-effector pose, solve IK, then send the resulting joint
positions to the arm.
"""
import time
import numpy as np
from Panthera_lib import Panthera

def main():
    # Send position control command
    print("\nsend control command...")

    pos1 = robot.inverse_kinematics(ik_pos1, ik_rot2, robot.get_current_pos())
    # Always check that IK converged before executing; otherwise the program
    # will exit and the motors will power down.
    if pos1 is not None:
        issuccess1 = robot.moveJ(pos1, duration=3.0, max_tqu = max_torque, iswait=True)
        print(f"execution status 1: {issuccess1}")
        time.sleep(3)

    pos2 = robot.inverse_kinematics(ik_pos2, ik_rot2, robot.get_current_pos())
    if pos2 is not None:
        issuccess2 = robot.moveJ(pos2, duration=3.0, max_tqu = max_torque, iswait=True)
        print(f"execution status 2: {issuccess2}")
        time.sleep(3)

    pos3 = robot.inverse_kinematics(ik_pos3, ik_rot2, robot.get_current_pos())
    if pos3 is not None:
        issuccess3 = robot.moveJ(pos3, duration=3.0, max_tqu = max_torque, iswait=True)
        print(f"execution status 3: {issuccess3}")
        time.sleep(3)

    issuccess4 = robot.moveJ(zero_pos, duration=3.0, max_tqu = max_torque, iswait=True)
    print(f"execution status 4: {issuccess4}")
    # Hold position for 2 seconds
    print("\nholding position for 2 seconds...")
    time.sleep(2)
    #Motors auto-power-off on exit. Be careful.

if __name__ == "__main__":
    robot = Panthera()
    zero_pos = [0.0] * robot.motor_count
    vel = [0.5] * robot.motor_count
    max_torque = robot.max_torque.tolist()
    ik_pos1 = [0.20, 0.0, 0.1]
    ik_pos2 = [0.20, 0.0, 0.15]
    # ik_pos2 = [-0.16, 0.20, 0.18]
    # Out-of-range position provided as an example
    ik_pos3 = [0.74, 0.0, 0.2]

    # At the arm's zero pose, all frames share the same orientation.
    # Here we set the target EE orientation aligned with the base frame.
    ik_rot1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ik_rot2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    try:
        robot.Joint_Pos_Vel(zero_pos, vel, max_torque, iswait=True)
        main()
    except KeyboardInterrupt:
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")
