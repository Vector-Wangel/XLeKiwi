#!/usr/bin/env python3
"""
Simple moveJ control program for a 6-joint robot arm.
By specifying a duration, joint velocities are computed automatically so that
all joints reach the target position simultaneously.
Modify the target position array and duration directly in the code to control
the robot.
"""
import time
from Panthera_lib import Panthera

def main():
    # Manually send a get-state command before issuing motion commands to
    # refresh the cached state.
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    time.sleep(0.5)
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    time.sleep(0.5)
    # Send position-time control command
    print("\nsend control command...")
    zero_success = robot.moveJ(zero_pos, duration=2.0, max_tqu=max_torque, iswait=True)
    print(f"execution status 0: {zero_success}")
    time.sleep(1)

    # Move to position 1, reach in 3 seconds
    robot.moveJ(pos1, duration=3.0, max_tqu=max_torque, iswait=True)
    robot.gripper_close()
    time.sleep(2)

    # Move to position 2, reach in 2.5 seconds
    robot.moveJ(pos2, duration=2.5, max_tqu=max_torque, iswait=True)
    robot.gripper_open()
    time.sleep(2)

    # Move back to position 1, reach in 3 seconds
    robot.moveJ(pos1, duration=3.0, max_tqu=max_torque, iswait=True)
    robot.gripper_close()
    time.sleep(2)

    # Return to zero position, reach in 2 seconds
    zero_success = robot.moveJ(zero_pos, duration=2.0, max_tqu=max_torque, iswait=True)
    print(f"execution status 0: {zero_success}")
    time.sleep(2)

    # Hold position for 2 seconds
    print("\nholding position for 2 seconds...")
    time.sleep(2)
    # Motors auto-power-off on exit. Be careful.

if __name__ == "__main__":
    robot = Panthera()
    # Define target position (radians)
    zero_pos = [0.0] * robot.motor_count
    pos1 = [0.5, 0.8, 0.8, 0.3, 0.0, 0.0]
    pos2 = [-0.3, 1.2, 1.2, 0.4, 0.0, 0.0]
    vel = [0.5]*robot.motor_count

    # Max torque (Nm), loaded from yaml automatically for the current arm
    max_torque = robot.max_torque.tolist()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\ninterrupted")
    except Exception as e:
        print(f"\nerror: {e}")
    finally:
        print("\n\nall motors stopped")
