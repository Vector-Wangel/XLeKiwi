#!/usr/bin/env python3
"""
Read and print arm joint state
Continuously print 6 joints + gripper state
"""

import time
from Panthera_lib import Panthera

def print_robot_state(robot):
    """Print robot state"""
    # Read joint angles
    # Send a state-request frame so motors push their feedback
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    positions = robot.get_current_pos()
    velocities = robot.get_current_vel()

    # Read gripper state
    gripper_state = robot.Motors[6].get_current_motor_state()
    
    print("\n" + "="*50)
    print("Arm state")
    print("="*50)
    
    # Print 6 joints
    for i in range(robot.motor_count):
        print(f"joint{i+1}: pos={positions[i]:7.3f} rad, vel={velocities[i]:7.3f} rad/s")
    
    # Print gripper
    print(f"gripper:   pos={gripper_state.position:7.3f} rad, vel={gripper_state.velocity:7.3f} rad/s")

def main():
    robot = Panthera()
    
    try:
        robot.set_reset_zero()
        robot.motor_send_cmd()
        time.sleep(1)  # update every 0.5s
        while True:
            print_robot_state(robot)
            time.sleep(0.5)  # update every 0.5s
            
    except KeyboardInterrupt:
        print("\n\ninterrupted")

if __name__ == "__main__":
    main()