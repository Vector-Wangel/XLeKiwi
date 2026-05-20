#!/usr/bin/env python3
import time
import math
import os
import sys
# Add the python directory to the path so we can import the module
# Go up two levels from motor_example to reach the panthera_python directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import hightorque_robot as htr


if __name__ == "__main__":
    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../../robot_param/motor_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    motors = robot.get_motors()

    cnt = 0

    print(f"Controlling {len(motors)} motors")

    while True:
        # Print state for all motors
        if(cnt% 1000 == 0):
            for motor in motors:
                state = motor.get_current_motor_state()
                print(f"Motor {motor.get_motor_id()} state:")
                print(f"  Position: {state.position:.4f} rad ({htr.rad_to_deg(state.position):.2f} deg)")
                print(f"  Velocity: {state.velocity:.4f} rad/s")
                print(f"  Torque: {state.torque:.4f} Nm")
                print(f"  Mode: {state.mode}")
                print(f"  Fault code: 0x{state.fault:02X}")
            print("-" * 40)
        cnt += 1

        for motor in motors:
            # Set target position to a sinusoidal trajectory
            t = time.time()
            target_position = 1.0 * math.sin(2.0 * math.pi * 0.1 * t)  # 0.1 Hz, amplitude 1.0 rad
            motor.position(target_position)
        robot.motor_send_cmd()

        time.sleep(0.001)

    robot.set_stop()
