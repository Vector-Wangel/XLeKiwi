#!/usr/bin/env python3
"""
Simple PD control program for a 6-joint robot arm.
Modify the target position array directly in the code to control the robot.
"""
import time
from Panthera_lib import Panthera

if __name__ == "__main__":
    robot = Panthera()
    zero_pos = [0.0] * robot.motor_count
    zero_vel = [0.0] * robot.motor_count
    zero_tqe = [0.0] * robot.motor_count
    pos1 = [0.0, 0.7, 0.7, -0.1, 0.0, 0.0]
    kp = [4.0, 10.0, 10.0, 2.0, 2.0, 1.0]
    kd = [0.5, 0.8, 0.8, 0.2, 0.2, 0.1]
    vel = [0.3] * robot.motor_count
    max_torque = robot.max_torque.tolist()
    try:
        while(1):
            robot.pos_vel_tqe_kp_kd(pos1, zero_vel, zero_tqe, kp, kd)

            positions = robot.get_current_pos()
            velocities = robot.get_current_vel()
            torque = robot.get_current_torque()
            # Print 6 joints
            for i in range(robot.motor_count):
                print(f"joint{i+1}: pos={positions[i]:7.3f} rad, vel={velocities[i]:7.3f} rad/s, tqe={torque[i]:7.3f}")
            print("-" * 60)

            time.sleep(1)
            
    except KeyboardInterrupt:
        # Without this line, motors power down when the script stops
        # robot.set_stop()
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")