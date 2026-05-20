#!/usr/bin/env python3
"""
Simple single-joint velocity control program.
The first joint toggles between +0.2 and -0.2 rad/s every 3 seconds.
"""
import time
import math
from Panthera_lib import Panthera

if __name__ == "__main__":
    robot = Panthera()

    try:
        while True:
            t = time.time()
            target_vel = [0.2 if (math.floor(t) % 6) >= 3 else -0.2] + [0.0] * (robot.motor_count - 1)
            robot.Joint_Vel(target_vel)

            # Print status
            print(f"target velocity: {target_vel[0]:.2f} rad/s")
            print(f"current position: {robot.get_current_pos()[0]:.3f} rad")
            print(f"current velocity: {robot.get_current_vel()[0]:.3f} rad/s")
            print("-" * 40)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\ninterrupted")
    except Exception as e:
        print(f"\nerror: {e}")
