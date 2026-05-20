#!/usr/bin/env python3
"""
Gravity + friction compensation program.
Compensates the gravity term and the friction term.
Uses a Coulomb + viscous friction model.
"""
import time
import numpy as np
from Panthera_lib import Panthera

def main():
    # Get current joint velocity
    vel = robot.get_current_vel()

    # Get gravity compensation torque
    tau_gravity = robot.get_Gravity()

    # Compute friction compensation torque
    tau_friction = robot.get_friction_compensation(vel, Fc, Fv, vel_threshold)

    # Total compensation torque = gravity compensation + friction compensation
    tau_total = tau_gravity + tau_friction

    # Torque cap (per motor spec)
    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    tau_total = np.clip(tau_total, -tau_limit, tau_limit)

    # Send control command
    robot.pos_vel_tqe_kp_kd(zero_pos, zero_vel, tau_total, zero_kp, zero_kd)

    # Print info
    print(f"velocity: {vel}")
    print(f"gravity compensation: {tau_gravity}")
    print(f"friction compensation: {tau_friction}")
    print(f"total compensation torque: {tau_total}")
    print("-" * 60)

    time.sleep(0.005)


if __name__ == "__main__":
    robot = Panthera()

    # Friction params are loaded from the friction section of the current yaml
    # (auto-switched per arm type: large/small arm).
    Fc = robot.Fc
    Fv = robot.Fv
    vel_threshold = robot.vel_threshold

    # Create zero position / velocity / stiffness / damping arrays
    zero_pos = [0.0] * robot.motor_count
    zero_vel = [0.0] * robot.motor_count
    zero_kp = [0.0] * robot.motor_count
    zero_kd = [0.0] * robot.motor_count

    print("=" * 60)
    print("Gravity + friction compensation control started")
    print("=" * 60)
    print(f"Coulomb friction coefficient Fc: {Fc}")
    print(f"Viscous friction coefficient Fv: {Fv}")
    print(f"Velocity threshold: {vel_threshold} rad/s")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the program\n")

    try:
        while True:
            main()
    except KeyboardInterrupt:
        # robot.set_stop()
        print("\n\ninterrupted")
        print("all motors stopped")
    except Exception as e:
        # robot.set_stop()
        print(f"\nerror: {e}")
        print("all motors stopped")
