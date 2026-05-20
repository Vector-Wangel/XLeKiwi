#!/usr/bin/env python3
"""
Gravity compensation with FK output program.
Features:
1. Performs gravity compensation control
2. Streams the end-effector position and rotation matrix in real time
"""
import time
import numpy as np
from Panthera_lib import Panthera

def rotation_matrix_to_euler(R):
    """Convert a rotation matrix to Euler angles (ZYX order, in degrees)."""
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees([x, y, z])

def print_matrix(matrix, title, precision=3):
    """Pretty-print a matrix with a title."""
    print(f"\n{title}:")
    for row in matrix:
        print("  [" + "  ".join([f"{val:8.{precision}f}" for val in row]) + "]")

def main():
    # Get current joint angles
    current_angles = robot.get_current_pos()

    # Compute gravity compensation torque
    gravity_torque = robot.get_Gravity(current_angles)

    # Torque cap (per motor spec)
    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    gravity_torque = np.clip(gravity_torque, -tau_limit, tau_limit)

    # Apply gravity compensation control
    zero_pos = [0.0] * robot.motor_count
    zero_vel = [0.0] * robot.motor_count
    zero_kp = [0.0] * robot.motor_count
    zero_kd = [0.0] * robot.motor_count

    robot.pos_vel_tqe_kp_kd(zero_pos, zero_vel, gravity_torque, zero_kp, zero_kd)

    # Compute forward kinematics (FK)
    fk = robot.forward_kinematics(current_angles)

    # Print results
    print("\n" + "="*80)
    print("Gravity compensation control + forward kinematics (FK) result")
    print("="*80)

    # Print joint angles
    joint_angles_deg = [np.degrees(angle) for angle in current_angles]
    print(f"joint angles (deg): {[f'{a:7.2f}' for a in joint_angles_deg]}")

    # Print gravity compensation torque
    print(f"gravity compensation torque (Nm): {[f'{t:7.2f}' for t in gravity_torque]}")

    if fk:
        # Print end-effector position
        pos = fk['position']
        print(f"\nend-effector position (m): x={pos[0]:8.4f}, y={pos[1]:8.4f}, z={pos[2]:8.4f}")

        # Print rotation matrix
        rotation_matrix = fk['rotation']
        print_matrix(rotation_matrix, "rotation matrix (R)")

        # Print Euler angles
        euler_angles = rotation_matrix_to_euler(rotation_matrix)
        print(f"\nEuler angles (deg): Roll={euler_angles[0]:7.2f}, Pitch={euler_angles[1]:7.2f}, Yaw={euler_angles[2]:7.2f}")

        # Print 4x4 transformation matrix
        print_matrix(fk['transform'], "4x4 transformation matrix (T)", precision=4)

    time.sleep(0.002)

if __name__ == "__main__":
    robot = Panthera()

    try:
        print("Starting gravity compensation control and streaming forward kinematics (FK) result...")
        print("Press Ctrl+C to stop the program")

        while True:
            main()

    except KeyboardInterrupt:
        # Without this line, motors power down when the script stops
        # robot.set_stop()
        print("\n\ninterrupted")
        print("all motors stopped")
    except Exception as e:
        # robot.set_stop()
        print(f"\nerror: {e}")
        print("all motors stopped")
