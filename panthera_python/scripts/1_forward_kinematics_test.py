#!/usr/bin/env python3
"""
Forward kinematics (FK) test program.
Continuously reads and prints the end-effector position and orientation.
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
    # Get current end-effector position and orientation
    # Send a state-request frame so motors push their feedback
    # robot.pos_vel_tqe_kp_kd([0.0]*6, [0.0]*6, [0.0]*6, [0.0]*6, [0.0]*6, iswait=False)
    robot.motor_send_cmd()
    current_angles = robot.get_current_pos()
    fk = robot.forward_kinematics(current_angles)

    if fk:
        print("\n" + "="*60)
        print("Arm forward kinematics (FK) result")
        print("="*60)

        # Print joint angles
        joint_angles_deg = [np.degrees(angle) for angle in fk['joint_angles']]
        print(f"joint angles (deg): {[f'{a:7.2f}' for a in joint_angles_deg]}")

        # Print end-effector position
        pos = fk['position']
        print(f"end-effector position (m): x={pos[0]:8.4f}, y={pos[1]:8.4f}, z={pos[2]:8.4f}")

        # Print rotation matrix
        rotation_matrix = fk['rotation']
        print_matrix(rotation_matrix, "rotation matrix (R)")

        # Print Euler angles
        euler_angles = rotation_matrix_to_euler(rotation_matrix)
        print(f"\nEuler angles (deg): Roll={euler_angles[0]:7.2f}, Pitch={euler_angles[1]:7.2f}, Yaw={euler_angles[2]:7.2f}")

        # Print 4x4 transformation matrix
        print_matrix(fk['transform'], "4x4 transformation matrix (T)", precision=4)

    time.sleep(1.0)

if __name__ == "__main__":
    robot = Panthera()

    try:
        while(1):
            main()
    except KeyboardInterrupt:
        robot.set_stop()
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")
