#!/usr/bin/env python3
"""
moveL orientation-change example.

Demonstrates how to use moveL to keep the end-effector position fixed while
only changing its orientation. Shows SLERP (spherical linear interpolation)
producing a smooth orientation transition.
"""

import sys
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from Panthera_lib import Panthera

def print_orientation_info(rotation_matrix, label="current orientation"):
    """Print orientation info (Euler angles)."""
    rot = R.from_matrix(rotation_matrix)
    euler = rot.as_euler('xyz', degrees=True)
    print(f"{label}:")
    print(f"  Roll:  {euler[0]:7.2f} deg")
    print(f"  Pitch: {euler[1]:7.2f} deg")
    print(f"  Yaw:   {euler[2]:7.2f} deg")


def main():
    print("="*60)
    print("moveL orientation-change example")
    print("Hold the EE position fixed and change only the orientation")
    print("="*60)

    # 1. Initialize the arm
    print("\nInitializing the arm...")
    robot = Panthera()

    # 2. Move to initial position
    print("\nMoving to initial position...")
    zero_pos = [0.0] * robot.motor_count
    vel = [0.5] * robot.motor_count
    robot.Joint_Pos_Vel(zero_pos, vel, iswait=True)
    time.sleep(1)

    # 3. Move to a comfortable working pose
    print("\nMoving to working pose...")
    # Define an initial pose
    target_pos = [0.32, 0.0, 0.25]  # EE position (m)
    target_rot = robot.rotation_matrix_from_euler(0, 0, 0)  # initial orientation: no rotation

    # Solve inverse kinematics (IK)
    joint_pos = robot.inverse_kinematics(target_pos, target_rot, robot.get_current_pos())
    if joint_pos is not None:
        robot.moveJ(joint_pos, duration=3.0, iswait=True)
        time.sleep(1)
    else:
        print("Initial-pose IK failed!")
        return

    # 4. Get current pose
    current_fk = robot.forward_kinematics()
    current_pos = np.array(current_fk['position'])
    current_rot = current_fk['rotation']

    print(f"\ncurrent EE position: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}] m")
    print_orientation_info(current_rot, "current orientation")


    # ========== Example 1: orbit around Z 45 deg ==========
    print("\n" + "="*60)
    print("Example 2: hold position, orbit around Z 45 deg")
    print("="*60)

    # Hold position, change only orientation
    target_rot_1 = robot.rotation_matrix_from_euler(0, 0, np.pi/4)  # orbit around Z 45 deg
    print_orientation_info(target_rot_1, "target orientation")

    success = robot.moveL(
        target_position=current_pos,  # hold position
        target_rotation=target_rot_1,  # change orientation
        duration=3.0,
        use_spline=True
    )

    if not success:
        print("Example 1 failed")
        return

    time.sleep(1)

    # ========== Example 2: rotate 45 deg about Y axis ==========
    print("\n" + "="*60)
    print("Example 2: hold position, rotate 45 deg about Y axis")
    print("="*60)

    # Get current orientation
    current_fk = robot.forward_kinematics()
    current_rot = current_fk['rotation']

    # Starting from the current orientation, rotate 45 deg about Y
    additional_rot = robot.rotation_matrix_from_euler(0, np.pi/4, 0)
    # Right-multiply, i.e. rotation in the EE frame; you can clearly see the EE rotate
    # 45 deg about its own Y axis.
    target_rot_2 = current_rot @ additional_rot
    print_orientation_info(target_rot_2, "target orientation")

    success = robot.moveL(
        target_position=current_pos,  # hold position
        target_rotation=target_rot_2,  # change orientation
        duration=3.0,
        use_spline=True
    )

    if not success:
        print("Example 2 failed")
        return

    time.sleep(1)


    # ========== Example 3: rotate -45 deg about X axis ==========
    print("\n" + "="*60)
    print("Example 3: hold position, rotate -45 deg about X axis")
    print("="*60)

    # Get current orientation
    current_fk = robot.forward_kinematics()
    current_rot = current_fk['rotation']

    # Starting from the current orientation, rotate -45 deg about X
    additional_rot = robot.rotation_matrix_from_euler(-np.pi/4, 0, 0)
    # Right-multiply, i.e. rotation in the EE frame; you can clearly see the EE rotate
    # -45 deg about its own X axis.
    target_rot_3 = current_rot @ additional_rot
    print_orientation_info(target_rot_3, "target orientation")

    success = robot.moveL(
        target_position=current_pos,  # hold position
        target_rotation=target_rot_3,  # change orientation
        duration=3.0,
        use_spline=True
    )

    if not success:
        print("Example 3 failed")
        return

    time.sleep(1)

    # ========== Example 4: return to initial orientation ==========
    print("\n" + "="*60)
    print("Example 4: return to initial orientation (no rotation)")
    print("="*60)

    target_rot_4 = robot.rotation_matrix_from_euler(0, 0, 0)
    print_orientation_info(target_rot_4, "target orientation")

    success = robot.moveL(
        target_position=current_pos,  # hold position
        target_rotation=target_rot_4,  # back to initial orientation
        duration=3.0,
        use_spline=True
    )

    if not success:
        print("Example 4 failed")
        return

    time.sleep(1)

    # ========== Example 5: continuous orientation change (cone motion) ==========
    print("\n" + "="*60)
    print("Example 5: continuous orientation change (cone motion)")
    print("The end-effector traces a cone around a fixed point")
    print("="*60)

    # Cone motion parameters
    cone_angle = np.pi / 6  # cone half-angle 30 deg
    num_steps = 8  # 8 orientations

    for i in range(num_steps):
        angle = 2 * np.pi * i / num_steps  # angle around the Z axis

        # Compute target orientation: tilt about Y, then orbit about Z
        rot_y = robot.rotation_matrix_from_euler(0, cone_angle, 0)
        rot_z = robot.rotation_matrix_from_euler(0, 0, angle)
        # Left-multiply (z-axis angle keeps changing); rotation is in the base frame.
        # Visually you see the EE sweep out a cone around the base-frame Z axis.
        target_rot = rot_z @ rot_y

        print(f"\nStep {i+1}/{num_steps}: angle = {np.rad2deg(angle):.1f} deg")

        success = robot.moveL(
            target_position=current_pos,
            target_rotation=target_rot,
            duration=1.5,
            use_spline=True
        )

        if not success:
            print(f"Step {i+1} failed")
            break

        time.sleep(0.5)

    print("\nCone motion completed!")
    time.sleep(1)

    # Return to zero pose
    print("\nReturning to zero pose...")
    robot.moveJ(zero_pos, duration=3.0, iswait=True)

    print("\n" + "="*60)
    print("Example program finished!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
