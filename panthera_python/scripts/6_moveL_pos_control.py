#!/usr/bin/env python3
"""
moveL position control example.

Demonstrates how to use moveL for smooth end-effector motion.
"""

import sys
import os
import numpy as np
import time
from Panthera_lib import Panthera


def main():
    print("="*60)
    print("MoveIt-style Cartesian control example")
    print("="*60)

    # 1. Initialize the arm
    print("\nInitializing the arm...")
    robot = Panthera()

    # 2. Move to initial position
    print("\nMoving to initial position...")
    ik_pos1 = [0.24, 0.0, 0.1]
    ik_rot1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    zero_pos = [0.0] * robot.motor_count
    vel = [0.5] * robot.motor_count
    robot.Joint_Pos_Vel(zero_pos, vel, iswait=True)

    pos1 = robot.inverse_kinematics(ik_pos1, ik_rot1, robot.get_current_pos())
    if pos1 is not None:
        robot.moveJ(pos1, duration=3.0, iswait=True)

    # 3. Get current pose
    current_fk = robot.forward_kinematics()
    current_pos = np.array(current_fk['position'])
    current_rot = current_fk['rotation']

    print(f"\ncurrent position: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}] m")

    # ========== Example 1: linear motion (specified duration) ==========
    print("\n" + "="*60)
    print("Example 1: move 10 cm along X axis (duration 2 s)")
    print("="*60)

    target_pos_1 = current_pos + np.array([0.1, 0.0, 0.0])

    success = robot.moveL(
        target_position=target_pos_1,
        target_rotation=current_rot,
        duration=2.0,  # 2 s
        use_spline=True  # use spline smoothing
    )

    if not success:
        print("Example 1 failed")
        return

    time.sleep(1)

    # ========== Example 2: linear motion (specified duration) ==========
    print("\n" + "="*60)
    print("Example 2: move 8 cm along Y axis (duration 1.5 s)")
    print("="*60)

    current_fk = robot.forward_kinematics()
    current_pos = np.array(current_fk['position'])
    current_rot = current_fk['rotation']

    target_pos_2 = current_pos + np.array([0.0, 0.08, 0.0])

    success = robot.moveL(
        target_position=target_pos_2,
        target_rotation=current_rot,
        duration=1.5,  # 1.5 s
        use_spline=True
    )

    if not success:
        print("Example 2 failed")
        return

    time.sleep(0.5)

    # ========== Example 3: diagonal motion ==========
    print("\n" + "="*60)
    print("Example 3: diagonal motion (X:-4 cm, Y:-4 cm, Z:+4 cm)")
    print("="*60)

    current_fk = robot.forward_kinematics()
    current_pos = np.array(current_fk['position'])
    current_rot = current_fk['rotation']

    target_pos_3 = current_pos + np.array([-0.04, -0.04, 0.04])

    success = robot.moveL(
        target_position=target_pos_3,
        target_rotation=current_rot,
        duration=2.0,  # 2 s
        use_spline=True
    )

    if not success:
        print("Example 3 failed")
        return

    time.sleep(0.5)

    # ========== Example 4: multi-segment path (MoveIt-style waypoints) ==========
    print("\n" + "="*60)
    print("Example 4: multi-segment path (square)")
    print("="*60)

    current_fk = robot.forward_kinematics()
    start_pos = np.array(current_fk['position'])
    start_rot = current_fk['rotation']

    # Define the 4 corners of the square
    side_length = 0.07  # 7 cm
    waypoints = [
        {'position': start_pos, 'rotation': start_rot},
        {'position': start_pos - np.array([side_length, 0, 0]), 'rotation': start_rot},
        {'position': start_pos - np.array([side_length, side_length, 0]), 'rotation': start_rot},
        {'position': start_pos - np.array([0, side_length, 0]), 'rotation': start_rot},
        {'position': start_pos, 'rotation': start_rot},  # back to start
    ]

    # Compute the full path
    print("  Computing square path...")
    joint_trajectory, fraction = robot.compute_cartesian_path(waypoints)

    if joint_trajectory is None or fraction < 0.99:
        print(f"  Path planning failed or incomplete (fraction={fraction*100:.1f}%)")
        return

    print(f"  OK - path planned: {len(joint_trajectory)} points")

    # Time parameterization
    timestamps = robot.compute_time_parameterization(joint_trajectory, duration=3.0)
    print(f"  OK - total time: {timestamps[-1]:.2f}s")

    # Spline smoothing
    joint_trajectory, timestamps, velocities = robot.smooth_trajectory_spline(
        joint_trajectory, timestamps
    )
    print(f"  OK - spline smoothing done: {len(joint_trajectory)} points")

    # Execute
    print("  Executing...")
    success = robot._execute_trajectory(
        joint_trajectory, timestamps, velocities, max_tqu=None
    )

    if success:
        print("  OK - square trajectory executed successfully")
    else:
        print("  FAIL - square trajectory execution failed")

    robot.moveJ(zero_pos, 3.0, iswait=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
