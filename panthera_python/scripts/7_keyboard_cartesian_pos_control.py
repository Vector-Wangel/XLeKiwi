#!/usr/bin/env python3
"""
Arm Cartesian space keyboard control

Features:
    1. On startup, the arm slowly moves to a safe position
    2. Use the keyboard to move the end-effector in Cartesian space
    3. Real-time display of the current end-effector position

Keyboard control:
    W/S: Move forward/backward along X axis
    A/D: Move left/right along Y axis
    Q: Move up along Z axis
    E: Move down along Z axis
    1/2: Rotate about X axis (+/-)
    3/4: Rotate about Y axis (+/-)
    5/6: Rotate about Z axis (+/-)
    ESC: Exit program

Notes:
    - Because inverse kinematics (IK) can have multiple or no solutions,
      joints may jump in certain poses
    - It is recommended to operate near the center of the workspace
    - The operator should stay clear of the arm's workspace
"""

import time
import numpy as np
from pynput import keyboard
from Panthera_lib import Panthera

# Global variables
target_position = np.array([0.24, 0.0, 0.15])  # Initial target position (m)
target_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # Initial orientation
position_delta = 0.005  # Position increment 3mm
rotation_delta = 0.03  # Rotation increment, about 1.7 degrees
running = True
position_changed = False  # Flag indicating whether the target changed

def rotation_matrix_x(angle):
    """Rotation matrix about the X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(angle):
    """Rotation matrix about the Y axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotation_matrix_z(angle):
    """Rotation matrix about the Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def on_press(key):
    """Keyboard press event handling"""
    global target_position, target_rotation, running, position_changed

    # Letter keys control position
    if hasattr(key, 'char') and key.char:
        if key.char == 'w' or key.char == 'W':
            target_position[0] += position_delta  # X axis forward
            position_changed = True
        elif key.char == 's' or key.char == 'S':
            target_position[0] -= position_delta  # X axis backward
            position_changed = True
        elif key.char == 'a' or key.char == 'A':
            target_position[1] += position_delta  # Y axis left
            position_changed = True
        elif key.char == 'd' or key.char == 'D':
            target_position[1] -= position_delta  # Y axis right
            position_changed = True
        elif key.char == 'q' or key.char == 'Q':
            target_position[2] += position_delta  # Z axis up
            position_changed = True
        elif key.char == 'e' or key.char == 'E':
            target_position[2] -= position_delta  # Z axis down
            position_changed = True
        # Number keys control rotation
        # (Right-multiply increment, Euler-angle convention: each rotation is
        # applied relative to the current rotated frame)
        elif key.char == '1':
            target_rotation = target_rotation @ rotation_matrix_x(rotation_delta)
            position_changed = True
        elif key.char == '2':
            target_rotation = target_rotation @ rotation_matrix_x(-rotation_delta)
            position_changed = True
        elif key.char == '3':
            target_rotation = target_rotation @ rotation_matrix_y(rotation_delta)
            position_changed = True
        elif key.char == '4':
            target_rotation = target_rotation @ rotation_matrix_y(-rotation_delta)
            position_changed = True
        elif key.char == '5':
            target_rotation = target_rotation @ rotation_matrix_z(rotation_delta)
            position_changed = True
        elif key.char == '6':
            target_rotation = target_rotation @ rotation_matrix_z(-rotation_delta)
            position_changed = True

        # (Left-multiply increment, fixed-angle convention: each rotation is
        # applied relative to the fixed world frame)
        # elif key.char == '1':
        #     target_rotation = rotation_matrix_x(rotation_delta) @ target_rotation
        #     position_changed = True
        # elif key.char == '2':
        #     target_rotation = rotation_matrix_x(-rotation_delta) @ target_rotation
        #     position_changed = True
        # elif key.char == '3':
        #     target_rotation = rotation_matrix_y(rotation_delta) @ target_rotation
        #     position_changed = True
        # elif key.char == '4':
        #     target_rotation = rotation_matrix_y(-rotation_delta) @ target_rotation
        #     position_changed = True
        # elif key.char == '5':
        #     target_rotation = rotation_matrix_z(rotation_delta) @ target_rotation
        #     position_changed = True
        # elif key.char == '6':
        #     target_rotation = rotation_matrix_z(-rotation_delta) @ target_rotation
        #     position_changed = True

def on_release(key):
    """Keyboard release event handling"""
    global running
    if key == keyboard.Key.esc:
        print("\nESC detected, preparing to exit...")
        running = False
        return False  # Stop listening

def move_to_safe_position(robot):
    """Slowly move to a safe position on startup"""
    print("\n" + "="*60)
    print("Moving to safe position...")
    print("="*60)
    # Define the safe position (joint space)
    safe_joint_pos = [0.0, 0.5, 0.6, 0.0, 0.0, 0.0]

    # Refresh state first
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    time.sleep(0.3)

    # Slowly move to the safe position (3 seconds)
    print("Moving...")
    success = robot.Joint_Pos_Vel(safe_joint_pos, [0.5]*robot.motor_count, iswait=True)

    if success:
        print("Reached safe position")
    else:
        print("Failed to move to safe position")
        return False

    time.sleep(0.5)
    return True

def main():
    global target_position, target_rotation, running, position_changed

    print("="*60)
    print("Arm Cartesian space keyboard control")
    print("="*60)

    # Initialize the arm
    print("\nInitializing arm...")
    robot = Panthera()

    # Move to safe position
    if not move_to_safe_position(robot):
        print("Initialization failed, exiting program")
        return

    # Refresh state to ensure we get the latest joint angles
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    time.sleep(0.1)

    # Use the current pose as the initial target
    current_fk = robot.forward_kinematics()
    target_position = np.array(current_fk['position'])
    target_rotation = np.array(current_fk['rotation'], dtype=float)  # Convert to a standard numpy array

    print(f"\nInitial position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] m")
    print("\n" + "="*60)
    print("Keyboard control:")
    print("  W/S: Move forward/backward along X axis")
    print("  A/D: Move left/right along Y axis")
    print("  Q: Move up along Z axis")
    print("  E: Move down along Z axis")
    print("  1/2: Rotate about X axis")
    print("  3/4: Rotate about Y axis")
    print("  5/6: Rotate about Z axis")
    print("  ESC: Exit program")
    print("="*60)
    print("\nControl started, please operate carefully!\n")

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Control loop
    last_valid_joint_pos = robot.get_current_pos()
    control_rate = 0.01  # 20Hz control frequency
    kp = [30.0, 50.0, 60.0, 25.0, 15.0, 10.0]        # Playback stiffness (tunable)
    kd = [3.0, 5.0, 6.0, 2.5, 1.5, 1.0]           # Playback damping

    # Send the current position once first to stabilize the arm
    robot_gra = robot.get_Gravity()
    robot_torque = np.array(robot_gra)
    robot.pos_vel_tqe_kp_kd(last_valid_joint_pos, [0.0]*robot.motor_count, robot_torque, kp, kd)
    time.sleep(0.2)

    try:
        while running:
            # Only recompute inverse kinematics (IK) when the target has changed
            if position_changed:
                # Compute inverse kinematics (IK)
                joint_pos = robot.inverse_kinematics(
                    target_position.tolist(),
                    target_rotation,
                    last_valid_joint_pos,
                    multi_init=False
                )

                if joint_pos is not None:
                    # Smoother behavior with MIT control + gravity feedforward
                    robot_gra = robot.get_Gravity()
                    robot_torque = np.array(robot_gra)
                    robot.pos_vel_tqe_kp_kd(joint_pos, [0.0]*robot.motor_count, robot_torque, kp, kd)

                    last_valid_joint_pos = joint_pos
                    position_changed = False  # Reset flag
                else:
                    # No IK solution, hold current position
                    print("\rinverse kinematics (IK) has no solution, holding current position", end='')
                    position_changed = False  # Reset flag

            # Read actual current position and display
            current_fk = robot.forward_kinematics()
            current_pos = current_fk['position']

            # Print the current position
            print(f"\rtarget position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] | "
                  f"current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]", end='')

            time.sleep(control_rate)

    except KeyboardInterrupt:
        print("\n\ninterrupted")
    finally:
        listener.stop()
        print("\n\nReturning to zero position...")
        zero_pos = [0.0] * robot.motor_count
        robot.Joint_Pos_Vel(zero_pos, [0.5]*robot.motor_count, iswait=True)
        print("all motors stopped")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nerror: {e}")
        import traceback
        traceback.print_exc()
