#!/usr/bin/env python3
"""
Arm Cartesian space velocity control (Jacobian-based)

Features:
    1. On startup, the arm slowly moves to a safe position
    2. Use the keyboard to control end-effector velocity (continuous motion while held)
    3. Use a damped pseudo-inverse to avoid Jacobian singularities
    4. Real-time display of current end-effector position and velocity

Keyboard control:
    W/S: Linear velocity along tool-frame X axis (+/-)
    A/D: Linear velocity along tool-frame Y axis (+/-)
    Q/E: Linear velocity along tool-frame Z axis (+/-)
    1/2: Angular velocity about tool-frame X axis (+/-)
    3/4: Angular velocity about tool-frame Y axis (+/-)
    5/6: Angular velocity about tool-frame Z axis (+/-)
    ESC: Exit program

Notes:
    - The end-effector keeps moving while the key is held, and stops on release
    - Damped pseudo-inverse avoids singularities for smoother motion
    - Automatically slows down when approaching workspace boundaries
"""

import time
import numpy as np
from pynput import keyboard
from Panthera_lib import Panthera

# Global variables
end_velocity = np.zeros(6)  # Target velocity [vx, vy, vz, wx, wy, wz]
actual_velocity = np.zeros(6)  # Actual velocity (after smoothing)
linear_speed = 0.3  # Linear speed m/s
angular_speed = 2.0  # Angular speed rad/s
running = True
keys_pressed = set()  # Track currently pressed keys

# Acceleration parameters
acceleration_factor = 0.02  # Acceleration factor (0-1, larger means faster ramp up)
                            # 0.15 reaches target velocity in about 0.2-0.3 s

def on_press(key):
    """Keyboard press event handling"""
    global end_velocity, running, keys_pressed

    # Letter keys control linear velocity
    if hasattr(key, 'char') and key.char:
        keys_pressed.add(key.char.lower())

        if key.char.lower() == 'w':
            end_velocity[0] = linear_speed  # X axis forward
        elif key.char.lower() == 's':
            end_velocity[0] = -linear_speed  # X axis backward
        elif key.char.lower() == 'a':
            end_velocity[1] = linear_speed  # Y axis left
        elif key.char.lower() == 'd':
            end_velocity[1] = -linear_speed  # Y axis right
        elif key.char.lower() == 'q':
            end_velocity[2] = linear_speed  # Z axis up
        elif key.char.lower() == 'e':
            end_velocity[2] = -linear_speed  # Z axis down
        # Number keys control angular velocity
        elif key.char == '1':
            end_velocity[3] = angular_speed  # About X axis
        elif key.char == '2':
            end_velocity[3] = -angular_speed
        elif key.char == '3':
            end_velocity[4] = angular_speed  # About Y axis
        elif key.char == '4':
            end_velocity[4] = -angular_speed
        elif key.char == '5':
            end_velocity[5] = angular_speed  # About Z axis
        elif key.char == '6':
            end_velocity[5] = -angular_speed

def on_release(key):
    """Keyboard release event handling"""
    global end_velocity, running, keys_pressed

    if key == keyboard.Key.esc:
        print("\nESC detected, preparing to exit...")
        running = False
        return False  # Stop listening

    # Clear the corresponding velocity when the letter key is released
    if hasattr(key, 'char') and key.char:
        char = key.char.lower()
        if char in keys_pressed:
            keys_pressed.remove(char)

        if char in ['w', 's']:
            end_velocity[0] = 0.0
        elif char in ['a', 'd']:
            end_velocity[1] = 0.0
        elif char in ['q', 'e']:
            end_velocity[2] = 0.0
        elif char in ['1', '2']:
            end_velocity[3] = 0.0
        elif char in ['3', '4']:
            end_velocity[4] = 0.0
        elif char in ['5', '6']:
            end_velocity[5] = 0.0

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
    global end_velocity, running

    print("="*60)
    print("Arm Cartesian space velocity control (Jacobian-based)")
    print("="*60)

    # Initialize the arm
    print("\nInitializing arm...")
    robot = Panthera()

    # Check whether the Pinocchio model is available
    if robot.model is None:
        print("Error: Pinocchio model not found, cannot compute Jacobian")
        return

    # Move to safe position
    if not move_to_safe_position(robot):
        print("Initialization failed, exiting program")
        return

    # Get current pose
    current_fk = robot.forward_kinematics()
    current_pos = current_fk['position']

    print(f"\nInitial position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] m")
    print("\n" + "="*60)
    print("Keyboard control (continuous motion while held):")
    print("  W/S: Linear velocity along tool-frame X axis (+/-)")
    print("  A/D: Linear velocity along tool-frame Y axis (+/-)")
    print("  Q/E: Linear velocity along tool-frame Z axis (+/-)")
    print("  1/2: Angular velocity about tool-frame X axis (+/-)")
    print("  3/4: Angular velocity about tool-frame Y axis (+/-)")
    print("  5/6: Angular velocity about tool-frame Z axis (+/-)")
    print("  ESC: Exit program")
    print("="*60)
    print("\nControl started, please operate carefully!\n")

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Control parameters
    control_rate = 0.01  # 100Hz control frequency
    damping_base = 0.01  # Base damping coefficient
    damping_adaptive = True  # Adaptive damping

    # MIT control parameters
    kp = [0.0] * robot.motor_count  # Position gain set to 0 (pure velocity control)
    kd = [10.0, 15.0, 15.0, 10.0, 5.0, 5.0]  # Damping gains

    # Joint velocity limits
    max_joint_vel = np.array([2.0, 2.0, 2.0, 2.0, 3.0, 3.0])  # rad/s

    try:
        while running:
            # Get current joint angles
            q = robot.get_current_pos()

            # Compute Jacobian
            try:
                J = robot.get_jacobian(q)
            except Exception as e:
                print(f"\rJacobian computation failed: {e}                    ", end='')
                time.sleep(control_rate)
                continue

            # Compute manipulability
            manipulability = robot.get_manipulability(q)

            # Adaptive damping: increase damping near singularities
            if damping_adaptive:
                # Manipulability threshold
                manip_threshold = 0.01
                if manipulability < manip_threshold:
                    # Near singularity, increase damping and slow down
                    damping = damping_base * (1.0 + (manip_threshold - manipulability) * 100)
                    speed_scale = manipulability / manip_threshold
                    end_velocity_scaled = end_velocity * speed_scale
                else:
                    damping = damping_base
                    end_velocity_scaled = end_velocity
            else:
                damping = damping_base
                end_velocity_scaled = end_velocity

            # Compute damped pseudo-inverse
            J_damp = Panthera.compute_damped_pseudoinverse(J, damping)

            # === Velocity smoothing (acceleration/deceleration) ===
            # Use exponential smoothing so the actual velocity gradually approaches the target
            global actual_velocity
            actual_velocity = actual_velocity + (end_velocity_scaled - actual_velocity) * acceleration_factor

            # Use the smoothed velocity
            end_velocity_to_use = actual_velocity

            # Get current end-effector position and orientation
            current_fk = robot.forward_kinematics()
            current_pos = current_fk['position']
            R_tool = current_fk['rotation']  # Rotation matrix of the tool frame

            # Transform linear and angular velocity from tool frame to world frame
            # v_world = R_tool @ v_tool
            # w_world = R_tool @ w_tool
            end_velocity_world = np.zeros(6)
            end_velocity_world[0:3] = R_tool @ end_velocity_to_use[0:3]  # Linear velocity transform
            end_velocity_world[3:6] = R_tool @ end_velocity_to_use[3:6]  # Angular velocity transform

            # Compute joint velocities
            q_dot = J_damp @ end_velocity_world

            # Clip joint velocities
            q_dot = np.clip(q_dot, -max_joint_vel, max_joint_vel)

            robot.Joint_Vel(q_dot)

            # Display info
            vel_norm = np.linalg.norm(actual_velocity[:3])  # Use the actual velocity
            print(f"\rPosition: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] | "
                  f"Velocity: {vel_norm:.3f} m/s | Manipulability: {manipulability:.4f} | Damping: {damping:.4f}", end='')

            time.sleep(control_rate)

    except KeyboardInterrupt:
        print("\n\ninterrupted")
    finally:
        listener.stop()
        # Stop motion
        print("\n\nStopping motion...")
        q = robot.get_current_pos()
        gra = robot.get_Gravity()
        robot.pos_vel_tqe_kp_kd(q, [0.0]*robot.motor_count, gra, kp, kd)
        time.sleep(0.5)

        print("Returning to zero position...")
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
