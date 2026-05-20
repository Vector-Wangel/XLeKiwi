#!/usr/bin/env python3
"""
Sinusoidal trajectory tracking control program.
Robot joints follow a sine-wave trajectory.
"""
import time
import numpy as np
from Panthera_lib import Panthera

def main():
    # Control parameters
    frequency = 0.2  # Hz, sine-wave frequency (tunable: 0.1-2.0 Hz). Higher frequency => higher speed.
    duration = 600.0  # motion duration (seconds)
    control_rate = 500  # control rate, Hz
    dt = 1.0 / control_rate
    max_torque = robot.max_torque.tolist()

    # Per-joint angle limits (radians)
    # Joint 1: ±180° = ±π, joints 2-3: 0 to -180° = 0 to -π, joints 4-5: ±90° = ±π/2, joint 6: ±180° = ±π
    joint_limits = [
        [-np.pi, np.pi],        # joint1: ±180 deg
        [0, np.pi],             # joint2: -180 to 0 deg
        [0, np.pi],             # joint3: -180 to 0 deg
        [0, np.pi/2],           # joint4: ±90 deg
        [-np.pi/2, np.pi/2],    # joint5: ±90 deg
        [-np.pi, np.pi]         # joint6: ±180 deg
    ]

    # Use the initial position as the center position (returns np.ndarray)
    print("Reading initial position...")
    center_pos = robot.get_current_pos()
    print(f"center position: {center_pos}")

    # Convert joint limits to numpy arrays for vectorized operations
    joint_limits_array = np.array(joint_limits)
    lower_limits = joint_limits_array[:, 0]
    upper_limits = joint_limits_array[:, 1]

    # Check whether the initial position is within limits
    for i, pos in enumerate(center_pos):
        if pos < lower_limits[i] or pos > upper_limits[i]:
            print(f"WARNING: joint{i+1} initial position {pos:.3f} is out of range [{lower_limits[i]:.3f}, {upper_limits[i]:.3f}]")

    # Per-joint amplitude (radians) - automatically clipped to avoid exceeding limits (vectorized)
    dist_to_upper = upper_limits - center_pos
    dist_to_lower = center_pos - lower_limits
    safe_amplitudes = np.minimum(dist_to_upper, dist_to_lower) * 0.8
    preset_amplitudes = np.array([0.4, 0.6, 0.6, 0.5, 0.4, 0.0])
    amplitudes = np.minimum(safe_amplitudes, preset_amplitudes)

    print(f"adjusted amplitudes: {amplitudes} rad")

    # Compute and print maximum velocity (vectorized)
    max_velocities = amplitudes * 2 * np.pi * frequency
    print(f"per-joint max velocity: {max_velocities} rad/s")

    # Per-joint phase offset (can desynchronize joints)
    # phase_offsets = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 0])  # phase offsets
    phase_offsets = np.zeros(robot.motor_count)  # zero phase offset

    print("\nStarting sinusoidal trajectory motion...")
    print(f"frequency: {frequency} Hz, duration: {duration} s")
    print(f"amplitudes: {amplitudes}")

    start_time = time.time()
    step = 0

    try:
        while (time.time() - start_time) < duration:
            loop_start = time.time()
            current_time = time.time() - start_time

            # Compute sinusoidal trajectory (vectorized)
            omega = 2 * np.pi * frequency

            # Position: x = x0 + A * sin(omega*t + phi) (vectorized)
            pos = center_pos + amplitudes * np.sin(omega * current_time + phase_offsets)

            # Velocity (derivative of position): v = A * omega * cos(omega*t + phi) (vectorized)
            vel = amplitudes * omega * np.cos(omega * current_time + phase_offsets)

            # Angle clipping (vectorized)
            # Clamp position to joint limits
            below_limit = pos < lower_limits
            above_limit = pos > upper_limits
            pos = np.clip(pos, lower_limits, upper_limits)
            # Zero out velocity when at a limit
            vel[below_limit | above_limit] = 0

            robot.Joint_Pos_Vel(pos, vel, max_torque, iswait=False)

            # Print status periodically
            if step % 50 == 0:  # every 0.5 s
                print(f"\rtime: {current_time:.2f}s | "
                      f"joint1 pos: {pos[0]:.3f} | "
                      f"joint2 pos: {pos[1]:.3f} | "
                      f"joint3 pos: {pos[2]:.3f}", end="")

            step += 1

            # Maintain control loop rate
            loop_time = time.time() - loop_start
            if loop_time < dt:
                time.sleep(dt - loop_time)

    except KeyboardInterrupt:
        print("\n\nTrajectory interrupted")

    # Return to center position
    print("\n\nReturning to center position...")
    robot.Joint_Pos_Vel(center_pos, [0.5] * robot.motor_count, [10.0] * robot.motor_count, iswait=True)

    print("Motion finished")
    time.sleep(1)

if __name__ == "__main__":
    robot = Panthera()

    # Move to a safe initial position first
    print("Moving to initial position...")
    zero_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # init_pos = [0.0, -1.0, -1.0, 0.0, 0.0, 0.0]
    init_pos = [-0.3, 1.1, 1.1, 0.2, -0.3, 0.0]
    vel = [0.5] * robot.motor_count
    max_torque = [10.0] * robot.motor_count

    success = robot.Joint_Pos_Vel(zero_pos, vel, max_torque, iswait=True)
    time.sleep(3)

    success = robot.Joint_Pos_Vel(init_pos, vel, max_torque, iswait=True)
    if success:
        print("Reached initial position")
        time.sleep(1)

    try:
        main()
        success = robot.Joint_Pos_Vel(zero_pos, vel, max_torque, iswait=True)
        time.sleep(2)
    except KeyboardInterrupt:
        # robot.set_stop()
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")
        # robot.set_stop()
