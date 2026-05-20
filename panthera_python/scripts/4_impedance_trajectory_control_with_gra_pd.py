#!/usr/bin/env python3
"""
Trajectory-based impedance control program.
Uses septic polynomial interpolation + gravity-compensation feed-forward
impedance control.
In the full impedance equation, set the desired inertia matrix Md = M(q),
and under low-speed conditions set q_des_dot = q_des_ddot = 0,
which gives t = g(q) + tor_impedance.
"""
import time
import numpy as np
from Panthera_lib import Panthera


def precise_sleep(duration):
    """High-precision sleep helper."""
    if duration <= 0:
        return

    end_time = time.perf_counter() + duration

    # Sleep for most of the time (leave 1 ms margin)
    if duration > 0.001:
        time.sleep(duration - 0.001)

    # Busy-wait for the last bit to keep timing accurate
    while time.perf_counter() < end_time:
        pass

def execute_impedance_trajectory(robot, waypoints, durations, K, B, control_rate=200):
    """Execute trajectory-based impedance control."""
    if len(waypoints) != len(durations) + 1:
        print("waypoint count should be one more than segment count")
        return False

    dt = 1.0 / control_rate
    zero_kp = [0.0] * robot.motor_count
    zero_kd = [0.0] * robot.motor_count
    zero_pos = [0.0] * robot.motor_count
    zero_vel = [0.0] * robot.motor_count

    for segment in range(len(durations)):
        start_pos = waypoints[segment]
        end_pos = waypoints[segment + 1]
        duration = durations[segment]

        steps = int(duration * control_rate)
        segment_start = time.perf_counter()

        for step in range(steps):
            target_time = segment_start + (step + 1) * dt
            current_time = step * dt

            # Generate the desired trajectory
            pos_des, vel_des, _ = robot.septic_interpolation(start_pos, end_pos, duration, current_time)

            # Get current state
            states = robot.get_current_state()
            q_current = np.array([state.position for state in states])
            vel_current = np.array([state.velocity for state in states])

            # Impedance control
            tor_impedance = K * (np.array(pos_des) - q_current) + B * (np.array(vel_des) - vel_current)

            # Gravity compensation
            G = np.array(robot.get_Gravity(q_current))

            # Total torque
            tor = tor_impedance + G

            # Torque clipping
            tau_limit = np.array([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])
            tor = np.clip(tor, -tau_limit, tau_limit)
            print(tor)

            # send control command
            robot.pos_vel_tqe_kp_kd(zero_pos, zero_vel, tor, zero_kp, zero_kd)

            # High-precision wait
            wait_time = target_time - time.perf_counter()
            if wait_time > 0:
                precise_sleep(wait_time)

    return True

def main():
    # Define trajectory waypoints
    waypoints1 = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.9, 0.9, -0.10, 0.0, 0.0]
    ]

    waypoints2 = [
        [0.5, 0.9, 0.9, -0.10, 0.0, 0.0],
        [0.0, 1.4, 1.1, -0.10, 0.0, 0.0]
    ]

    waypoints3 = [
        [0.0, 1.4, 1.1, -0.10, 0.0, 0.0],
        [0.0, 0.9, 0.9, -0.10, 0.0, 0.0]
    ]

    # Duration of each segment
    durations1 = [3.0]

    execute_impedance_trajectory(robot, waypoints1, durations1, K, B, control_rate=200)
    execute_impedance_trajectory(robot, waypoints2, durations1, K, B, control_rate=200)
    execute_impedance_trajectory(robot, waypoints3, durations1, K, B, control_rate=200)

if __name__ == "__main__":
    try:
        robot = Panthera()
        # Impedance control parameters
        K = np.array([5.0, 10.0, 15.0, 6.0, 5.0, 5.0])
        B = np.array([0.5, 1.0, 1.50, 0.6, 0.5, 0.5])

        main()

        # Hold-position impedance control
        print("Starting hold-position impedance control...")
        final_pos = np.array([0.0, 0.9, 0.9, -0.10, 0.0, 0.0])
        zero_kp = [0.0] * robot.motor_count
        zero_kd = [0.0] * robot.motor_count
        zero_pos = [0.0] * robot.motor_count
        zero_vel = [0.0] * robot.motor_count

        while(1):
            # Get current state
            states = robot.get_current_state()
            q_current = np.array([state.position for state in states])
            vel_current = np.array([state.velocity for state in states])

            # Hold-position impedance control
            tor_impedance = K * (final_pos - q_current) + B * (np.zeros(6) - vel_current)

            # Gravity compensation
            G = np.array(robot.get_Gravity(q_current))

            # Total torque
            tor = tor_impedance + G

            # Torque clipping
            tau_limit = np.array([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])
            tor = np.clip(tor, -tau_limit, tau_limit)

            # send control command
            robot.pos_vel_tqe_kp_kd(zero_pos, zero_vel, tor, zero_kp, zero_kd)
            print(tor)

            time.sleep(0.002)  # 500 Hz control rate

    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"\nerror: {e}")
