#!/usr/bin/env python3
"""
Septic polynomial interpolation control (with non-zero intermediate
velocities). Achieves continuous motion where intermediate waypoints have
non-zero velocity.
"""
import time
from Panthera_lib import Panthera

def precise_sleep(duration):
    """High-precision sleep helper."""
    if duration <= 0:
        return

    end_time = time.perf_counter() + duration

    if duration > 0.001:
        time.sleep(duration - 0.001)

    while time.perf_counter() < end_time:
        pass

def execute_continuous_trajectory(robot, waypoints, velocities, durations, control_rate=200):
    """
    Execute a continuous trajectory (velocity is not interrupted).
    waypoints: list of waypoints
    velocities: velocity at each waypoint (start and end velocities are 0)
    durations: duration of each segment
    """
    if len(waypoints) != len(velocities):
        print("waypoint count and velocity count must match")
        return False

    if len(waypoints) != len(durations) + 1:
        print("waypoint count should be one more than segment count")
        return False

    dt = 1.0 / control_rate

    for segment in range(len(durations)):
        start_pos = waypoints[segment]
        end_pos = waypoints[segment + 1]
        start_vel = velocities[segment]
        end_vel = velocities[segment + 1]
        duration = durations[segment]

        steps = int(duration * control_rate)

        # Record the start time of this segment
        segment_start = time.perf_counter()

        for step in range(steps):
            # Absolute target time for this step
            target_time = segment_start + (step + 1) * dt
            current_time = step * dt

            # Use the septic polynomial with velocity boundary conditions
            pos, vel, _ = robot.septic_interpolation_with_velocity(
                start_pos, end_pos, start_vel, end_vel, duration, current_time
            )

            # send control command
            robot.Joint_Pos_Vel(pos, vel, [10.0]*robot.motor_count)

            # High-precision wait
            wait_time = target_time - time.perf_counter()
            if wait_time > 0:
                precise_sleep(wait_time)

    # Lock at the final position
    final_pos = waypoints[-1]
    robot.Joint_Pos_Vel(final_pos, [0.0]*robot.motor_count, [10.0]*robot.motor_count)

    return True

def main():
    robot = Panthera()

    # Return to zero first
    zero_pos = [0.0] * robot.motor_count
    robot.Joint_Pos_Vel(zero_pos, [0.5]*6, [10.0]*6, iswait=True)
    time.sleep(1)

    # Define three key waypoints
    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # start (zero pose)
        [-0.26, 0.6, 0.60, 0.4, -0.3, -0.2],     # intermediate waypoint 1
        [0.2, 1.0, 1.2, 0.6, -0.5, 0.2]          # end point
    ]

    # Velocity at each waypoint (start and end are 0, intermediate is non-zero)
    velocities = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # start velocity = 0
        [0.2, 0.6, 0.6, 0.3, 0.4, 0.2],          # intermediate velocity != 0 (continuous motion)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]           # end velocity = 0
    ]

    # Duration of each segment
    durations = [1.0, 1.0]

    print("Executing continuous trajectory (no stop at intermediate waypoint)...")
    execute_continuous_trajectory(robot, waypoints, velocities, durations, control_rate=100)

    time.sleep(2)

    robot.Joint_Pos_Vel(zero_pos, [0.5]*6, [10.0]*6, iswait=True)
    time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # robot.set_stop()
        print("\nProgram interrupted")
    except Exception as e:
        print(f"\nerror: {e}")
