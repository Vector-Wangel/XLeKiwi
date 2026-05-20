#!/usr/bin/env python3
"""
Cubic polynomial interpolation trajectory control program.
Edit waypoints and durations to set the trajectory path.
"""
import time
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

def execute_trajectory(robot, waypoints, durations, control_rate=100):
    """Execute trajectory tracking using high-precision timing."""
    if len(waypoints) != len(durations) + 1:
        print("waypoint count should be one more than segment count")
        return False

    dt = 1.0 / control_rate

    for segment in range(len(durations)):
        start_pos = waypoints[segment]
        end_pos = waypoints[segment + 1]
        duration = durations[segment]

        steps = int(duration * control_rate)

        # Record segment start using absolute time to avoid drift
        segment_start = time.perf_counter()

        for step in range(steps):
            # Absolute target time for this step
            target_time = segment_start + (step + 1) * dt
            current_time = step * dt

            # Generate interpolated trajectory
            # Septic polynomial (continuous jerk, smoothest)
            pos, vel, _ = robot.septic_interpolation(start_pos, end_pos, duration, current_time)

            # send control command
            robot.Joint_Pos_Vel(pos, vel, [10.0]*robot.motor_count)

            # High-precision wait until the next control cycle
            wait_time = target_time - time.perf_counter()
            if wait_time > 0:
                precise_sleep(wait_time)

    # Arrive at the final position
    final_pos = waypoints[-1]
    robot.Joint_Pos_Vel(final_pos, [0.0]*robot.motor_count, [10.0]*robot.motor_count)

    return True

def main():
    robot = Panthera()

    # Return to zero first
    zero_pos = [0.0] * robot.motor_count
    robot.Joint_Pos_Vel(zero_pos, [0.5]*6, [10.0]*6, iswait=True)
    time.sleep(1)

    # Define trajectory waypoints (you can add more)
    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # start
        [-0.6, 0.7, 0.90, 0.2, -0.3, -0.2],
        [-0.4, 1.4, 1.8, 0.5, -0.7, 0.2],
        [-0.2, 0.8, 1.2, 0.7, 0.0, 0.4],
        [-0.4, 1.4, 1.8, 0.5, -0.7, 0.2],
        [-0.6, 0.7, 0.90, 0.2, -0.3, -0.2]# end
    ]

    # Duration of each segment (seconds)
    durations = [1.2, 1.0, 1.0, 1.0, 1.2]

    execute_trajectory(robot, waypoints, durations, control_rate=100)
    time.sleep(1)
    robot.Joint_Pos_Vel(zero_pos, [0.5]*6, [10.0]*6, iswait=True)
    time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # robot.set_stop()
        print("\nProgram interrupted")
