#!/usr/bin/env python3
"""
Basic motor control example

Demonstrates how to use the hightorque_robot library to perform basic motor control.
"""
import time
import math
import os
import sys
# Add the python directory to the path so we can import the module
# Go up two levels from motor_example to reach the panthera_python directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import hightorque_robot as htr


def basic_example():
    """Basic example: position control of a single motor"""
    print("=" * 60)
    print("Basic example: position control of a single motor")
    print("=" * 60)

    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../../robot_param/motor_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    # Get all motors
    motors = robot.get_motors()
    print(f"Detected {len(motors)} motors")

    if len(motors) == 0:
        print("No motors found. Please check the configuration and connection.")
        return

    # Control the first motor
    motor = motors[0]
    print(f"Controlling motor: {motor}")

    # Simple reciprocating motion
    positions = [0.5, -0.5, 0.0]
    for pos in positions:
        print(f"Moving to position: {pos} rad ({htr.rad_to_deg(pos):.1f} deg)")
        motor.position(pos)
        robot.motor_send_cmd()
        time.sleep(1.0)

        # Read motor state
        state = motor.get_current_motor_state()
        print(f"  Actual position: {state.position:.3f} rad, "
              f"velocity: {state.velocity:.3f} rad/s, "
              f"torque: {state.torque:.3f} Nm")

    # Stop the motor
    robot.set_stop()
    print("Motor stopped")


def multi_motor_example():
    """Multi-motor control example"""
    print("\n" + "=" * 60)
    print("Multi-motor control example")
    print("=" * 60)

    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    motors = robot.get_motors()

    if len(motors) < 2:
        print("This example requires at least 2 motors")
        return

    print(f"Controlling {len(motors)} motors")

    # Drive all motors to different positions simultaneously
    for i, motor in enumerate(motors):
        angle = 0.5 if i % 2 == 0 else -0.5
        motor.position(angle)

    robot.motor_send_cmd()
    time.sleep(2.0)

    # Print state for all motors
    htr.print_motor_states(motors)

    robot.set_stop()


def sinusoidal_motion_example():
    """Sinusoidal motion example"""
    print("\n" + "=" * 60)
    print("Sinusoidal motion example")
    print("=" * 60)

    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    motors = robot.get_motors()

    if len(motors) == 0:
        return

    motor = motors[0]
    print(f"Motor {motor.get_motor_id()} performing sinusoidal motion")

    # Create a sinusoidal trajectory generator
    trajectory = htr.create_sinusoidal_trajectory(
        amplitude=1.0,      # Amplitude 1 rad
        frequency=0.5,      # Frequency 0.5 Hz
        offset=0.0          # No offset
    )

    # Run for 5 seconds
    start_time = time.time()
    duration = 5.0

    try:
        while time.time() - start_time < duration:
            t = time.time() - start_time
            target_pos = trajectory(t)

            motor.position(target_pos)
            robot.motor_send_cmd()

            # Print status every 0.5 seconds
            if int(t * 2) != int((t - 0.001) * 2):
                state = motor.get_current_motor_state()
                print(f"t={t:.2f}s: target={target_pos:.3f}, "
                      f"actual={state.position:.3f}, "
                      f"velocity={state.velocity:.3f}")

            time.sleep(0.001)  # 1ms control period

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    robot.set_stop()
    print("Motion complete")


def advanced_control_example():
    """Advanced control example: hybrid control modes"""
    print("\n" + "=" * 60)
    print("Advanced control example: position + velocity + torque control")
    print("=" * 60)

    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    motors = robot.get_motors()

    if len(motors) == 0:
        return

    motor = motors[0]

    # Use position + velocity + max torque control
    print("Using motor.pos_vel_MAXtqe control mode (low-level motor method)")

    target_positions = [
        (1.0, 0.5, 10.0),   # Position 1 rad, velocity 0.5, max torque 10 Nm
        (-1.0, 0.5, 10.0),
        (0.0, 0.2, 5.0),
    ]

    for pos, vel, tqe_max in target_positions:
        print(f"Target: position={pos:.2f}, velocity={vel:.2f}, max torque={tqe_max:.2f}")
        motor.pos_vel_MAXtqe(pos, vel, tqe_max)
        robot.motor_send_cmd()
        time.sleep(2.0)

        state = motor.get_current_motor_state()
        print(f"  Reached: position={state.position:.3f}, "
              f"velocity={state.velocity:.3f}, "
              f"torque={state.torque:.3f}\n")

    # Use five-parameter control
    print("Using pos_vel_tqe_kp_kd five-parameter control mode")
    motor.pos_vel_tqe_kp_kd(
        position=0.5,
        velocity=0.1,
        torque=0.0,    # Feed-forward torque
        kp=50.0,       # PID proportional gain
        kd=5.0         # PID derivative gain
    )
    robot.motor_send_cmd()
    time.sleep(2.0)

    robot.set_stop()


def motor_info_example():
    """Motor information query example"""
    print("\n" + "=" * 60)
    print("Motor information query example")
    print("=" * 60)

    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    # Print robot info
    print(f"Robot: {robot}")
    print(f"Robot name: {robot.robot_params.robot_name}")
    print(f"Timeout setting: {robot.motor_timeout_ms} ms")
    print(f"Number of CAN boards: {robot.robot_params.CANboard_num}")

    motors = robot.get_motors()
    print(f"\nTotal motors: {len(motors)}")
    print("-" * 60)

    for motor in motors:
        print(f"\nMotor {motor.get_motor_id()}:")
        print(f"  Name: {motor.get_motor_name()}")
        print(f"  Type: {motor.get_motor_enum_type()}")

        # Get version info
        version = motor.get_version()
        print(f"  Version: v{version.major}.{version.minor}.{version.patch}")

        # Get current state
        state = motor.get_current_motor_state()
        print(f"  Position: {state.position:.4f} rad ({htr.rad_to_deg(state.position):.2f} deg)")
        print(f"  Velocity: {state.velocity:.4f} rad/s")
        print(f"  Torque: {state.torque:.4f} Nm")
        print(f"  Mode: {state.mode}")
        print(f"  Fault code: 0x{state.fault:02X}")


def safety_features_example():
    """Safety features example"""
    print("\n" + "=" * 60)
    print("Safety features example")
    print("=" * 60)

    # Create the robot instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param", "robot_config.yaml")
    robot = htr.Robot(config_path)

    motors = robot.get_motors()

    if len(motors) == 0:
        return

    motor = motors[0]

    # Set timeout
    print("Setting timeout: 100 ms")
    robot.set_timeout(100)

    # Test control
    print("\nTesting motor control...")
    motor.position(1.0)
    robot.motor_send_cmd()
    time.sleep(1.0)

    robot.set_stop()

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("High-torque robot motor control - Python example program")
    print("=" * 60)

    examples = [
        ("1", "Basic example", basic_example),
        ("2", "Multi-motor control", multi_motor_example),
        ("3", "Sinusoidal motion", sinusoidal_motion_example),
        ("4", "Advanced control", advanced_control_example),
        ("5", "Motor info query", motor_info_example),
        ("6", "Safety features", safety_features_example),
        ("0", "Run all examples", None),
    ]

    print("\nAvailable examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")

    choice = input("\nPlease select an example (0-6): ").strip()

    if choice == "0":
        # Run all examples
        for num, name, func in examples:
            if func is not None:
                try:
                    func()
                except Exception as e:
                    print(f"\nExample '{name}' failed: {e}")
                    import traceback
                    traceback.print_exc()
                input("\nPress Enter to continue to the next example...")
    else:
        # Run the selected example
        for num, name, func in examples:
            if num == choice and func is not None:
                try:
                    func()
                except Exception as e:
                    print(f"\nExample failed: {e}")
                    import traceback
                    traceback.print_exc()
                break
        else:
            print("Invalid selection")

    print("\n" + "=" * 60)
    print("End of example program")
    print("=" * 60)


if __name__ == "__main__":
    main()
