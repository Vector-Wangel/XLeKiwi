import time
import sys
import os
import yaml
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

try:
    import hightorque_robot as htr
except ImportError as e:
    print(f"Failed to import hightorque_robot: {e}")
    print("Please make sure the hightorque_robot whl package is installed")
    print("Installation: pip install hightorque_robot-*.whl")
    sys.exit(1)

#######################
# Panthera arm control class
#######################
class Panthera(htr.Robot):  # Inherits from htr.Robot
    #######################
    # Initialization
    #######################
    def __init__(self, config_path=None):
        """
        Initialize the Panthera arm

        Args:
            config_path: Path to the configuration file. If None, the default
                path is used.
        """
        # Determine the configuration file path
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(
                os.path.join(script_dir, "..", "..", "robot_param", "Follower.yaml")
            )

        # Initialize member variables
        self._init_member_variables()

        # Load the configuration file
        self._load_config_file(config_path)

        # Save the directory of the configuration file
        self.config_dir = os.path.dirname(os.path.abspath(config_path))

        # Load base configuration parameters (independent of motor count)
        self._load_joint_limits()
        self._load_gripper_limits()

        # Initialize the parent class and the motors
        super().__init__(config_path)
        self.Motors = self.get_motors()
        self._init_motors()

        # Load motor-related parameters (depend on motor count)
        self._load_motor_parameters()
        self._load_moveit_parameters()
        self._load_friction_parameters()

        # Load the URDF model
        self._load_urdf_model()

    def _init_member_variables(self):
        """Initialize member variables"""
        self.config = None
        self.model = None
        self.data = None
        self.joint_names = None
        self.joint_ids = []
        self.joint_limits = None
        self.gripper_limits = None
        self.end_effector_frame_id = None

    def _load_config_file(self, config_path):
        """
        Load the YAML configuration file

        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                print(f"Configuration loaded successfully: {config_path}")
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            sys.exit(1)

    def _load_joint_limits(self):
        """Load joint limits from the configuration file"""
        try:
            if 'robot' in self.config and 'joint_limits' in self.config['robot']:
                self.joint_limits = {
                    'lower': np.array(self.config['robot']['joint_limits']['lower']),
                    'upper': np.array(self.config['robot']['joint_limits']['upper'])
                }
                print(f"Joint limits loaded: lower={self.joint_limits['lower']}, upper={self.joint_limits['upper']}")
            else:
                print("Warning: joint_limits not found in the configuration file")
        except Exception as e:
            print(f"Failed to load joint limits: {e}")

    def _load_gripper_limits(self):
        """Load gripper limits from the configuration file"""
        try:
            if 'robot' in self.config and 'gripper_limits' in self.config['robot']:
                self.gripper_limits = {
                    'lower': self.config['robot']['gripper_limits']['lower'],
                    'upper': self.config['robot']['gripper_limits']['upper']
                }
                print(f"Gripper limits loaded: lower={self.gripper_limits['lower']}, upper={self.gripper_limits['upper']}")
            else:
                print("Warning: gripper_limits not found in the configuration file")
        except Exception as e:
            print(f"Failed to load gripper limits: {e}")

    def _init_motors(self):
        """Initialize motors and print motor information"""
        self.gripper_id = len(self.Motors)
        self.motor_count = len(self.Motors) - 1

        print("Initializing arm...")
        print(f"Found {self.motor_count} motors")

        if self.motor_count == 0:
            print("No motors found. Please check your configuration and connection.")
            return

        # Print motor information
        for i, motor in enumerate(self.Motors):
            print(f"Motor {i}: ID={motor.get_motor_id()}, "
                  f"Type={motor.get_motor_enum_type()}, "
                  f"Name={motor.get_motor_name()}")

    def _load_motor_parameters(self):
        """Load motor-related parameters from the config file (max torque, velocity limits, acceleration limits)"""
        # Load max torque
        if 'robot' not in self.config or 'max_torque' not in self.config['robot']:
            print("Error: missing robot.max_torque parameter in the configuration file")
            sys.exit(1)

        self.max_torque = np.array(self.config['robot']['max_torque'])
        if len(self.max_torque) != self.motor_count:
            print(f"Error: max_torque length ({len(self.max_torque)}) does not match motor count ({self.motor_count})")
            sys.exit(1)
        print(f"Max torque loaded: {self.max_torque.tolist()}")

        # Load velocity limits
        if 'robot' not in self.config or 'velocity_limits' not in self.config['robot']:
            print("Error: missing robot.velocity_limits parameter in the configuration file")
            sys.exit(1)

        self.velocity_limits = np.array(self.config['robot']['velocity_limits'])
        if len(self.velocity_limits) != self.motor_count:
            print(f"Error: velocity_limits length ({len(self.velocity_limits)}) does not match motor count ({self.motor_count})")
            sys.exit(1)
        print(f"Velocity limits loaded: {self.velocity_limits.tolist()}")

        # Load acceleration limits
        if 'robot' not in self.config or 'acceleration_limits' not in self.config['robot']:
            print("Error: missing robot.acceleration_limits parameter in the configuration file")
            sys.exit(1)

        self.acceleration_limits = np.array(self.config['robot']['acceleration_limits'])
        if len(self.acceleration_limits) != self.motor_count:
            print(f"Error: acceleration_limits length ({len(self.acceleration_limits)}) does not match motor count ({self.motor_count})")
            sys.exit(1)
        print(f"Acceleration limits loaded: {self.acceleration_limits.tolist()}")

    def _load_friction_parameters(self):
        """Load Coulomb/viscous friction compensation parameters (Fc, Fv, vel_threshold)
        from the 'friction' section of the yaml. Falls back to 0 (no friction compensation)
        when missing. Scripts should use self.Fc / self.Fv / self.vel_threshold directly."""
        f = self.config.get('friction', {})
        if 'Fc' in f and 'Fv' in f:
            self.Fc = np.array(f['Fc'], dtype=float)
            self.Fv = np.array(f['Fv'], dtype=float)
            self.vel_threshold = float(f.get('vel_threshold', 0.02))
            if len(self.Fc) != self.motor_count or len(self.Fv) != self.motor_count:
                print(f"Warning: friction Fc/Fv length does not match motor count ({self.motor_count})")
            print(f"Friction compensation parameters loaded: Fc={self.Fc.tolist()}, Fv={self.Fv.tolist()}, vel_threshold={self.vel_threshold}")
        else:
            self.Fc = np.zeros(self.motor_count)
            self.Fv = np.zeros(self.motor_count)
            self.vel_threshold = 0.02
            print("Note: configuration does not provide a friction section. Fc/Fv default to 0")

    def _load_moveit_parameters(self):
        """Load MoveIt Cartesian controller parameters from the configuration file"""
        if 'moveit_cartesian' not in self.config:
            print("Error: missing moveit_cartesian parameters in the configuration file")
            sys.exit(1)

        moveit_config = self.config['moveit_cartesian']

        # Load eef_step
        if 'eef_step' not in moveit_config:
            print("Error: missing moveit_cartesian.eef_step parameter in the configuration file")
            sys.exit(1)
        self.eef_step = moveit_config['eef_step']

        # Load jump_threshold
        if 'jump_threshold' not in moveit_config:
            print("Error: missing moveit_cartesian.jump_threshold parameter in the configuration file")
            sys.exit(1)
        self.jump_threshold = moveit_config['jump_threshold']

        # Load resample_dt
        if 'resample_dt' not in moveit_config:
            print("Error: missing moveit_cartesian.resample_dt parameter in the configuration file")
            sys.exit(1)
        self.resample_dt = moveit_config['resample_dt']

        print(f"MoveIt Cartesian parameters loaded: eef_step={self.eef_step}m, "
              f"jump_threshold={self.jump_threshold}rad, resample_dt={self.resample_dt}s")

    def _load_urdf_model(self):
        """Load the URDF model for kinematic computations"""
        try:
            # Get the URDF file path (relative to the configuration file)
            urdf_relative_path = self.config['urdf']['file_path']

            # Compute the absolute URDF path (relative to the directory of the config file)
            config_dir = getattr(self, "config_dir", os.path.dirname(os.path.abspath(__file__)))
            urdf_path = os.path.normpath(os.path.join(config_dir, urdf_relative_path))

            # Load the URDF with pinocchio
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

            # Get joint information
            self.joint_names = self.config['kinematics']['joint_names']

            # Get joint IDs (skip the universe joint)
            for joint_name in self.joint_names:
                if self.model.existJointName(joint_name):
                    joint_id = self.model.getJointId(joint_name)
                    self.joint_ids.append(joint_id)
                else:
                    print(f"Warning: joint {joint_name} not found in the model")

            print(f"URDF loaded successfully: {urdf_path}")
            print(f"The model contains {self.model.njoints - 1} joints (excluding base)")
            print(f"Configured joints: {len(self.joint_ids)}")

            # Get the end-effector frame ID
            end_effector_link = self.config['urdf']['end_effector_link']
            if self.model.existFrame(end_effector_link):
                self.end_effector_frame_id = self.model.getFrameId(end_effector_link)
                print(f"End-effector frame: {end_effector_link} (ID: {self.end_effector_frame_id})")
            else:
                print(f"Warning: end-effector frame '{end_effector_link}' not found. Falling back to the last joint.")
                self.end_effector_frame_id = self.model.getFrameId(self.joint_names[-1])
        except Exception as e:
            print(f"Failed to load URDF: {e}")

    #######################
    # State accessors
    #######################
    def get_current_state(self):
        """Get the current joint states"""
        state = []
        for i in range(self.motor_count):
            motor_state = self.Motors[i].get_current_motor_state()
            state.append(motor_state)
        return state

    def get_current_pos(self):
        """Get current joint angles. Returns np.ndarray."""
        joint_angles = np.zeros(self.motor_count)
        for i in range(self.motor_count):
            state = self.Motors[i].get_current_motor_state()
            joint_angles[i] = state.position
        return joint_angles

    def get_current_vel(self):
        """Get current joint velocities. Returns np.ndarray."""
        joint_velocities = np.zeros(self.motor_count)
        for i in range(self.motor_count):
            state = self.Motors[i].get_current_motor_state()
            joint_velocities[i] = state.velocity
        return joint_velocities

    def get_current_torque(self):
        """Get current joint torques. Returns np.ndarray."""
        joint_torques = np.zeros(self.motor_count)
        for i in range(self.motor_count):
            state = self.Motors[i].get_current_motor_state()
            joint_torques[i] = state.torque
        return joint_torques

    def get_current_state_gripper(self):
        """Get the current gripper state"""
        return self.Motors[self.gripper_id-1].get_current_motor_state()

    def get_current_pos_gripper(self):
        """Get the current gripper position"""
        state = self.Motors[self.gripper_id-1].get_current_motor_state()
        return state.position

    def get_current_vel_gripper(self):
        """Get the current gripper velocity"""
        state = self.Motors[self.gripper_id-1].get_current_motor_state()
        return state.velocity

    def get_current_torque_gripper(self):
        """Get the current gripper torque"""
        state = self.Motors[self.gripper_id-1].get_current_motor_state()
        return state.torque

    #######################
    # Basic motion control
    #######################
    def Joint_Pos_Vel(self, pos, vel, max_tqu=None, iswait=False, tolerance=0.1, timeout=15.0):
        """
        Per-joint position + velocity + max-torque control (each joint configured
        independently).

        Args:
            pos: Target positions list/array [joint1, joint2, ..., jointN]
            vel: Target velocities list/array [joint1, joint2, ..., jointN]
            max_tqu: Max-torque list/array. If None, the defaults from the
                configuration file are used.
            iswait: Whether to block until the motion completes
            tolerance: Position tolerance (rad)
            timeout: Wait timeout (seconds)

        Returns:
            bool: Whether the control command was executed successfully

        Notes:
            Each joint receives an independent position and velocity, which is
            useful when different joints need to move at different speeds.
        """
        # If max_tqu is not provided, use the defaults from the configuration file
        if max_tqu is None:
            max_tqu = self.max_torque
        else:
            max_tqu = np.asarray(max_tqu)

        # Check joint count (excluding the gripper motor)
        if not (len(pos) == len(vel) == len(max_tqu) == self.motor_count):
            raise ValueError(f"Joint parameter length must be {self.motor_count}")
        # Convert to numpy array
        pos = np.asarray(pos)

        # Check that positions are within the joint limits
        if self.joint_limits is not None:
            lower = self.joint_limits['lower']
            upper = self.joint_limits['upper']
            # Check whether any position is out of range
            out_of_range = np.logical_or(pos < lower, pos > upper)
            if np.any(out_of_range):
                print("\n" + "="*60)
                print("Warning: target position is outside the joint limits!")
                print(f"Target position: {pos}")
                print(f"Lower limit: {lower}")
                print(f"Upper limit: {upper}")
                out_indices = np.where(out_of_range)[0]
                for idx in out_indices:
                    print(f"  Joint {idx+1}: {pos[idx]:.3f} is not within [{lower[idx]:.3f}, {upper[idx]:.3f}]")
                print("Control command rejected to keep the arm safe")
                print("="*60 + "\n")
                return False

        # Control the joints (excluding the gripper motor)
        for i in range(self.motor_count):
            motor = self.Motors[i]
            motor.pos_vel_MAXtqe(pos[i], vel[i], max_tqu[i])
        self.motor_send_cmd()
        if iswait:
            return self.wait_for_position(pos, tolerance, timeout)
        return True

    def Joint_Vel(self, vel):
        """
        Joint velocity control

        Args:
            vel: Target velocity list/array [joint1, joint2, ..., jointN] (rad/s)

        Returns:
            bool: Whether the control command was executed successfully

        Notes:
            Drives the joints directly in velocity mode without performing a
            position-limit check. Useful when precise velocity control is
            required. Velocities are clamped to the limits configured in the
            configuration file.
        """
        # Argument check
        if len(vel) != self.motor_count:
            raise ValueError(f"Target velocity length must be {self.motor_count}")

        # Convert to numpy array
        vel = np.asarray(vel)

        # Velocity-limit check
        if self.velocity_limits is not None:
            # Check whether any velocity exceeds the limit
            abs_vel = np.abs(vel)
            out_of_limit = abs_vel > self.velocity_limits
            if np.any(out_of_limit):
                print("\n" + "="*60)
                print("Warning: target velocity exceeds the limit!")
                print(f"Target velocity: {vel}")
                print(f"Velocity limit: +/-{self.velocity_limits}")
                out_indices = np.where(out_of_limit)[0]
                for idx in out_indices:
                    print(f"  Joint {idx+1}: {vel[idx]:.3f} rad/s exceeds limit +/-{self.velocity_limits[idx]:.3f} rad/s")
                print("Velocity will be clamped to the safe range")
                print("="*60 + "\n")
                # Clamp
                vel = np.clip(vel, -self.velocity_limits, self.velocity_limits)

        # Joint-limit protection: when at a limit, zero out the velocity component
        # toward that limit; motion in the opposite direction is unaffected.
        if self.joint_limits is not None:
            current_pos = self.get_current_pos()
            lower = np.asarray(self.joint_limits['lower'])
            upper = np.asarray(self.joint_limits['upper'])
            limit_margin = 0.02  # rad, margin to trigger protection early

            at_upper = current_pos >= (upper - limit_margin)
            at_lower = current_pos <= (lower + limit_margin)

            vel = np.where(at_upper & (vel > 0), 0.0, vel)
            vel = np.where(at_lower & (vel < 0), 0.0, vel)

        # Control the joints (excluding the gripper motor)
        for i in range(self.motor_count):
            motor = self.Motors[i]
            motor.velocity(vel[i])
        self.motor_send_cmd()
        return True

    def moveJ(self, pos, duration, max_tqu=None, iswait=False, tolerance=0.1, timeout=15.0):
        """
        Joint-space motion control (all joints reach the targets synchronously
        within the specified duration).

        Args:
            pos: Target positions list/array [joint1, joint2, ..., jointN] (rad)
            duration: Motion duration (seconds). All joints reach their targets
                simultaneously within this time.
            max_tqu: Max-torque list/array. If None, the defaults from the
                configuration file are used.
            iswait: Whether to block until the motion completes
            tolerance: Position tolerance (rad)
            timeout: Wait timeout (seconds)

        Returns:
            bool: Whether the control command was executed successfully

        Notes:
            Computes per-joint average velocity as
            (target - current) / duration so that every joint arrives at its
            target at the same time. Useful for coordinated motion, similar to
            the moveJ command on industrial robots.
        """
        # If max_tqu is not provided, use the defaults from the configuration file
        if max_tqu is None:
            max_tqu = self.max_torque
        else:
            max_tqu = np.asarray(max_tqu)

        # Argument check
        if len(pos) != self.motor_count:
            raise ValueError(f"Target position length must be {self.motor_count}")
        if len(max_tqu) != self.motor_count:
            raise ValueError(f"Max-torque length must be {self.motor_count}")
        if duration <= 0:
            raise ValueError(f"Duration must be greater than 0, got: {duration}")

        # Convert to numpy array
        pos = np.asarray(pos)

        # Check that positions are within the joint limits
        if self.joint_limits is not None:
            lower = self.joint_limits['lower']
            upper = self.joint_limits['upper']
            out_of_range = np.logical_or(pos < lower, pos > upper)
            if np.any(out_of_range):
                print("\n" + "="*60)
                print("Warning: target position is outside the joint limits!")
                print(f"Target position: {pos}")
                print(f"Lower limit: {lower}")
                print(f"Upper limit: {upper}")
                out_indices = np.where(out_of_range)[0]
                for idx in out_indices:
                    print(f"  Joint {idx+1}: {pos[idx]:.3f} is not within [{lower[idx]:.3f}, {upper[idx]:.3f}]")
                print("Control command rejected to keep the arm safe")
                print("="*60 + "\n")
                return False

        # Get current positions
        current_pos = self.get_current_pos()

        # Compute velocity: v = (target - current) / duration
        # This ensures all joints reach their targets together at 'duration'
        vel = (pos - current_pos) / duration

        # Call per-joint position+velocity control
        return self.Joint_Pos_Vel(pos, vel, max_tqu, iswait, tolerance, timeout)

    def pos_vel_tqe_kp_kd(self, pos, vel, tqe, kp, kd):
        """Five-parameter MIT joint control mode"""
        # Check joint count (excluding the gripper motor)
        params = [pos, vel, tqe, kp, kd]
        if not all(len(p) == self.motor_count for p in params):
            raise ValueError(f"Joint parameter length must be {self.motor_count}")

        # Convert to numpy array
        pos = np.asarray(pos)

        # Check that positions are within the joint limits
        if self.joint_limits is not None:
            lower = self.joint_limits['lower']
            upper = self.joint_limits['upper']
            # Check whether any position is out of range
            out_of_range = np.logical_or(pos < lower, pos > upper)
            if np.any(out_of_range):
                print("\n" + "="*60)
                print("Warning: target position is outside the joint limits!")
                print(f"Target position: {pos}")
                print(f"Lower limit: {lower}")
                print(f"Upper limit: {upper}")
                out_indices = np.where(out_of_range)[0]
                for idx in out_indices:
                    print(f"  Joint {idx+1}: {pos[idx]:.3f} is not within [{lower[idx]:.3f}, {upper[idx]:.3f}]")
                print("Control command rejected to keep the arm safe")
                print("="*60 + "\n")
                return False

        # Control the joints (excluding the gripper motor)
        for i in range(self.motor_count):
            motor = self.Motors[i]
            motor.pos_vel_tqe_kp_kd(pos[i], vel[i], tqe[i], kp[i], kd[i])
        self.motor_send_cmd()
        return True

    #######################
    # Gripper control
    #######################
    def gripper_control(self, pos, vel, max_tqu=0.5):
        """Gripper control (position + velocity + max-torque mode)"""
        # Check that the gripper target position is within its limits
        if self.gripper_limits is not None:
            lower = self.gripper_limits['lower']
            upper = self.gripper_limits['upper']
            # Check whether the position is out of range
            if pos < lower or pos > upper:
                print("\n" + "="*60)
                print("Warning: gripper target position is outside its limits!")
                print(f"Target position: {pos}")
                print(f"Lower limit: {lower}")
                print(f"Upper limit: {upper}")
                print(f"Gripper position {pos:.3f} is not within [{lower:.3f}, {upper:.3f}]")
                print("Control command rejected to keep the gripper safe")
                print("="*60 + "\n")
                return False

        self.Motors[self.gripper_id-1].pos_vel_MAXtqe(pos, vel, max_tqu)
        self.motor_send_cmd()
        return True

    def gripper_control_MIT(self, pos, vel, tqe, kp, kd):
        """Gripper control (5-parameter MIT mode)"""
        # Check that the gripper target position is within its limits
        if self.gripper_limits is not None:
            lower = self.gripper_limits['lower']
            upper = self.gripper_limits['upper']
            # Check whether the position is out of range
            if pos < lower or pos > upper:
                print("\n" + "="*60)
                print("Warning: gripper target position is outside its limits!")
                print(f"Target position: {pos}")
                print(f"Lower limit: {lower}")
                print(f"Upper limit: {upper}")
                print(f"Gripper position {pos:.3f} is not within [{lower:.3f}, {upper:.3f}]")
                print("Control command rejected to keep the gripper safe")
                print("="*60 + "\n")
                return False

        self.Motors[self.gripper_id-1].pos_vel_tqe_kp_kd(pos, vel, tqe, kp, kd)
        self.motor_send_cmd()
        return True

    def gripper_open(self, pos=1.6, vel=0.5, max_tqu=0.5):
        """Open the gripper"""
        self.gripper_control(pos, vel, max_tqu)

    def gripper_close(self, pos=0.0, vel=0.5, max_tqu=0.5):
        """Close the gripper"""
        self.gripper_control(pos, vel, max_tqu)

    #######################
    # Position-check helpers
    #######################
    def check_position_reached(self, target_positions, tolerance=0.1):
        """Check whether the arm joints have reached the target positions"""
        all_reached = True
        position_errors = []

        self.send_get_motor_state_cmd()
        self.motor_send_cmd()
        # Check the arm joints
        for i in range(self.motor_count):
            state = self.Motors[i].get_current_motor_state()
            error = abs(state.position - target_positions[i])
            position_errors.append(error)
            if error > tolerance:
                all_reached = False

        return all_reached, position_errors

    def wait_for_position(self, target_positions, tolerance=0.01, timeout=15.0):
        """Wait for the joints to reach the target positions"""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            reached, _ = self.check_position_reached(target_positions, tolerance)
            if reached:
                return True
            time.sleep(0.02)
        return False

    #######################
    # Kinematics
    #######################
    def forward_kinematics(self, joint_angles=None):
        """Compute forward kinematics with pinocchio. Returns end-effector position and transform."""
        if self.model is None:
            print("Model not loaded")
            return None

        # If no joint angles are provided, use the current ones
        if joint_angles is None:
            joint_angles = self.get_current_pos()

        # Build the joint configuration vector
        q = np.zeros(self.model.nq)
        for i, joint_name in enumerate(self.joint_names):
            if i < len(joint_angles):
                joint_id = self.model.getJointId(joint_name)
                idx = self.model.joints[joint_id].idx_q
                q[idx] = joint_angles[i]

        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the transform of the end-effector frame
        eef_transform = self.data.oMf[self.end_effector_frame_id]
        position = eef_transform.translation.copy()
        rotation = eef_transform.rotation.copy()

        # Build the 4x4 transform matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position

        return {
            'position': position.tolist(),
            'rotation': rotation,
            'transform': T,
            'joint_angles': joint_angles
        }

    def get_jacobian(self, joint_angles=None):
        """
        Get the end-effector Jacobian (world-aligned frame).

        Args:
            joint_angles: Joint angles. If None, the current angles are used.

        Returns:
            J: 6xN Jacobian matrix. Columns follow the order of joint_names.
        """
        if self.model is None:
            print("Model not loaded")
            return None

        if joint_angles is None:
            joint_angles = self.get_current_pos()

        q = np.zeros(self.model.nq)
        for i, joint_name in enumerate(self.joint_names):
            joint_id = self.model.getJointId(joint_name)
            idx = self.model.joints[joint_id].idx_q
            q[idx] = joint_angles[i]

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        J_full = pin.computeFrameJacobian(
            self.model, self.data, q,
            self.end_effector_frame_id, pin.LOCAL_WORLD_ALIGNED
        )

        J = np.zeros((6, len(self.joint_names)))
        for i, joint_name in enumerate(self.joint_names):
            jid = self.model.getJointId(joint_name)
            idx = self.model.joints[jid].idx_v
            J[:, i] = J_full[:, idx]

        return J

    def get_manipulability(self, joint_angles=None):
        """
        Compute the manipulability of the current configuration:
        mu = sqrt(det(J J^T)).

        Args:
            joint_angles: Joint angles. If None, the current angles are used.

        Returns:
            float: Manipulability. Smaller values indicate proximity to a
                singular configuration.
        """
        J = self.get_jacobian(joint_angles)
        if J is None:
            return 0.0
        JJT = J @ J.T
        det = np.linalg.det(JJT)
        return np.sqrt(max(det, 0.0))

    @staticmethod
    def compute_damped_pseudoinverse(J, damping=0.01):
        """
        Compute the damped pseudo-inverse of the Jacobian:
        J_damp = J^T (J J^T + lambda^2 I)^(-1).

        Args:
            J: Jacobian matrix
            damping: Damping coefficient lambda

        Returns:
            J_damp: Damped pseudo-inverse matrix
        """
        m = J.shape[0]
        JJT = J @ J.T
        try:
            J_damp = J.T @ np.linalg.inv(JJT + (damping ** 2) * np.eye(m))
        except np.linalg.LinAlgError:
            J_damp = J.T @ np.linalg.inv(JJT + (damping * 10) ** 2 * np.eye(m))
        return J_damp

    def inverse_kinematics(self, target_position, target_rotation=None, init_q=None,
                               max_iter=1000, eps=1e-3, damping=1e-2, adaptive_damping=True,
                               multi_init=True, num_attempts=8):
        """
        Solve inverse kinematics using Damped Least Squares (DLS).

        Args:
            target_position: Target position [x, y, z] (m)
            target_rotation: Target 3x3 rotation matrix. If None, only position
                is considered.
            init_q: Initial joint angles. If None, the current angles are used
                (effective only when multi_init=False).
            max_iter: Maximum iterations
            eps: Convergence threshold (norm of the position error)
            damping: Damping coefficient lambda used to avoid Jacobian singularities
            adaptive_damping: Whether to use adaptive damping
            multi_init: Whether to try multiple initial guesses (improves success rate)
            num_attempts: Number of multi-init attempts (only when multi_init=True)

        Returns:
            np.ndarray: Joint-angle array [joint1, joint2, ..., jointN] (rad)
            None: If the solver fails

        Notes:
            Damped Least Squares uses: dq = J^T (J J^T + lambda^2 I)^(-1) * e.
            Compared with the standard pseudo-inverse, DLS is more stable and
            robust near singular configurations. Adaptive damping adjusts the
            damping coefficient dynamically based on the error magnitude.

            When multi_init=True, several initial joint configurations are
            tried:
            - Current configuration
            - Zero configuration
            - Mid-point of the joint limits
            - Random configurations (within joint limits)
            Returns the first successful solution, or the best one found.
        """
        if self.model is None:
            print("Model not loaded")
            return None

        # If multi-init is enabled
        if multi_init:
            return self._inverse_kinematics_dls_multi_init_impl(
                target_position, target_rotation, num_attempts,
                max_iter, eps, damping, adaptive_damping
            )

        # Single-init solver
        return self._inverse_kinematics_dls_single_impl(
            target_position, target_rotation, init_q,
            max_iter, eps, damping, adaptive_damping
        )

    def _inverse_kinematics_dls_single_impl(self, target_position, target_rotation, init_q,
                                            max_iter, eps, damping, adaptive_damping):
        """
        Single-init implementation of the DLS IK solver (internal helper).
        """
        # Target pose
        if target_rotation is None:
            target_rotation = np.eye(3)

        target_rotation_matrix = np.array(target_rotation)
        oMdes = pin.SE3(target_rotation_matrix, np.array(target_position))

        # Initial joint angles
        if init_q is None:
            init_q = self.get_current_pos()

        q = np.zeros(self.model.nq)
        for i, joint_name in enumerate(self.joint_names):
            if i < len(init_q):
                joint_id = self.model.getJointId(joint_name)
                idx = self.model.joints[joint_id].idx_q
                q[idx] = init_q[i]

        # End-effector frame ID
        frame_id = self.end_effector_frame_id

        # Joint limits
        lower_limits = None
        upper_limits = None
        if self.joint_limits is not None:
            lower_limits = self.joint_limits['lower']
            upper_limits = self.joint_limits['upper']

        # Iterative solver
        dt = 1e-1
        lambda_base = damping  # Base damping coefficient

        for i in range(max_iter):
            # Compute FK and the error
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            iMd = self.data.oMf[frame_id].actInv(oMdes)
            err = pin.log(iMd).vector

            # Error norm
            err_norm = np.linalg.norm(err)

            # Convergence check
            if err_norm < eps:
                # Extract joint angles and return as numpy array
                result = []
                for joint_name in self.joint_names:
                    jid = self.model.getJointId(joint_name)
                    idx = self.model.joints[jid].idx_q
                    result.append(q[idx])
                return np.array(result)

            # Compute the Jacobian
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            # Adaptive damping (adjusted by the error magnitude)
            if adaptive_damping:
                # Larger error -> smaller damping, allowing bigger steps.
                # Smaller error -> larger damping, improving stability.
                lambda_adaptive = lambda_base * (1.0 + 1.0 / (err_norm + 0.1))
            else:
                lambda_adaptive = lambda_base

            # Damped Least Squares solve:
            # dq = J^T (J J^T + lambda^2 I)^(-1) * e
            JJT = J.dot(J.T)
            damping_matrix = lambda_adaptive**2 * np.eye(6)

            # Solve the linear system (J J^T + lambda^2 I) * alpha = e
            try:
                alpha = np.linalg.solve(JJT + damping_matrix, err)
            except np.linalg.LinAlgError:
                print(f"DLS solver failed (iteration {i+1}); the matrix may be ill-conditioned")
                return None

            # Compute joint velocity: v = J^T * alpha
            v = -J.T.dot(alpha)

            # Limit velocity magnitude to prevent numerical blow-up
            v_norm = np.linalg.norm(v)
            max_velocity = 10.0
            if v_norm > max_velocity:
                v = v * (max_velocity / v_norm)

            # Update joint angles
            q_new = pin.integrate(self.model, q, v * dt)

            # Check whether the new joint angles are still within limits
            if lower_limits is not None and upper_limits is not None:
                q_check = []
                for joint_name in self.joint_names:
                    jid = self.model.getJointId(joint_name)
                    idx = self.model.joints[jid].idx_q
                    q_check.append(q_new[idx])
                q_check = np.array(q_check)

                # Out of range?
                out_of_range = np.logical_or(q_check < lower_limits, q_check > upper_limits)
                if np.any(out_of_range):
                    print("DLS IK detected out-of-range joint angles during iteration; target pose may be unreachable")
                    print(f"Current iteration: {i+1}/{max_iter}, error norm: {err_norm:.6f}")
                    return None

            q = q_new

        print(f"DLS IK did not converge. Final error: {err_norm:.6f}. Check whether the target is outside the workspace.")
        return None

    def _inverse_kinematics_dls_multi_init_impl(self, target_position, target_rotation,
                                                num_attempts, max_iter, eps, damping, adaptive_damping):
        """
        Multi-init implementation of the DLS IK solver (internal helper).
        """
        # Prepare multiple initial guesses
        init_configs = []

        # 1. Current configuration
        init_configs.append(self.get_current_pos())

        # 2. Zero configuration
        init_configs.append(np.zeros(self.motor_count))

        # 3. Mid-point of the joint limits
        if self.joint_limits is not None:
            mid_config = (self.joint_limits['lower'] + self.joint_limits['upper']) / 2
            init_configs.append(mid_config)

        # 4. Random configurations within the joint limits
        if self.joint_limits is not None:
            lower = self.joint_limits['lower']
            upper = self.joint_limits['upper']

            for _ in range(num_attempts - 3):
                random_config = np.random.uniform(lower, upper)
                init_configs.append(random_config)
        else:
            # When no limits are available, use small random angles
            for _ in range(num_attempts - 3):
                random_config = np.random.uniform(-np.pi/4, np.pi/4, self.motor_count)
                init_configs.append(random_config)

        # Try each initial guess
        best_result = None
        best_error = float('inf')

        for i, init_q in enumerate(init_configs[:num_attempts]):
            result_q = self._inverse_kinematics_dls_single_impl(
                target_position=target_position,
                target_rotation=target_rotation,
                init_q=init_q,
                max_iter=max_iter,
                eps=eps,
                damping=damping,
                adaptive_damping=adaptive_damping
            )

            if result_q is not None:
                # Verify solution quality (FK position vs. target position error)
                fk_result = self.forward_kinematics(result_q)
                if fk_result is not None:
                    actual_pos = np.array(fk_result['position'])
                    target_pos = np.array(target_position)
                    error = np.linalg.norm(actual_pos - target_pos)

                    if error < best_error:
                        best_error = error
                        best_result = result_q

                    # Early exit if the error is small enough
                    if error < eps:
                        print(f"Multi-init IK succeeded (attempt {i+1}/{num_attempts}). Error: {error:.6f}m")
                        return result_q

        if best_result is not None:
            print(f"Multi-init IK finished. Best error: {best_error:.6f}m")
            return best_result

        print(f"Multi-init IK failed after trying {num_attempts} different initial configurations")
        return None

    #######################
    # MoveIt-style Cartesian control
    #######################
    def compute_cartesian_path(self, waypoints, avoid_collisions=False):
        """
        Compute a Cartesian path.

        Args:
            waypoints: List of waypoints [{'position': [x,y,z], 'rotation': R}]
            avoid_collisions: Whether to perform collision checking

        Returns:
            joint_trajectory: Joint trajectory
            fraction: Completed fraction in [0, 1]
        """
        if len(waypoints) < 2:
            print("Error: at least 2 waypoints are required")
            return None, 0.0

        joint_trajectory = []
        current_q = self.get_current_pos()

        # Interpolate each consecutive pair of waypoints
        for i in range(len(waypoints) - 1):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]

            # Compute interpolation points for this segment
            segment_traj, success = self._interpolate_segment(
                start_pose, end_pose, current_q
            )

            if not success:
                # Partial success: return what we have so far
                fraction = (i + len(segment_traj) / self._compute_segment_steps(start_pose, end_pose)) / (len(waypoints) - 1)
                return joint_trajectory, fraction

            # Append to the full trajectory
            joint_trajectory.extend(segment_traj)
            if len(segment_traj) > 0:
                current_q = segment_traj[-1]

        return joint_trajectory, 1.0

    def _interpolate_segment(self, start_pose, end_pose, init_q):
        """
        Interpolate a single path segment (internal method).

        Args:
            start_pose: Start pose dict {'position': [x,y,z], 'rotation': R}
            end_pose: End pose dict {'position': [x,y,z], 'rotation': R}
            init_q: Initial joint angles used to seed the IK solver

        Returns:
            segment_trajectory: List of joint configurations
            success: Whether the segment was interpolated successfully (bool)

        Notes:
            This method computes the number of interpolation steps based on
            eef_step, performs linear interpolation on position and SLERP on
            orientation, then solves IK for each sample. Joint-jump detection
            (jump_threshold) is included to keep the trajectory smooth and
            continuous.
        """
        # 1. Number of steps required
        num_steps = self._compute_segment_steps(start_pose, end_pose)

        segment_trajectory = []
        current_q = init_q

        # 2. Process each interpolation point
        for step in range(1, num_steps + 1):
            t = step / num_steps

            # 3. Linear interpolation of position
            pos = (1 - t) * np.array(start_pose['position']) + t * np.array(end_pose['position'])

            # 4. SLERP interpolation of orientation
            rot_start = R.from_matrix(start_pose['rotation'])
            rot_end = R.from_matrix(end_pose['rotation'])

            # Use scipy's slerp helpers
            key_times = [0, 1]
            key_rots = R.from_quat([rot_start.as_quat(), rot_end.as_quat()])
            slerp = R.from_quat(key_rots.as_quat())

            # Simplified SLERP: linear quaternion interpolation + normalization
            q_start = rot_start.as_quat()
            q_end = rot_end.as_quat()

            # Ensure shortest path (flip sign when dot product is negative)
            if np.dot(q_start, q_end) < 0:
                q_end = -q_end

            # Linear interpolation
            q_interp = (1 - t) * q_start + t * q_end

            # Normalize
            q_interp = q_interp / np.linalg.norm(q_interp)

            # Back to rotation matrix
            rot_interp = R.from_quat(q_interp)

            # 5. IK solve
            q_solution = self.inverse_kinematics(
                target_position=pos,
                target_rotation=rot_interp.as_matrix(),
                init_q=current_q,
                multi_init=False
            )

            if q_solution is None:
                # IK failed
                print(f"  IK failed at step {step}/{num_steps}")
                return segment_trajectory, False

            # 6. Joint-jump detection (a key MoveIt feature)
            if len(segment_trajectory) > 0:
                if self._has_joint_jump(current_q, q_solution):
                    print(f"  Joint jump detected at step {step}/{num_steps}")
                    return segment_trajectory, False

            segment_trajectory.append(q_solution)
            current_q = q_solution

        return segment_trajectory, True

    def _compute_segment_steps(self, start_pose, end_pose):
        """
        Compute the number of steps needed for the segment (based on eef_step
        and orientation change).

        Takes the larger of position distance and orientation change.
        """
        # Cartesian position distance
        position_distance = np.linalg.norm(
            np.array(end_pose['position']) - np.array(start_pose['position'])
        )

        # Orientation change (rotation angle)
        from scipy.spatial.transform import Rotation as R
        rot_start = R.from_matrix(start_pose['rotation'])
        rot_end = R.from_matrix(end_pose['rotation'])

        # Relative rotation
        rot_diff = rot_end * rot_start.inv()
        angle_diff = rot_diff.magnitude()  # Rotation angle (rad)

        # Steps required by position distance
        steps_from_position = int(np.ceil(position_distance / self.eef_step))

        # Steps required by orientation change (assume max 0.1 rad ~= 5.7 deg per step)
        max_rotation_per_step = 0.1  # rad
        steps_from_rotation = int(np.ceil(angle_diff / max_rotation_per_step))

        # Take the larger value, at least 1
        num_steps = max(1, steps_from_position, steps_from_rotation)

        return num_steps

    def _has_joint_jump(self, q1, q2):
        """
        Detect a joint jump (MoveIt's jump_threshold).

        Returns True if any joint changes by more than the threshold.
        """
        q1 = np.array(q1)
        q2 = np.array(q2)

        joint_deltas = np.abs(q2 - q1)

        # Any joint change exceeding the threshold?
        if np.any(joint_deltas > self.jump_threshold):
            max_jump_idx = np.argmax(joint_deltas)
            print(f"    Joint {max_jump_idx + 1} jump: {np.rad2deg(joint_deltas[max_jump_idx]):.2f} deg")
            return True

        return False

    def compute_time_parameterization(self, joint_trajectory, duration=None):
        """
        Time-parameterize the trajectory (similar to MoveIt's
        IterativeParabolicTimeParameterization).

        Automatically computes the timestamp of each waypoint while respecting
        velocity / acceleration limits.
        """
        if len(joint_trajectory) < 2:
            return [0.0]

        timestamps = [0.0]

        if duration is not None:
            # User-specified total duration, distribute uniformly
            dt = duration / (len(joint_trajectory) - 1)
            for i in range(1, len(joint_trajectory)):
                timestamps.append(timestamps[-1] + dt)
        else:
            # Compute timings automatically (based on velocity/acceleration limits)
            for i in range(1, len(joint_trajectory)):
                q_prev = np.array(joint_trajectory[i - 1])
                q_curr = np.array(joint_trajectory[i])

                # Joint displacement
                delta_q = q_curr - q_prev

                # Time needed considering the velocity limit
                dt_vel = np.max(np.abs(delta_q) / self.velocity_limits)

                # Consider the acceleration limit (simplified)
                dt_acc = np.sqrt(2 * np.max(np.abs(delta_q)) / np.max(self.acceleration_limits))

                # Take the larger value
                dt = max(dt_vel, dt_acc, 0.01)  # Minimum 10 ms

                timestamps.append(timestamps[-1] + dt)

        return timestamps

    def smooth_trajectory_spline(self, joint_trajectory, timestamps):
        """
        Smooth the trajectory with a cubic spline (improves smoothness).

        Produces continuous velocity and acceleration.
        """
        if len(joint_trajectory) < 2:
            return joint_trajectory, [0.0], [np.zeros(self.motor_count)]

        # Convert to numpy arrays
        q_array = np.array(joint_trajectory)
        t_array = np.array(timestamps)

        # Create a cubic spline for each joint
        splines = []
        for joint_idx in range(q_array.shape[1]):
            spline = CubicSpline(t_array, q_array[:, joint_idx], bc_type='clamped')
            splines.append(spline)

        # Resample (denser sampling - the key step!)
        dt_resample = self.resample_dt  # Configurable resampling step
        t_new = np.arange(t_array[0], t_array[-1], dt_resample)

        # Make sure the last point is included
        if t_new[-1] < t_array[-1]:
            t_new = np.append(t_new, t_array[-1])

        q_smooth = []
        v_smooth = []

        for t in t_new:
            q_t = [spline(t) for spline in splines]
            v_t = [spline(t, 1) for spline in splines]  # First derivative = velocity

            q_smooth.append(q_t)
            v_smooth.append(v_t)

        return q_smooth, t_new.tolist(), v_smooth

    def moveL(self, target_position, target_rotation=None, duration=None,
              use_spline=True, max_tqu=None):
        """
        Cartesian-space straight-line motion (modeled on MoveIt's approach).

        Args:
            target_position: Target position [x, y, z] (m)
            target_rotation: Target orientation (3x3 rotation matrix). If None,
                the current orientation is kept.
            duration: Motion duration (seconds). If None, it is computed
                automatically from the velocity / acceleration limits.
            use_spline: Whether to smooth the trajectory with a cubic spline
                (default True)
            max_tqu: Max-torque limits array. If None, the defaults from the
                configuration file are used.

        Returns:
            bool: Whether the motion executed successfully

        Notes:
            This method implements a MoveIt-style Cartesian planning and
            execution pipeline:
            1. compute_cartesian_path produces the Cartesian path
               (interpolated based on eef_step).
            2. compute_time_parameterization adds timing information.
            3. Optionally smooth_trajectory_spline applies spline smoothing.
            4. Joint_Pos_Vel executes the trajectory.

            Features:
            - End-effector moves in a straight line
            - Orientation interpolated with SLERP
            - Joint-jump detection (jump_threshold)
            - Respects velocity and acceleration limits
            - Smooth, continuous trajectory

        Examples:
            # Move to a target position while keeping current orientation
            robot.moveL([0.3, 0.0, 0.4])

            # Move to a target pose with a specified duration
            robot.moveL([0.3, 0.0, 0.4], target_rotation=R, duration=3.0)
        """
        print("="*60)
        print("MoveIt-style moveL started")
        print("="*60)

        # 1. Get current pose
        current_fk = self.forward_kinematics()
        if current_fk is None:
            print("Error: cannot get current pose")
            return False

        start_pose = {
            'position': current_fk['position'],
            'rotation': current_fk['rotation']
        }

        if target_rotation is None:
            target_rotation = start_pose['rotation']

        end_pose = {
            'position': target_position,
            'rotation': target_rotation
        }

        # 2. Compute the Cartesian path (MoveIt's computeCartesianPath)
        print(f"\nStep 1: compute Cartesian path (eef_step={self.eef_step*1000:.1f} mm)")
        waypoints = [start_pose, end_pose]
        joint_trajectory, fraction = self.compute_cartesian_path(waypoints)

        if joint_trajectory is None or len(joint_trajectory) == 0:
            print("Error: path planning failed")
            return False

        print(f"  Path planning done: {len(joint_trajectory)} joint configurations")
        print(f"  Completed fraction: {fraction*100:.1f}%")

        if fraction < 0.99:
            print(f"  Warning: only {fraction*100:.1f}% of the path was completed")

        # 3. Time parameterization (MoveIt's IterativeParabolicTimeParameterization)
        print(f"\nStep 2: trajectory time parameterization")
        timestamps = self.compute_time_parameterization(joint_trajectory, duration)
        total_time = timestamps[-1]
        print(f"  Total time: {total_time:.2f}s")
        if total_time > 0:
            print(f"  Average rate: {len(joint_trajectory)/total_time:.1f}Hz")
        else:
            print(f"  Warning: trajectory time is 0 (position may not change, only orientation)")

        # 4. Spline smoothing (optional, further improves smoothness)
        if use_spline:
            print(f"\nStep 3: cubic-spline smoothing")
            joint_trajectory, timestamps, velocities = self.smooth_trajectory_spline(
                joint_trajectory, timestamps
            )
            print(f"  After resampling: {len(joint_trajectory)} points")
        else:
            # Compute velocities by finite differences
            velocities = []
            for i in range(len(joint_trajectory) - 1):
                dt = timestamps[i+1] - timestamps[i]
                vel = (np.array(joint_trajectory[i+1]) - np.array(joint_trajectory[i])) / dt
                velocities.append(vel)
            velocities.append(velocities[-1] if velocities else np.zeros(self.motor_count))

        # 5. Execute the trajectory
        print(f"\nStep 4: execute the trajectory")
        success = self._execute_trajectory(
            joint_trajectory, timestamps, velocities, max_tqu
        )

        if success:
            print("\nmoveL executed successfully")
        else:
            print("\nmoveL execution failed")

        return success

    def _execute_trajectory(self, joint_trajectory, timestamps, velocities, max_tqu=None):
        """
        Execute the trajectory (using Joint_Pos_Vel mode).
        """
        # Resolve the max-torque limits
        if max_tqu is None:
            if hasattr(self, 'max_torque'):
                max_tqu = self.max_torque
            else:
                max_tqu = np.array([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])

        start_time = time.perf_counter()

        for i in range(len(joint_trajectory)):
            loop_start = time.perf_counter()

            # Target time for the current sample
            target_time = timestamps[i]

            # Wait until the correct time
            while (time.perf_counter() - start_time) < target_time:
                time.sleep(0.0001)

            # Send the command using Joint_Pos_Vel mode
            success = self.Joint_Pos_Vel(
                pos=joint_trajectory[i],
                vel=velocities[i],
                max_tqu=max_tqu,
                iswait=False
            )

            if not success:
                print(f"  Control failed at point {i+1}/{len(joint_trajectory)}")
                return False

            # Monitor timing
            actual_time = time.perf_counter() - start_time
            time_error = actual_time - target_time
            if time_error > 0.005:  # > 5 ms
                print(f"  Timing lag: {time_error*1000:.1f} ms")

        total_time = time.perf_counter() - start_time
        print(f"  Actual execution time: {total_time:.3f}s")

        return True

    #######################
    # Dynamics
    #######################
    def get_Gravity(self, q=None):
        """
        Get the gravity-compensation torque G(q). Returns np.ndarray.
        The default gravity direction is the negative Z axis: [0, 0, -9.81].
        Adjust as needed for your setup.

        Args:
            q: Joint-angle array. If None, the current angles are used.

        Returns:
            G: Gravity-compensation torque array (np.ndarray)
        """
        if q is None:
            q = self.get_current_pos()
        # Make sure it is a numpy array (no copy if already an array)
        q = np.asarray(q)

        # Temporarily save the original gravity setting
        original_gravity = self.model.gravity.copy()

        # Set gravity to the negative Z axis
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

        # Compute gravity compensation
        G = pin.computeGeneralizedGravity(self.model, self.data, q)

        # Restore the original gravity setting
        self.model.gravity.linear = original_gravity.linear

        return G

    def get_Coriolis(self, q=None, v=None):
        """Get the Coriolis matrix C(q, v). Returns np.ndarray."""
        if q is None:
            q = self.get_current_pos()
        if v is None:
            v = self.get_current_vel()
        # Make sure inputs are numpy arrays
        q = np.asarray(q)
        v = np.asarray(v)
        # Compute the Coriolis matrix
        C = pin.computeCoriolisMatrix(self.model, self.data, q, v)
        return C

    def get_Coriolis_vector(self, q=None, v=None):
        """Get the Coriolis vector C(q, v) * v (kept for backward compatibility). Returns np.ndarray."""
        C = self.get_Coriolis(q, v)
        if v is None:
            v = self.get_current_vel()
        else:
            v = np.asarray(v)
        return C.dot(v)

    def get_Mass_Matrix(self, q=None):
        """Get the full mass matrix. Returns np.ndarray."""
        if q is None:
            q = self.get_current_pos()
        # Make sure it is a numpy array
        q = np.asarray(q)
        # Compute the mass matrix
        M = pin.crba(self.model, self.data, q)
        # Return the full mass matrix
        return M[:len(q), :len(q)]

    def get_Inertia_Terms(self, q=None, a=None):
        """Get the inertia torque M(q) * a. Returns np.ndarray."""
        if q is None:
            q = self.get_current_pos()
        if a is None:
            a = np.zeros(self.motor_count)
        # Make sure inputs are numpy arrays
        q = np.asarray(q)
        a = np.asarray(a)
        # Compute the mass matrix
        M = pin.crba(self.model, self.data, q)
        # Compute the inertia torque M * a
        inertia_torque = M[:len(q), :len(q)].dot(a)
        return inertia_torque

    def get_Dynamics(self, q=None, v=None, a=None):
        """Get the full dynamics: tau = M(q)*a + C(q,v)*v + G(q). Returns np.ndarray."""
        if q is None:
            q = self.get_current_pos()
        if v is None:
            v = self.get_current_vel()
        if a is None:
            a = np.zeros(self.model.nv)
        # Make sure inputs are numpy arrays
        q = np.asarray(q)
        v = np.asarray(v)
        a = np.asarray(a)
        # Compute the full dynamics
        tau = pin.rnea(self.model, self.data, q, v, a)
        return tau

    def get_friction_compensation(self, vel=None, Fc=None, Fv=None, vel_threshold=0.01):
        """
        Compute the friction-compensation torque (Coulomb + viscous friction model).
        Returns np.ndarray.

        Args:
            vel: Joint-velocity array [N,] (rad/s). If None, the current velocity is used.
            Fc: Coulomb friction coefficients [N,] (Nm) - constant friction
            Fv: Viscous friction coefficients [N,] (Nm.s/rad) - velocity-dependent friction
            vel_threshold: Velocity threshold (rad/s). Below this value a special
                handling is used to avoid sign-flip chatter.

        Returns:
            tau_friction: Friction-compensation torque array (np.ndarray, [N,], Nm)

        Friction model:
            tau_friction = Fc * sign(vel) + Fv * vel
            When |vel| < vel_threshold only the viscous term is used to avoid
            sign-induced jumps.
        """
        # Get velocity
        if vel is None:
            vel = self.get_current_vel()
        else:
            vel = np.asarray(vel)

        # Make sure the friction coefficients are numpy arrays
        Fc = np.asarray(Fc)
        Fv = np.asarray(Fv)

        # Vectorized friction-compensation computation
        # Full friction model (Coulomb + viscous)
        full_friction = Fc * np.sign(vel) + Fv * vel

        # Viscous-only model for low speeds
        low_speed_friction = Fv * vel

        # Conditional select: use the low-speed model when |vel| < threshold,
        # otherwise use the full model.
        tau_friction = np.where(np.abs(vel) < vel_threshold, low_speed_friction, full_friction)

        return tau_friction

    #######################
    # Trajectory helpers
    #######################
    @staticmethod
    def septic_interpolation(start_pos, end_pos, duration, current_time):
        """7th-order polynomial trajectory interpolation (velocity, acceleration and jerk are continuous). Returns np.ndarray."""
        # Convert to numpy arrays
        start_pos = np.asarray(start_pos)
        end_pos = np.asarray(end_pos)

        if current_time <= 0:
            return start_pos, np.zeros_like(start_pos), np.zeros_like(start_pos)
        if current_time >= duration:
            return end_pos, np.zeros_like(end_pos), np.zeros_like(end_pos)

        # Normalized time
        t = current_time / duration
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        t6 = t5 * t
        t7 = t6 * t

        # 7th-order polynomial coefficients (position)
        a0 = 1 - 35*t4 + 84*t5 - 70*t6 + 20*t7
        a1 = 35*t4 - 84*t5 + 70*t6 - 20*t7

        # First-derivative coefficients (velocity)
        da0 = -140*t3 + 420*t4 - 420*t5 + 140*t6
        da1 = 140*t3 - 420*t4 + 420*t5 - 140*t6

        # Second-derivative coefficients (acceleration)
        dda0 = -420*t2 + 1680*t3 - 2100*t4 + 840*t5
        dda1 = 420*t2 - 1680*t3 + 2100*t4 - 840*t5

        # Vectorized position, velocity and acceleration
        pos = a0 * start_pos + a1 * end_pos
        vel = (da0 * start_pos + da1 * end_pos) / duration
        acc = (dda0 * start_pos + dda1 * end_pos) / (duration * duration)

        return pos, vel, acc

    @staticmethod
    def septic_interpolation_with_velocity(start_pos, end_pos, start_vel, end_vel, duration, current_time):
        """
        7th-order polynomial trajectory interpolation with specified start and end velocities.
        Returns np.ndarray. Enables smooth transitions with non-zero velocities.
        """
        # Convert to numpy arrays
        start_pos = np.asarray(start_pos)
        end_pos = np.asarray(end_pos)
        start_vel = np.asarray(start_vel)
        end_vel = np.asarray(end_vel)

        if current_time <= 0:
            return start_pos, start_vel, np.zeros_like(start_pos)
        if current_time >= duration:
            return end_pos, end_vel, np.zeros_like(end_pos)

        # Normalized time
        t = current_time / duration
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        t6 = t5 * t
        t7 = t6 * t

        # 7th-order polynomial coefficients (with velocity boundary conditions)
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5 + a6*t^6 + a7*t^7
        # Boundary conditions: p(0)=p0, p(1)=p1, v(0)=v0, v(1)=v1, a(0)=0, a(1)=0, j(0)=0, j(1)=0

        p0 = start_pos
        p1 = end_pos
        v0 = start_vel * duration  # Convert to normalized velocity
        v1 = end_vel * duration

        # Vectorized coefficient computation (satisfies 8 boundary conditions)
        a0 = p0
        a1 = v0
        a2 = np.zeros_like(p0)  # Initial acceleration is 0
        a3 = np.zeros_like(p0)  # Initial jerk is 0

        # Coefficients obtained by solving the linear system
        a4 = 35*(p1 - p0) - 20*v0 - 15*v1
        a5 = -84*(p1 - p0) + 45*v0 + 39*v1
        a6 = 70*(p1 - p0) - 36*v0 - 34*v1
        a7 = -20*(p1 - p0) + 10*v0 + 10*v1

        # Vectorized position
        pos = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5 + a6*t6 + a7*t7

        # Vectorized velocity (first derivative)
        vel = (a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4 + 6*a6*t5 + 7*a7*t6) / duration

        # Vectorized acceleration (second derivative)
        acc = (2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3 + 30*a6*t4 + 42*a7*t5) / (duration * duration)

        return pos, vel, acc

    @staticmethod
    def rotation_matrix_from_euler(roll, pitch, yaw):
        """
        Build a rotation matrix from Euler angles (RPY).

        Args:
            roll: Rotation about the X axis (rad)
            pitch: Rotation about the Y axis (rad)
            yaw: Rotation about the Z axis (rad)

        Returns:
            3x3 rotation matrix
        """
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        return rot.as_matrix()


if __name__ == "__main__":
    robot = Panthera()
