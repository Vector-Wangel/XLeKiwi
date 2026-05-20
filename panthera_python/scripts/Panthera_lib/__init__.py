"""
Panthera-HT arm control library

Provides high-level arm control interfaces, including kinematics, dynamics,
trajectory planning, and leader-follower teleoperation.

Example:
    from Panthera_lib import Panthera, TrajectoryRecorder

    # Create a robot instance
    robot = Panthera()

    # Get current joint angles
    joint_pos = robot.get_current_pos()

    # Per-joint position + velocity control (each joint set independently)
    robot.Joint_Pos_Vel([0, 0.5, -0.5, 0, 0.5, 0],
                        [0.5]*6, [10, 10, 10, 5, 5, 5])

    # Synchronized multi-joint arrival (all joints arrive together)
    robot.Joints_Sync_Arrival([0, 0.5, -0.5, 0, 0.5, 0],
                              duration=2.0)
"""

__version__ = "1.0.0"
__author__ = "HighTorque Robotics"

# Import main classes
from .Panthera import Panthera
from .recorder import Recorder as TrajectoryRecorder

# Public API
__all__ = [
    'Panthera',
    'TrajectoryRecorder',
]
