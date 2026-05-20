#!/usr/bin/env python3
"""
Inverse kinematics (IK) verification program.
Two verification methods:
1. Compute FK from the current joint position to get the EE pose, then use
   that EE pose for IK verification (you can drag the arm to see how IK
   behaves).
2. Specify a target position and orientation explicitly for IK verification.
"""
import time
import numpy as np
from Panthera_lib import Panthera

def main():
    # Method 1: compute FK from current joint position to get EE pose, then
    # use that EE pose for IK verification.
    # Send a state-request frame so motors push their feedback
    robot.send_get_motor_state_cmd()
    current_angles = robot.get_current_pos()
    fk = robot.forward_kinematics(current_angles)
    if not fk:
        return
    ik_pos = fk['position']
    ik_rot = fk['rotation']
    # Use current joint angles minus 0.1 as the initial guess (vector op)
    init_q = current_angles - 0.1
    # # Or use the zero pose as the initial guess
    # init_q = np.zeros(robot.motor_count)
    # Solve IK using current EE position and orientation
    solved_angles = robot.inverse_kinematics(ik_pos, ik_rot, init_q)
    if solved_angles is not None:
        print(f"\ncurrent joints: {current_angles}")
        print(f"IK joints     : {solved_angles}")
        # Compute error (vector op)
        errors = np.abs(current_angles - solved_angles)
        max_error = np.max(errors)
        print(f"max error: {max_error:.4f} rad")
    time.sleep(0.5)

    # # Method 2: specify a target position and orientation for IK verification
    # target_pos = [0.3, 0.2, 0.2]
    # target_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # identity as example
    # # IK computation
    # init_q = [0.0] * robot.motor_count
    # ik_angles = robot.inverse_kinematics(target_pos, target_rot, init_q)
    # if ik_angles:
    #     # Use the IK result for FK verification
    #     fk_verify = robot.forward_kinematics(ik_angles)
    #     if fk_verify:
    #         # Position error
    #         pos_error = np.linalg.norm(np.array(target_pos) - np.array(fk_verify['position']))
    #         # Orientation error (Frobenius norm of rotation matrix difference)
    #         rot_error = np.linalg.norm(target_rot - fk_verify['rotation'], 'fro')
    #         print(f"\ntarget position: {target_pos}")
    #         print(f"verified position: {[f'{p:.3f}' for p in fk_verify['position']]}")
    #         print(f"position error: {pos_error:.4f} m")
    #         print(f"orientation error: {rot_error:.4f}")
    # time.sleep(0.5)


if __name__ == "__main__":
    robot = Panthera()
    try:
        while(1):
            main()
    except KeyboardInterrupt:
        robot.set_stop()
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")
