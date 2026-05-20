#!/usr/bin/env python3
"""
Joint impedance control + friction compensation.
Effectively equivalent to six-joint PD control plus a feed-forward torque.
(Compare with 1_PD_control.py to see the difference.)
"""
import time
import numpy as np
from Panthera_lib import Panthera

def main():
    # At startup, linearly interpolate from q_start to q_des to avoid an
    # impulsive torque the moment impedance control kicks in.
    t_elapsed = time.time() - t_start
    alpha = min(t_elapsed / interp_duration, 1.0)
    q_des_interp = (1 - alpha) * q_start + alpha * q_des

    # Compute impedance control output torque
    q_current = robot.get_current_pos()
    vel_current = robot.get_current_vel()
    tor_impedance = K * (q_des_interp - q_current) + B * (v_des - vel_current)
    # Total torque (with gravity-compensation feed-forward torque)
    G = robot.get_Gravity()
    f = robot.get_friction_compensation(vel_current, Fc, Fv, vel_threshold)
    tor = tor_impedance + G
    # Torque cap (per motor spec)
    tau_limit = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])
    tor = np.clip(tor, -tau_limit, tau_limit)
    robot.pos_vel_tqe_kp_kd(zero_pos, zero_vel, tor, zero_kp, zero_kd)
    print(f"interp progress: {alpha*100:.1f}%, impedance torque: {[f'{t:.3f}' for t in tor_impedance]}, \ngravity compensation torque: {[f'{t:.3f}' for t in G]}, \ntotal torque: {[f'{t:.3f}' for t in tor]}")
    time.sleep(0.002)
    #Motors auto-power-off on exit. Be careful.

if __name__ == "__main__":
    robot = Panthera()
    # Stiffness and damping coefficients
    K = np.array([4.0, 10.0, 10.0, 2.0, 2.0, 1.0])
    B = np.array([0.5, 0.8, 0.8, 0.2, 0.2, 0.1])
    # When both are zero, this degenerates to gravity compensation mode
    # K = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_des = np.array([0.0, 0.7, 0.7, -0.1, 0.0, 0.0]  )  # desired target position
    # q_des = np.zeros(6)  # desired target position
    v_des = np.zeros(6) # desired target velocity is zero
    # Create zero position/velocity arrays
    zero_kp = [0.0] * robot.motor_count
    zero_kd = [0.0] * robot.motor_count
    zero_pos = [0.0]*6
    zero_vel = [0.0]*6
    q = np.array([])
    vel = np.array([])

    # Friction params are loaded from the friction section of the yaml
    # (auto-switched per arm type).
    Fc = robot.Fc
    Fv = robot.Fv
    vel_threshold = robot.vel_threshold

    # Interpolation start: read current pose as the start before impedance
    # control engages, then transition linearly to q_des over 3 seconds.
    q_start = robot.get_current_pos()
    t_start = time.time()
    interp_duration = 3.0  # interpolation duration (seconds)

    try:
        while(1):
            main()
    except KeyboardInterrupt:
        # Without this line, motors power down when the script stops
        # robot.set_stop()
        print("\n\ninterrupted")
        print("\n\nall motors stopped")
    except Exception as e:
        print(f"\nerror: {e}")
