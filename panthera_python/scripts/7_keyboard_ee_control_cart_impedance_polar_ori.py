#!/usr/bin/env python3
"""
Keyboard Cartesian impedance control -- horizontal polar variant (Polar Cartesian Impedance Control)

Based on 11_keyboard_cartesian_impedance_control.py, with a horizontal polar wrapper added.

Architecture: two threads
  Control thread (2000Hz) -- FK + analytical Jacobian + Cartesian impedance -> joint torques
  Keyboard thread (250Hz) -- pygame detects key presses and updates the Cartesian target pose x_des via polar coordinates

Horizontal polar keyboard mapping:
  W/S: Radial motion (away/towards the base center), orientation unchanged
  A/D: Rotate about the base Z axis; position and orientation rotate together
  Q/E: Vertical (same as the original)

Control law:
  e_x = [p_des - p ; orientation_error_axis_angle(R_des, R)]
  x_dot = J(q) * dq
  F   = K_cart * e_x - B_cart * x_dot
  tau = J^T(JJ^T + lambda^2 I)^{-1} F + G(q) + f(dq) - B_joint * dq

Keyboard control (horizontal polar):
  Position: W/S=radial (away/towards base)  A/D=about Z axis (CCW/CW)  Q/E=vertical
  Orientation (EE frame): I/K=Pitch  J/L=Yaw  U/O=Roll
  Gripper: Z=close  X=open
  Space=print target pose  R=return home  Esc=exit
"""
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pinocchio as pin
import pygame
from Panthera_lib import Panthera

# --- Control parameters (ported from C++ pure_cartesian_impedance_control.cpp) ---
POS_STEP        = 0.001             # Per-frame position step (m)
ROT_STEP        = 0.2               # Per-frame rotation step (degrees)
POLAR_ANGLE_MAX = np.radians(0.5)   # Max polar rotation step (rad/frame), prevents jumps when r is near 0
CTRL_FREQ_IMP = 1500    # Control thread frequency (Hz)
CTRL_FREQ_KEY = 250     # Keyboard thread frequency (Hz)

MAX_TORQUE = [21.0, 36.0, 36.0, 30.0, 10.0, 10.0]
JOINT_VEL  = [0.5] * 6

# Cartesian space stiffness (lowered and balanced to avoid J^T amplification chatter)
K_POS = np.array([20.0, 20.0, 60.0])     # N/m, was [10,20,30]
K_ROT = np.array([50.0, 50.0, 40.0])     # Nm/rad, was [10,20,20], significantly lowered
K_CART = np.concatenate([K_POS, K_ROT])

# Cartesian space damping (keep small! Large values amplify velocity noise via J^T@B@J)
B_POS = np.array([0.01, 0.01, 0.01])     # N.s/m, only a small amount of Cartesian damping
B_ROT = np.array([0.01, 0.01, 0.01])     # Nm.s/rad, main damping comes from JOINT_DAMPING
B_CART = np.concatenate([B_POS, B_ROT])

# DLS singularity regularization parameter
LAMBDA_DAMP = 0.05

# Joint torque cap
TAU_LIMIT = np.array([20.0, 30.0, 30.0, 20.0, 10.0, 10.0])

# Joint-space damping (main damping source, directly suppresses joint velocity without J^T amplification)
# Large joints can take more force -> higher damping; small-torque joints -> lower damping
JOINT_DAMPING = np.array([1.2, 2, 2, 2, 0.8, 0.6])

# Joint friction compensation
IMP_Fc         = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01])
IMP_Fv         = np.array([0.03, 0.03, 0.03, 0.03, 0.01, 0.01])
IMP_VEL_THRESH = 0.02

# tool offset
TOOL_OFFSET = np.array([0.165, 0.0, 0.0])

# gripper control parameters
GRIPPER_STEP        = 0.02
GRIPPER_KP          = 8.0
GRIPPER_KD          = 0.5
GRIPPER_MIN_DEFAULT = 0.0
GRIPPER_MAX_DEFAULT = 1.6

# Joint limit safety margin (margin from hard limits, in rad)
JOINT_LIMIT_MARGIN = 0.1
# How long the limit warning stays visible (seconds)
JOINT_LIMIT_WARN_DURATION = 1.0

# Initial end-effector pose
HOME_POS       = [0.24, 0.0, 0.15]
HOME_ROT_EULER = (0.0, np.pi / 2, 0.0)


# ─── Math utilities ────────────────────────────────────────────────

def skew(v):
    """skew-symmetric matrix [v]×"""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def orientation_error_axis_angle(R_des, R_cur):
    """
    Orientation error (axis-angle), matches C++ computeOrientationError() exactly.

    R_err = R_cur^T @ R_des -> angle*axis (current frame) -> rotated by R_cur into the world frame
    """
    R_err = R_cur.T @ R_des
    rot = Rot.from_matrix(R_err)
    rotvec = rot.as_rotvec()
    return R_cur @ rotvec


def _build_pin_q(robot, joint_angles):
    """Convert joint-angle array to a pinocchio q vector."""
    q = np.zeros(robot.model.nq)
    for i, name in enumerate(robot.joint_names):
        jid = robot.model.getJointId(name)
        q[robot.model.joints[jid].idx_q] = joint_angles[i]
    return q


def compute_fk_and_jacobian(robot, data, joint_angles):
    """
    Single pinocchio call returns both TCP pose and the analytical Jacobian (tool offset already accounted for).
    """
    q = _build_pin_q(robot, joint_angles)
    pin.computeJointJacobians(robot.model, data, q)

    last_jid = robot.model.getJointId(robot.joint_names[-1])
    T_last = data.oMi[last_jid]
    R_last = T_last.rotation
    p_last = T_last.translation

    r_world = R_last @ TOOL_OFFSET
    p_tcp = (p_last + r_world).copy()
    R_tcp = R_last.copy()

    J_full = pin.getJointJacobian(
        robot.model, data, last_jid,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    J_tcp = J_full.copy()
    J_tcp[:3, :] -= skew(r_world) @ J_full[3:, :]

    cols = [robot.model.joints[robot.model.getJointId(n)].idx_v
            for n in robot.joint_names]
    J6 = J_tcp[:, cols]

    return p_tcp, R_tcp, J6


def print_pose(pos, rot, label="End-effector pose"):
    euler = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
    print(f"\n[{label}]")
    print(f"  pos (m):  X={pos[0]:+.4f}  Y={pos[1]:+.4f}  Z={pos[2]:+.4f}")
    print(f"  pose (deg):  Roll={euler[0]:+.1f}  Pitch={euler[1]:+.1f}  Yaw={euler[2]:+.1f}")


# ─── Main ──────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Keyboard Cartesian impedance (horizontal polar + axis-angle)║
╠══════════════════════════════════════════════════════════════╣
║  Position (horizontal polar):                                ║
║    W / S   ->  Radial away / towards base                    ║
║    A / D   ->  Orbit base Z axis CCW / CW (orientation sync) ║
║    Q / E   ->  Z axis up / down                              ║
╠══════════════════════════════════════════════════════════════╣
║  Orientation (EE frame):                                      ║
║    I / K   ->  Pitch about Y axis (+/-)                      ║
║    J / L   ->  Yaw about Z axis (+/-)                        ║
║    U / O   ->  Roll about X axis (+/-)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Other:                                                       ║
║    Z / X   ->  gripper close / open                           ║
║    Space   ->  Print current Cartesian target pose            ║
║    R       ->  Return to initial pose                         ║
║    Esc     ->  Exit program                                   ║
╚══════════════════════════════════════════════════════════════╝
NOTE: focus the pygame window that just opened
""")

    # ── Initialize pygame ────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((520, 182))
    pygame.display.set_caption("Cartesian Impedance (Polar) — focus here!")
    font = pygame.font.SysFont(None, 22)

    # ── Initialize robot ─────────────────────────────────────────
    robot = Panthera()
    zero_pos = [0.0] * robot.motor_count
    _z6 = [0.0] * 6

    ctrl_data = robot.model.createData()

    # 1. Return to zero
    print("Init: return joints to zero...")
    robot.Joint_Pos_Vel(zero_pos, JOINT_VEL, MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    # 2. Move to the initial pose
    home_rot = robot.rotation_matrix_from_euler(*HOME_ROT_EULER)
    init_joints = robot.inverse_kinematics(HOME_POS, home_rot, robot.get_current_pos())
    if init_joints is None:
        print("[Warning] IK for the initial pose failed, holding zero position")
    else:
        print(f"Moving to initial joint position, end-effector target: {HOME_POS}")
        robot.moveJ(init_joints, duration=3.0, max_tqu=MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    # 3. Raise the end-effector 0.1 m along Z
    fk_after_home = robot.forward_kinematics()
    p_lift = np.array(fk_after_home['position'])
    R_lift = np.array(fk_after_home['rotation'])
    p_lift[2] += 0.1

    print(f"Raising end-effector 0.1 m along Z, target Z = {p_lift[2]:.3f} m ...")
    success = robot.moveL(
        target_position=p_lift,
        target_rotation=R_lift,
        duration=2.0,
        use_spline=True
    )
    if not success:
        print("[Warning] moveL lift failed; using current position as Cartesian control start")
    time.sleep(0.5)

    # 4. Read current pose as the initial Cartesian target
    q0 = np.array(robot.get_current_pos())
    p0, R0, _ = compute_fk_and_jacobian(robot, robot.data, q0)
    print_pose(p0, R0, "Cartesian impedance control start pose (axis-angle)")

    # gripper limit
    gripper_min = robot.gripper_limits['lower'] if robot.gripper_limits else GRIPPER_MIN_DEFAULT
    gripper_max = robot.gripper_limits['upper'] if robot.gripper_limits else GRIPPER_MAX_DEFAULT
    print(f"gripper limit: [{gripper_min:.2f}, {gripper_max:.2f}] rad")

    # Joint limits (loaded from robot config, with a safety margin)
    jl_lower = np.array(robot.joint_limits['lower']) + JOINT_LIMIT_MARGIN
    jl_upper = np.array(robot.joint_limits['upper']) - JOINT_LIMIT_MARGIN
    print(f"Joint limits (with {JOINT_LIMIT_MARGIN} rad safety margin):")
    for i in range(len(jl_lower)):
        print(f"  J{i+1}: [{jl_lower[i]:+.2f}, {jl_upper[i]:+.2f}] rad")

    # ── Cross-thread shared state ───────────────────────────────────────
    lock = threading.Lock()
    x_des_pos = p0.copy()
    x_des_rot = R0.copy()
    gripper_des = float(np.clip(robot.get_current_pos_gripper(), gripper_min, gripper_max))
    ctrl_enabled = threading.Event()
    stop_event = threading.Event()
    ctrl_enabled.set()

    # Joint limits: last feasible target + warning message
    last_feasible_pos = p0.copy()
    last_feasible_rot = R0.copy()
    joint_limit_warn = ""        # When non-empty, GUI shows the warning
    joint_limit_warn_time = 0.0  # Timestamp when warning was triggered

    disp_err = np.zeros(6)
    disp_gripper = np.zeros(2)
    disp_lock = threading.Lock()

    # ── Cartesian impedance control thread ──────────────────────────────────
    def impedance_loop():
        nonlocal last_feasible_pos, last_feasible_rot
        nonlocal joint_limit_warn, joint_limit_warn_time
        dt = 1.0 / CTRL_FREQ_IMP
        while not stop_event.is_set():
            t0 = time.time()

            if ctrl_enabled.is_set():
                q = np.array(robot.get_current_pos())
                dq = np.array(robot.get_current_vel())

                # --- Joint limit check ---
                violated = (q < jl_lower) | (q > jl_upper)
                if np.any(violated):
                    # Build warning message
                    parts = []
                    for i in range(len(q)):
                        if violated[i]:
                            side = "lower" if q[i] < jl_lower[i] else "upper"
                            parts.append(f"J{i+1}{side}({q[i]:+.2f})")
                    with lock:
                        joint_limit_warn = " ".join(parts)
                        joint_limit_warn_time = time.time()
                        # Roll back to the last feasible target
                        x_des_pos[:] = last_feasible_pos
                        x_des_rot[:] = last_feasible_rot
                else:
                    # All joints within safe range; record current target as feasible
                    with lock:
                        last_feasible_pos[:] = x_des_pos
                        last_feasible_rot[:] = x_des_rot

                # FK + Jacobian
                p_cur, R_cur, J = compute_fk_and_jacobian(robot, ctrl_data, q)

                with lock:
                    p_des = x_des_pos.copy()
                    R_des = x_des_rot.copy()
                    g_des = gripper_des

                # 6D Cartesian error (axis-angle orientation error, matches C++)
                e_pos = p_des - p_cur
                e_rot = orientation_error_axis_angle(R_des, R_cur)
                e_x = np.concatenate([e_pos, e_rot])

                # Cartesian velocity
                dx = J @ dq

                # Cartesian spring-damper: F = K * e_x - B * x_dot
                F = K_CART * e_x - B_CART * dx

                # Damped least-squares mapping
                JJT = J @ J.T
                alpha = np.linalg.solve(JJT + LAMBDA_DAMP**2 * np.eye(6), F)
                tor_cart = J.T @ alpha

                # Joint-space damping (directly suppresses joint velocity, no J^T amplification)
                tor_joint_damp = -JOINT_DAMPING * dq

                # Gravity + friction compensation
                tor_gra = np.array(robot.get_Gravity())
                tor_fri = np.array(robot.get_friction_compensation(
                    dq, IMP_Fc, IMP_Fv, IMP_VEL_THRESH))

                tor = np.clip(tor_cart + tor_joint_damp + tor_gra + tor_fri,
                              -TAU_LIMIT, TAU_LIMIT)

                # gripper + arm torque commands
                robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                    g_des, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)
                robot.pos_vel_tqe_kp_kd(_z6, _z6, tor.tolist(), _z6, _z6)

                # Update display buffer
                g_actual = robot.get_current_pos_gripper()
                with disp_lock:
                    disp_err[:] = e_x
                    disp_gripper[:] = [g_des, g_actual]

            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    ctrl_thread = threading.Thread(target=impedance_loop, daemon=True)
    ctrl_thread.start()

    print("\nKeyboard control active (focus the pygame window)...\n")

    rot_step_rad = np.radians(ROT_STEP)
    # W/S/A/D are handled by polar logic, not in this map
    PYGAME_KEY_MAP = {
        pygame.K_q: (np.array([0, 0, POS_STEP]), None),
        pygame.K_e: (np.array([0, 0, -POS_STEP]), None),
        pygame.K_i: (np.zeros(3), (0, rot_step_rad, 0)),
        pygame.K_k: (np.zeros(3), (0, -rot_step_rad, 0)),
        pygame.K_j: (np.zeros(3), (0, 0, rot_step_rad)),
        pygame.K_l: (np.zeros(3), (0, 0, -rot_step_rad)),
        pygame.K_u: (np.zeros(3), (rot_step_rad, 0, 0)),
        pygame.K_o: (np.zeros(3), (-rot_step_rad, 0, 0)),
    }

    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            # ── Event handling ──────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        with lock:
                            p_print = x_des_pos.copy()
                            R_print = x_des_rot.copy()
                        print_pose(p_print, R_print, "Current Cartesian target pose")

                    elif event.key == pygame.K_r:
                        ctrl_enabled.clear()
                        print("\nreturn to initial pose...")
                        home_q = init_joints if init_joints is not None else zero_pos
                        robot.moveJ(home_q, duration=2.0, max_tqu=MAX_TORQUE, iswait=True)
                        q_now = np.array(robot.get_current_pos())
                        p_now, R_now, _ = compute_fk_and_jacobian(
                            robot, robot.data, q_now)
                        with lock:
                            x_des_pos[:] = p_now
                            x_des_rot[:] = R_now
                        print_pose(p_now, R_now, "Returned to initial pose")
                        ctrl_enabled.set()

            # ── key held → update Cartesian target ────────────────────
            if ctrl_enabled.is_set():
                keys = pygame.key.get_pressed()
                combined_pos = np.zeros(3)
                first_rot_euler = None
                any_pressed = False

                for k, (dp, rot_euler) in PYGAME_KEY_MAP.items():
                    if keys[k]:
                        combined_pos += dp
                        if rot_euler is not None and first_rot_euler is None:
                            first_rot_euler = rot_euler
                        any_pressed = True

                # --- Polar handling (W/S=radial, A/D=rotate about Z) ---
                with lock:
                    px, py = x_des_pos[0], x_des_pos[1]
                r_horiz = np.hypot(px, py)

                if keys[pygame.K_a] or keys[pygame.K_d]:
                    # A/D: Rotate position + orientation about base Z axis
                    dtheta = POS_STEP / max(r_horiz, 0.01)
                    dtheta = min(dtheta, POLAR_ANGLE_MAX)
                    if keys[pygame.K_d]:
                        dtheta = -dtheta
                    c, s = np.cos(dtheta), np.sin(dtheta)
                    Rz = np.array([[c, -s, 0.0],
                                   [s,  c, 0.0],
                                   [0.0, 0.0, 1.0]])
                    with lock:
                        x_des_pos[:] = Rz @ x_des_pos
                        x_des_rot[:] = Rz @ x_des_rot
                    any_pressed = True

                if keys[pygame.K_w] or keys[pygame.K_s]:
                    # W/S: Radial motion in the horizontal plane (orientation unchanged)
                    if r_horiz > 0.001:
                        radial_dir = np.array([px, py, 0.0]) / r_horiz
                    else:
                        radial_dir = np.array([1.0, 0.0, 0.0])
                    dr = POS_STEP if keys[pygame.K_w] else -POS_STEP
                    with lock:
                        x_des_pos += dr * radial_dir
                    any_pressed = True

                if any_pressed:
                    with lock:
                        # Non-polar translations such as Q/E
                        x_des_pos += combined_pos
                        if first_rot_euler is not None:
                            dR = robot.rotation_matrix_from_euler(*first_rot_euler)
                            new_rot = x_des_rot @ dR
                            U, _, Vt = np.linalg.svd(new_rot)
                            x_des_rot[:] = U @ Vt

                # gripper control
                if keys[pygame.K_z] or keys[pygame.K_x]:
                    with lock:
                        if keys[pygame.K_z]:
                            gripper_des = max(gripper_min, gripper_des - GRIPPER_STEP)
                        if keys[pygame.K_x]:
                            gripper_des = min(gripper_max, gripper_des + GRIPPER_STEP)

            # --- Refresh pygame window ---
            screen.fill((20, 20, 30))
            screen.blit(font.render(
                "W/S=radial  A/D=orbit  Q/E=Z  I/K/J/L/U/O  Z/X  R  Esc",
                True, (160, 190, 220)), (10, 6))

            with disp_lock:
                err_snap = disp_err.copy()
                g_snap = disp_gripper.copy()
            ep_mm = err_snap[:3] * 1000.0
            er_deg = np.degrees(err_snap[3:])
            ep_n = np.linalg.norm(ep_mm)
            er_n = np.linalg.norm(er_deg)

            def _err_color(val, lo, hi):
                if val < lo:  return (80, 220, 80)
                if val < hi:  return (220, 200, 60)
                return (220, 70, 70)

            screen.blit(font.render(
                f"Pos err(mm) X:{ep_mm[0]:+6.1f}  Y:{ep_mm[1]:+6.1f}  Z:{ep_mm[2]:+6.1f}  |e|:{ep_n:5.1f}",
                True, _err_color(ep_n, 3, 15)), (10, 36))
            screen.blit(font.render(
                f"Rot err(deg) X:{er_deg[0]:+6.1f}  Y:{er_deg[1]:+6.1f}  Z:{er_deg[2]:+6.1f}  |e|:{er_n:5.1f}",
                True, _err_color(er_n, 2, 8)), (10, 58))
            with lock:
                g_des_disp = gripper_des
                polar_pos = x_des_pos.copy()
                warn_text = joint_limit_warn
                warn_t = joint_limit_warn_time
            g_pct = (g_des_disp - gripper_min) / max(gripper_max - gripper_min, 1e-6) * 100
            polar_r = np.hypot(polar_pos[0], polar_pos[1])
            polar_theta = np.degrees(np.arctan2(polar_pos[1], polar_pos[0]))
            screen.blit(font.render(
                f"Gripper des:{g_des_disp:.3f} act:{g_snap[1]:.3f} {g_pct:.0f}%"
                f"   Polar r={polar_r:.3f}m  \u03b8={polar_theta:+.1f}\u00b0  z={polar_pos[2]:.3f}m",
                True, (160, 200, 240)), (10, 80))
            screen.blit(font.render(
                f"ctrl={'ON ' if ctrl_enabled.is_set() else 'OFF'}   "
                f"K_pos=[{K_POS[0]:.0f},{K_POS[1]:.0f},{K_POS[2]:.0f}]  "
                f"K_rot=[{K_ROT[0]:.0f},{K_ROT[1]:.0f},{K_ROT[2]:.0f}]  "
                f"Polar+AxisAngle",
                True, (130, 130, 150)), (10, 103))

            # Joint limit warning (blinking red, visible for JOINT_LIMIT_WARN_DURATION seconds)
            if warn_text and (time.time() - warn_t) < JOINT_LIMIT_WARN_DURATION:
                blink = int(time.time() * 4) % 2 == 0  # 4Hz blink
                if blink:
                    screen.blit(font.render(
                        f"JOINT LIMIT  {warn_text}",
                        True, (255, 60, 60)), (10, 125))

            pygame.display.flip()

            clock.tick(CTRL_FREQ_KEY)

    finally:
        stop_event.set()
        ctrl_thread.join(timeout=1.0)
        pygame.quit()
        print("\n\nExited (motors will hold current position until timeout)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as e:
        print(f"\nerror: {e}")
        import traceback
        traceback.print_exc()
