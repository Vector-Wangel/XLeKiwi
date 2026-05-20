#!/usr/bin/env python3
"""
Keyboard end-effector control - pygame three-thread async (Keyboard EE Control - pygame Async)

Architecture:
  Keyboard thread (50Hz)  -- pygame key.get_pressed() reads key state and pushes
                             incremental commands to ik_queue
  IK thread               -- consumes ik_queue, computes inverse kinematics, updates shared q_des
  Control thread (1000Hz) -- Joint impedance control with strict timing, unaffected by IK latency

pygame advantages:
  - key.get_pressed() reads hardware key state directly, no OS key-repeat startup delay
  - Long press responds immediately and continuously; releasing stops it instantly with no residual motion

[Note] You must focus the pygame window that opens for keyboard input to take effect.

Position control (base frame, 1mm * 50Hz = 5cm/s):
  W / S  ->  X axis forward / backward
  A / D  ->  Y axis left / right
  Q / E  ->  Z axis up / down

Orientation control (EE frame, 0.04 deg * 50Hz = 2 deg/s):
  I / K  ->  Pitch about Y axis (+/-)
  J / L  ->  Yaw about Z axis (+/-)
  U / O  ->  Roll about X axis (+/-)

Gripper control (hold continuously; stops updating once a limit is reached):
  Z      ->  Gripper close
  X      ->  Gripper open

Other:
  Space  ->  Print current end-effector pose
  R      ->  Return to initial pose
  Esc    ->  Exit program
"""
import queue
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import pygame
from Panthera_lib import Panthera

# --- Control parameters ---
POS_STEP      = 0.001   # Per-frame position step (m)
ROT_STEP      = 0.1    # Per-frame rotation step (degrees)
CTRL_FREQ_IMP = 1500    # Impedance control thread frequency (Hz)
CTRL_FREQ_KEY = 200      # Keyboard polling frequency (Hz)

MAX_TORQUE = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
JOINT_VEL  = [0.5] * 6

# --- Joint impedance control parameters ---
# tau = K*(q_des - q) + B*(0 - dq) + G + f
IMP_K         = np.array([20.0,  25.0, 25.0,  20.0,  10.0, 5.0])
IMP_B         = np.array([1,   1.0,  1.0,   0.8,   0.4, 0.2])
IMP_TAU_LIMIT = np.array([10.0, 20.0, 20.0,  10.0,   5.0, 5.0])
IMP_Fc        = np.array([0.05, 0.05, 0.05,  0.05,   0.0, 0.0])
IMP_Fv        = np.array([0.03, 0.03, 0.03,  0.03,   0.0, 0.0])
IMP_VEL_THRESH = 0.02

# --- Gripper control parameters ---
GRIPPER_STEP        = 0.02   # Per-frame keyboard step (rad), 200Hz * 0.02 ~= 4 rad/s max speed
GRIPPER_KP          = 8.0    # Position stiffness
GRIPPER_KD          = 0.5    # Velocity damping
GRIPPER_MIN_DEFAULT = 0.0
GRIPPER_MAX_DEFAULT = 1.6

# Initial end-effector pose
HOME_POS       = [0.24, 0.0, 0.15]
HOME_ROT_EULER = (0.0, np.pi / 2, 0.0)  # End-effector pointing forward


# --- Utility functions ---

def print_pose(pos, rot, label: str = "End-effector pose"):
    euler = R.from_matrix(rot).as_euler('xyz', degrees=True)
    print(f"\n[{label}]")
    print(f"  pos (m):  X={pos[0]:+.4f}  Y={pos[1]:+.4f}  Z={pos[2]:+.4f}")
    print(f"  pose (deg):  Roll={euler[0]:+.1f}  Pitch={euler[1]:+.1f}  Yaw={euler[2]:+.1f}")


# ─── Main ──────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║  Keyboard end-effector control - pygame async (3 threads)║
╠══════════════════════════════════════════════════════════╣
║  Position control (base frame):                          ║
║    W / S   ->  X axis forward / backward                 ║
║    A / D   ->  Y axis left / right                       ║
║    Q / E   ->  Z axis up / down                          ║
╠══════════════════════════════════════════════════════════╣
║  Orientation (EE frame):                                  ║
║    I / K   ->  Pitch about Y axis (+/-)                   ║
║    J / L   ->  Yaw about Z axis (+/-)                     ║
║    U / O   ->  Roll about X axis (+/-)                    ║
╠══════════════════════════════════════════════════════════╣
║  Gripper (hold; stops updating after reaching a limit):   ║
║    Z       ->  Gripper close                              ║
║    X       ->  Gripper open                               ║
╠══════════════════════════════════════════════════════════╣
║  Other:                                                   ║
║    Space   ->  Print current end-effector pose            ║
║    R       ->  Return to initial pose                     ║
║    Esc     ->  Exit program                               ║
╚══════════════════════════════════════════════════════════╝
NOTE: focus the pygame window that just opened so keyboard input takes effect
""")

    # ── Initialize pygame ────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((480, 92))
    pygame.display.set_caption("EE Ctrl — focus here!")
    font = pygame.font.SysFont(None, 22)

    # ── Initialize robot ─────────────────────────────────────────
    robot    = Panthera()
    zero_pos = [0.0] * robot.motor_count
    _z6      = [0.0] * 6

    # 1. Return to zero
    print("Init: return joints to zero...")
    robot.Joint_Pos_Vel(zero_pos, JOINT_VEL, MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    # 2. Move to the initial working pose
    home_rot    = robot.rotation_matrix_from_euler(*HOME_ROT_EULER)
    init_joints = robot.inverse_kinematics(HOME_POS, home_rot, robot.get_current_pos())
    if init_joints is None:
        print("[Warning] IK for initial pose failed, using zero position as start")
    else:
        print(f"Moving to initial end-effector pose: {HOME_POS}")
        robot.moveJ(init_joints, duration=3.0, max_tqu=MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    # 3. Read initial state
    fk         = robot.forward_kinematics()
    target_pos = np.array(fk['position'])
    target_rot = np.array(fk['rotation'])
    print_pose(target_pos, target_rot, "Initial end-effector pose")

    # Gripper limits (read from config if available, otherwise use defaults)
    gripper_min = robot.gripper_limits['lower'] if robot.gripper_limits else GRIPPER_MIN_DEFAULT
    gripper_max = robot.gripper_limits['upper'] if robot.gripper_limits else GRIPPER_MAX_DEFAULT
    print(f"gripper limit: [{gripper_min:.2f}, {gripper_max:.2f}] rad  (Z=close, X=open)")

    # --- Cross-thread shared state ---
    lock         = threading.Lock()
    q_des        = np.array(robot.get_current_pos())   # Written by IK thread, read by control thread
    q_cur_cache  = q_des.copy()                         # Written by control thread, read by IK thread (initial guess)
    # Gripper target (rad): written by keyboard thread, read by control thread, stops updating at limits
    gripper_des  = float(np.clip(robot.get_current_pos_gripper(), gripper_min, gripper_max))
    ctrl_enabled = threading.Event()   # set=running, clear=paused (during moveJ)
    stop_event   = threading.Event()   # set=fully exit
    ctrl_enabled.set()

    # maxsize=1: if IK can't keep up, drop the newest command to avoid backlog
    ik_queue   = queue.Queue(maxsize=1)
    warn_queue = queue.Queue()

    rot_step_rad = np.radians(ROT_STEP)

    PYGAME_KEY_MAP = {
        pygame.K_w: (np.array([ POS_STEP,  0,  0]),  None),
        pygame.K_s: (np.array([-POS_STEP,  0,  0]),  None),
        pygame.K_a: (np.array([0,  POS_STEP,  0]),  None),
        pygame.K_d: (np.array([0, -POS_STEP,  0]),  None),
        pygame.K_q: (np.array([0,  0,  POS_STEP]),  None),
        pygame.K_e: (np.array([0,  0, -POS_STEP]),  None),
        pygame.K_i: (np.zeros(3), ( 0,  rot_step_rad,  0)),
        pygame.K_k: (np.zeros(3), ( 0, -rot_step_rad,  0)),
        pygame.K_j: (np.zeros(3), ( 0,  0,  rot_step_rad)),
        pygame.K_l: (np.zeros(3), ( 0,  0, -rot_step_rad)),
        pygame.K_u: (np.zeros(3), ( rot_step_rad,  0,  0)),
        pygame.K_o: (np.zeros(3), (-rot_step_rad,  0,  0)),
    }

    # --- Impedance control thread (1000Hz, strict timing) ---
    def impedance_loop():
        dt = 1.0 / CTRL_FREQ_IMP
        while not stop_event.is_set():
            t0 = time.time()
            if ctrl_enabled.is_set():
                q  = np.array(robot.get_current_pos())
                dq = np.array(robot.get_current_vel())
                with lock:
                    q_cur_cache[:] = q
                    target  = q_des.copy()
                    g_des   = gripper_des
                tor_imp = IMP_K * (target - q) + IMP_B * (-dq)
                tor_gra = np.array(robot.get_Gravity())
                tor_fri = np.array(robot.get_friction_compensation(
                    dq, IMP_Fc, IMP_Fv, IMP_VEL_THRESH))
                tor = np.clip(tor_imp + tor_gra + tor_fri, -IMP_TAU_LIMIT, IMP_TAU_LIMIT)
                # Gripper command is written into the motor buffer and sent along with the arm motor_send_cmd (no extra packet)
                robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                    g_des, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)
                robot.pos_vel_tqe_kp_kd(_z6, _z6, tor, _z6, _z6)
            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # --- IK worker thread ---
    # ik_state is only read/written by the IK thread, no lock needed
    ik_state = {'pos': target_pos.copy(), 'rot': target_rot.copy()}

    def ik_worker():
        while not stop_event.is_set():
            try:
                cmd = ik_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cmd[0] == 'move':
                _, delta_pos, rot_euler = cmd
                new_pos = ik_state['pos'] + delta_pos

                if rot_euler is not None:
                    new_rot = ik_state['rot'] @ robot.rotation_matrix_from_euler(*rot_euler)
                    # SVD re-orthogonalization to prevent floating-point drift out of SO(3)
                    U, _, Vt = np.linalg.svd(new_rot)
                    new_rot = U @ Vt
                else:
                    new_rot = ik_state['rot']

                # Use the previous IK target (not the actual position) as the initial guess to keep solutions continuous
                with lock:
                    q_init = q_des.copy()
                joint_angles = robot.inverse_kinematics(new_pos, new_rot, q_init)

                if joint_angles is not None:
                    ik_state['pos'] = new_pos
                    ik_state['rot'] = new_rot
                    with lock:
                        q_des[:] = joint_angles
                else:
                    warn_queue.put(
                        f"[Warning] IK failed, target "
                        f"[{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}] "
                        f"is outside the workspace, motion stopped"
                    )

            elif cmd[0] == 'reset':
                _, new_pos, new_rot = cmd
                ik_state['pos'] = new_pos.copy()
                ik_state['rot'] = new_rot.copy()

    ctrl_thread = threading.Thread(target=impedance_loop, daemon=True)
    ik_thread   = threading.Thread(target=ik_worker,      daemon=True)
    ctrl_thread.start()
    ik_thread.start()

    print("\nKeyboard control active (focus the pygame window)...\n")

    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            # --- Handle pygame events (one-shot actions) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        with lock:
                            q_snap = q_cur_cache.copy()
                        fk = robot.forward_kinematics(q_snap)
                        print_pose(fk['position'], fk['rotation'], "Current end-effector pose")

                    elif event.key == pygame.K_r:
                        ctrl_enabled.clear()
                        print("\nreturn to initial pose...")
                        home_q = init_joints if init_joints is not None else zero_pos
                        robot.moveJ(home_q, duration=2.0, max_tqu=MAX_TORQUE, iswait=True)
                        with lock:
                            q_des[:] = robot.get_current_pos()
                        fk = robot.forward_kinematics()
                        new_pos = np.array(fk['position'])
                        new_rot = np.array(fk['rotation'])
                        # Clear ik_queue and sync IK state
                        while not ik_queue.empty():
                            try: ik_queue.get_nowait()
                            except queue.Empty: break
                        ik_queue.put(('reset', new_pos, new_rot))
                        print_pose(new_pos, new_rot, "Returned to initial pose")
                        ctrl_enabled.set()

            # --- Print IK warnings (to terminal) ---
            while not warn_queue.empty():
                try:
                    print(warn_queue.get_nowait())
                except queue.Empty:
                    break

            # --- Continuous key polling -> push IK task ---
            # get_pressed() reads hardware state directly, no OS key-repeat delay
            if ctrl_enabled.is_set():
                keys = pygame.key.get_pressed()
                # Combine all per-key position increments in this frame (supports diagonal motion)
                combined_pos   = np.zeros(3)
                first_rot_euler = None
                any_pressed    = False

                for k, (delta_pos, rot_euler) in PYGAME_KEY_MAP.items():
                    if keys[k]:
                        combined_pos += delta_pos
                        if rot_euler is not None and first_rot_euler is None:
                            first_rot_euler = rot_euler
                        any_pressed = True

                if any_pressed:
                    try:
                        ik_queue.put_nowait(('move', combined_pos, first_rot_euler))
                    except queue.Full:
                        pass  # IK is still computing the previous frame, skip this one (no backlog)

                # --- Gripper control (Z=close, X=open; stops updating at the limit) ---
                if keys[pygame.K_z] or keys[pygame.K_x]:
                    with lock:
                        if keys[pygame.K_z]:
                            gripper_des = max(gripper_min, gripper_des - GRIPPER_STEP)
                        if keys[pygame.K_x]:
                            gripper_des = min(gripper_max, gripper_des + GRIPPER_STEP)

            # --- Refresh pygame window ---
            screen.fill((20, 20, 20))
            screen.blit(font.render(
                "W/S/A/D/Q/E  I/K/J/L/U/O  Space  R  Esc",
                True, (160, 210, 160)), (10, 8))
            screen.blit(font.render(
                "Gripper:  Z=close  X=open",
                True, (160, 200, 240)), (10, 34))
            with lock:
                g_disp = gripper_des
            g_act  = robot.get_current_pos_gripper()
            g_pct  = (g_disp - gripper_min) / max(gripper_max - gripper_min, 1e-6) * 100
            screen.blit(font.render(
                f"des:{g_disp:.3f}  act:{g_act:.3f}  {g_pct:.0f}%"
                f"  [{gripper_min:.2f}~{gripper_max:.2f}]",
                True, (160, 200, 240)), (10, 58))
            pygame.display.flip()

            clock.tick(CTRL_FREQ_KEY)

    finally:
        stop_event.set()
        ctrl_thread.join(timeout=1.0)
        ik_thread.join(timeout=1.0)
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
