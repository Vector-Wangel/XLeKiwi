#!/usr/bin/env python3
"""
Trajectory recording and playback utility (no dirty-line version)
"""
import time, json, os
import numpy as np
from typing import List, Optional


class Recorder:
    def __init__(self, path: str = None):
        if path is None:
            path = time.strftime("trajectory_%Y%m%d_%H%M%S.jsonl")
        self.path = path
        self.fd = open(path, "w", encoding="utf-8")
        self.t0 = None

    # ---------------- Recording ----------------
    def log(self, pos: List[float], vel: List[float] = None, gripper_pos: float = None, gripper_vel: float = None):
        t = time.time()
        if self.t0 is None:
            self.t0 = t
        # Build a complete, valid JSON string in one shot to avoid partial writes
        data = {"t": t - self.t0, "pos": list(pos)}
        if vel is not None:
            data["vel"] = list(vel)
        if gripper_pos is not None:
            data["gripper_pos"] = gripper_pos
        if gripper_vel is not None:
            data["gripper_vel"] = gripper_vel
        line = json.dumps(data, ensure_ascii=False)
        self.fd.write(line + "\n")
        self.fd.flush()          # Flush immediately so Ctrl-C doesn't leave a broken line

    def close(self):
        if self.fd and not self.fd.closed:
            self.fd.close()
            print(f"[Recorder] Trajectory saved -> {self.path}")

    # ---------------- Static playback ----------------
    @staticmethod
    def play(
        robot,
        filepath: str,
        kp: List[float],
        kd: List[float],
        fc: Optional[List[float]] = None,
        fv: Optional[List[float]] = None,
        vel_threshold: float = 0.0,
        tau_limit: Optional[List[float]] = None,
        gripper_kp: float = 5.0,
        gripper_kd: float = 0.5,
    ):
        # Used when the playback mode selects the motor position+velocity mode
        max_torque = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
        with open(filepath, "r", encoding="utf-8") as f:
            frames = [json.loads(line) for line in f if line.strip()]
        if not frames:
            print("[Player] Empty file, nothing to play back")
            return

        print(f"[Player] {len(frames)} frames total")

        # Move to the starting waypoint
        first_frame = frames[0]
        print("[Player] Moving to trajectory start point...")

        # Move joints to the start (using Joint_Pos_Vel mode)
        start_pos = first_frame["pos"]
        move_vel = [0.5] * len(start_pos)  # Slow velocity 0.5 rad/s

        # Move the gripper to the start (if gripper data is present)
        if "gripper_pos" in first_frame:
            gripper_start_pos = first_frame["gripper_pos"]
            print(f"[Player] Gripper moving to start: {gripper_start_pos:.3f} rad")
            robot.gripper_control(gripper_start_pos, 0.5, 0.5)
            time.sleep(2.0)  # Wait for the gripper to arrive

        # Slowly move to the start point and wait for arrival
        robot.Joint_Pos_Vel(start_pos, move_vel, max_torque, iswait=True, tolerance=0.05, timeout=30.0)

        print("[Player] Reached start point, beginning playback...")
        t0 = time.time()

        for f in frames:
            while time.time() - t0 < f["t"]:
                time.sleep(0.001)

            # Joint control
            if fc is not None and fv is not None:
                robot_gra = robot.get_Gravity()
                robot_vel = robot.get_current_vel()
                robot_torque = np.array(robot_gra) + robot.get_friction_compensation(robot_vel, fc, fv, vel_threshold)
                if tau_limit is not None:
                    robot_torque = np.clip(robot_torque, -np.array(tau_limit), np.array(tau_limit))
            else:
                robot_torque = [0.0] * 6

            # Choose a mode as needed:

            # MIT mode provides some impedance behavior but lower position accuracy
            # robot.pos_vel_tqe_kp_kd(f["pos"], f["vel"], robot_torque, kp, kd)

            # Position+velocity mode gives higher position accuracy
            robot.Joint_Pos_Vel(f["pos"], f["vel"], max_torque)

            # Gripper control (if gripper data is present)
            if "gripper_pos" in f:
                robot.gripper_control_MIT(f["gripper_pos"], f["gripper_vel"], 0.0, gripper_kp, gripper_kd)

        print("[Player] Playback complete")
