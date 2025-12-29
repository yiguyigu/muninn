#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from dataclasses import dataclass

from pymavlink import mavutil


@dataclass
class Config:
    device: str = "/dev/ttyUSB0"
    baud: int = 921600

    # setpoint 发送频率（Hz），PX4 要求 > 2Hz，建议 20Hz
    sp_hz: float = 20.0

    # 动作参数
    cruise_vel: float = 0.5  # 巡航速度 m/s
    cruise_time: float = 2.0 # 2.0s * 0.5m/s = 1.0m

    # OFFBOARD 切换前预热时间
    warmup_time: float = 2.0


def now_ms() -> int:
    # MAVLink set_position_target_local_ned.time_boot_ms 是 uint32
    return int((time.time() * 1000) % 4294967296)


def send_vel_ned(master: mavutil.mavfile, vx: float, vy: float, vz: float):
    """发送 NED 速度 setpoint（PX4 OFFBOARD 常用）。"""
    master.mav.set_position_target_local_ned_send(
        now_ms(),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  # 只控制速度
        0,
        0,
        0,
        vx,
        vy,
        vz,
        0,
        0,
        0,
        0,
        0,
    )

def send_attitude_target(master, roll=0.0, pitch=0.0, yaw=0.0, thrust=0.5):
    """
    发送姿态指令实现盲飞。
    thrust: 0.0 到 1.0 (0.5 通常是持平推力)
    """
    # 将欧拉角转为四元数
    import math
    def euler_to_quaternion(r, p, y):
        sr, cr = math.sin(r/2), math.cos(r/2)
        sp, cp = math.sin(p/2), math.cos(p/2)
        sy, cy = math.sin(y/2), math.cos(y/2)
        return [
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ]

    q = euler_to_quaternion(roll, pitch, yaw)

    master.mav.set_attitude_target_send(
        now_ms(),
        master.target_system,
        master.target_component,
        0b00000000, # type_mask: 0表示所有字段都有效
        q,          # 四元数姿态
        0, 0, 0,    # 旋转速率 (忽略)
        thrust      # 推力 [0, 1]
    )


def arm_state(master: mavutil.mavfile) -> bool:
    try:
        return bool(master.motors_armed())
    except Exception:
        return False

def pump(master: mavutil.mavfile, timeout_s: float = 0.0):
    """单线程读取并处理消息：更新 flightmode、打印 STATUSTEXT。"""
    end = time.time() + timeout_s
    while True:
        remaining = end - time.time()
        if timeout_s <= 0:
            remaining = 0
        if remaining < 0:
            break

        msg = master.recv_match(blocking=timeout_s > 0, timeout=remaining if timeout_s > 0 else 0)
        if msg is None:
            break

        mtype = msg.get_type()
        if mtype == "STATUSTEXT":
            text = getattr(msg, "text", "")
            severity = getattr(msg, "severity", None)
            print(f"[STATUSTEXT][{severity}] {text}")
        elif mtype == "COMMAND_ACK":
            cmd = getattr(msg, "command", None)
            result = getattr(msg, "result", None)
            print(f"[ACK] command={cmd} result={result}")
        elif mtype in {"LOCAL_POSITION_NED", "ODOMETRY", "ESTIMATOR_STATUS", "EKF_STATUS_REPORT"}:
            pass


def set_mode_px4(master: mavutil.mavfile, mode_name: str, timeout_s: float = 5.0) -> bool:
    """请求 PX4 切模式，并通过 HEARTBEAT 观察是否生效（单线程读取）。"""
    try:
        master.set_mode(mode_name)
    except Exception as e:
        print(f"[MODE] set_mode({mode_name}) 调用失败: {e}")
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # OFFBOARD 切换前/过程中需要持续发送 setpoint
        send_vel_ned(master, 0, 0, 0)
        pump(master, timeout_s=0.5)
        if master.flightmode == mode_name:
            print(f"[MODE] 已进入 {master.flightmode}")
            return True

    print(f"[MODE] 切换 {mode_name} 超时，当前 {master.flightmode}")
    return False


def main():
    cfg = Config()

    print(f"[LINK] 连接 {cfg.device} @ {cfg.baud}...")
    master = mavutil.mavlink_connection(cfg.device, baud=cfg.baud)

    print("[LINK] 等待 HEARTBEAT...")
    master.wait_heartbeat(timeout=10)
    print(f"[LINK] 已连接 sys={master.target_system} comp={master.target_component} mode={master.flightmode}")

    try:
        print("[SAFE] 默认不自动解锁。请用遥控器 ARM；随时切回手动/断电可急停。")
        print("[SAFE] 等待 ARM...")
        deadline = time.time() + 30
        while time.time() < deadline:
            pump(master, timeout_s=0.1)
            if arm_state(master):
                break
            time.sleep(0.1)
        if not arm_state(master):
            raise RuntimeError("等待 ARM 超时（30s）。")

        print("[SAFE] 已 ARM。开始发送 setpoint 预热...")
        sp_dt = 1.0 / cfg.sp_hz
        warm_end = time.time() + cfg.warmup_time
        while time.time() < warm_end:
            pump(master, timeout_s=0.0)
            if not arm_state(master):
                raise RuntimeError("检测到 DISARM，终止。")
            send_vel_ned(master, 0, 0, 0)
            time.sleep(sp_dt)

        print("[MODE] 尝试切入 OFFBOARD...")
        if not set_mode_px4(master, "OFFBOARD", timeout_s=8.0):
            raise RuntimeError("OFFBOARD 切换失败。通常原因：无本地位置（EKF）、未持续 setpoint、或参数限制。")

        def run_vel_phase(name: str, vx: float, vy: float, vz: float, seconds: float):
            print(f"[RUN] {name}: vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} duration={seconds:.1f}s")
            end = time.time() + seconds
            while time.time() < end:
                pump(master, timeout_s=0.0)
                if not arm_state(master):
                    raise RuntimeError("检测到 DISARM，终止。")
                send_vel_ned(master, vx, vy, vz)
                time.sleep(sp_dt)

        # 巡航逻辑：上升1米 -> 前进1米 -> 下降
        # 假设速度为 0.5m/s，则 2秒 为 1米
        run_vel_phase("上升1米", vx=0, vy=0, vz=-0.5, seconds=2.0)
        run_vel_phase("悬停", vx=0, vy=0, vz=0, seconds=2.0)
        run_vel_phase("前进1米", vx=0.5, vy=0, vz=0, seconds=2.0)
        run_vel_phase("悬停", vx=0, vy=0, vz=0, seconds=1.0)
        run_vel_phase("下降", vx=0, vy=0, vz=0.5, seconds=2.0)

        print("[MODE] 切 LAND...")
        set_mode_px4(master, "LAND", timeout_s=5.0)

        print("[DONE] 等待自动 DISARM...")
        land_deadline = time.time() + 30
        while time.time() < land_deadline:
            pump(master, timeout_s=0.2)
            if not arm_state(master):
                print("[DONE] 已 DISARM，结束")
                return
            time.sleep(0.3)
        print("[DONE] 超时未 DISARM（可能需要你手动解除）。")

    finally:
        try:
            master.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
