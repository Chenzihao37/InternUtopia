from typing import List, Optional

from internutopia.core.config.robot import ControllerCfg


class HolonomicPlanarMoveToPointControllerCfg(ControllerCfg):
    """平面全向底盘（如 XLeRobot 的 root_x/y + 绕 z 转动）走向目标点的速度控制。"""

    type: Optional[str] = "HolonomicPlanarMoveToPointController"
    forward_speed: Optional[float] = 0.55
    rotation_speed: Optional[float] = 2.2
    # 平面距离小于该值视为「到达」（finished、与 waypoint 判定应对齐 demo）
    threshold: Optional[float] = 0.22
    # 仅当距离小于该值时才发零速度；须明显小于 threshold，否则会在 0.22~0.4m
    # 等区间过早停轮、demo 仍判未到达，路径永远走不完。
    stop_commanding_dist: Optional[float] = 0.08
    # 线速度朝目标的比例增益（乘在裁剪后的方向上）
    linear_gain: Optional[float] = 1.0
    # 距离仍大于 stop_commanding_dist 时线速度下限；>0 易在接触不稳时强行推底盘导致弹跳、乱飞，默认 0。
    min_linear_speed: Optional[float] = 0.0
    # (0,1] 对底盘 (vx,vy,ω) 做指数平滑，越小越平顺；0 表示关闭
    velocity_lpf_alpha: Optional[float] = 0.0
    base_joint_names: Optional[List[str]] = None
