from pathlib import Path
from typing import List, Optional

from pydantic import Field

from internutopia.core.config.robot import ControllerCfg, RobotCfg
from internutopia_extension.configs.controllers.holonomic_planar_move_to_point_controller import (
    HolonomicPlanarMoveToPointControllerCfg,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
XLEROBOT_USD_PATH = str((_REPO_ROOT / "xlerobot_isaaclab-" / "assets" / "robots" / "xlerobot" / "xlerobot.usd").resolve())

# DoF names from xlerobot Isaac Lab stack (see xlerobot_isaaclab-/.../utils/constant.py).
XLEROBOT_EXPECTED_DOF_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
    "Rotation_2",
    "Pitch_2",
    "Elbow_2",
    "Wrist_Pitch_2",
    "Wrist_Roll_2",
    "Jaw_2",
    "head_pan_joint",
    "head_tilt_joint",
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
]

# 与 test3_debug 中 `move_to_point_cfg.name` 一致，便于同一套 action 字典
move_to_point_cfg = HolonomicPlanarMoveToPointControllerCfg(
    name="move_to_point",
    # 与官方键盘一档 (0.1 m/s) 同量级；上身每步位置保持后略可提高线速度
    forward_speed=0.14,
    rotation_speed=0.42,
    threshold=0.42,
    stop_commanding_dist=0.07,
    linear_gain=0.32,
    min_linear_speed=0.0,
    velocity_lpf_alpha=0.25,
)


class XlerobotRobotCfg(RobotCfg):
    name: Optional[str] = "xlerobot"
    type: Optional[str] = "XlerobotRobot"
    prim_path: Optional[str] = "/xlerobot"
    usd_path: Optional[str] = XLEROBOT_USD_PATH
    base_link_name: Optional[str] = "base_link"
    controllers: Optional[List[ControllerCfg]] = Field(
        default_factory=lambda: [move_to_point_cfg],
    )
