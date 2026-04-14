from collections import OrderedDict

import numpy as np

from internutopia.core.robot.articulation import IArticulation
from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.articulation_subset import ArticulationSubset
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.xlerobot import (
    XLEROBOT_EXPECTED_DOF_NAMES,
    XlerobotRobotCfg,
)

# 与 xlerobot_isaaclab `JointVelocityActionCfg` 一致：平面全向速度控制，不可施加大位置刚度
_BASE_VELOCITY_JOINTS = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")


@BaseRobot.register("XlerobotRobot")
class XlerobotRobot(BaseRobot):
    """XLeRobot dual-arm mobile manipulator loaded from the project USD (Isaac Lab asset)."""

    EEF_CENTER_OFFSET = np.array([0.0, 0.0, 0.06], dtype=float)

    def __init__(self, config: XlerobotRobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f"xlerobot {config.name}: position    : {self._start_position}")
        log.debug(f"xlerobot {config.name}: orientation : {self._start_orientation}")
        log.debug(f"xlerobot {config.name}: usd_path         : {config.usd_path}")
        log.debug(f"xlerobot {config.name}: config.prim_path : {config.prim_path}")

        self._robot_scale = np.array(config.scale) if config.scale is not None else np.array([1.0, 1.0, 1.0])
        self.articulation = IArticulation.create(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=config.usd_path,
            scale=self._robot_scale,
        )
        self._manip_hold_q = None
        self._manip_joint_indices = None
        self._base_joint_indices = None
        self._holonomic_controller_name = None

    def post_reset(self):
        super().post_reset()
        self._robot_base = self._resolve_robot_base()
        self._arm_base = self._resolve_arm_base()
        self._end_effector = self._resolve_end_effector()
        self.set_gains()
        self._cache_manipulator_hold_and_base_indices()

    def _resolve_robot_base(self) -> IRigidBody:
        preferred_names = []
        if getattr(self.config, "base_link_name", None):
            preferred_names.append(self.config.base_link_name)
        preferred_names.extend(["base_link", "base", "trunk"])

        for link_name in preferred_names:
            rigid_body = self._rigid_body_map.get(self.config.prim_path + "/" + link_name)
            if rigid_body is not None:
                log.debug(f"xlerobot {self.config.name}: using base link {link_name}")
                return rigid_body

        available_links = sorted(path.split("/")[-1] for path in self._rigid_body_map)
        raise KeyError(
            f"Cannot find base link for {self.config.name}. "
            f"Tried {preferred_names}. Available link names (last segment): {available_links}"
        )

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_base(self) -> IRigidBody:
        return self._robot_base

    def get_arm_base(self) -> IRigidBody:
        return self._arm_base

    def get_end_effector(self) -> IRigidBody:
        return self._end_effector

    def get_pose(self):
        return self._robot_base.get_pose()

    def _resolve_end_effector(self) -> IRigidBody:
        for suffix in ("Fixed_Jaw", "Fixed_Jaw_2"):
            path = self.config.prim_path + "/" + suffix
            rigid_body = self._rigid_body_map.get(path)
            if rigid_body is not None:
                return rigid_body
        log.warning("xlerobot: no gripper rigid body found, falling back to base for eef pose")
        return self._robot_base

    def _resolve_arm_base(self) -> IRigidBody:
        for suffix in ("base_link", "torso_link", "base"):
            path = self.config.prim_path + "/" + suffix
            rigid_body = self._rigid_body_map.get(path)
            if rigid_body is not None:
                return rigid_body
        return self._robot_base

    def _cache_manipulator_hold_and_base_indices(self) -> None:
        """在 reset 后缓存上身/夹爪/头的关节角作为每步位置保持目标；并记录全向底盘控制器名。"""
        self._holonomic_controller_name = None
        ctrls = getattr(self.config, "controllers", None) or []
        for c in ctrls:
            if getattr(c, "type", None) == "HolonomicPlanarMoveToPointController":
                self._holonomic_controller_name = c.name
                break

        self._manip_joint_indices = None
        self._manip_hold_q = None
        self._base_joint_indices = None
        if not self.articulation.handles_initialized:
            return
        dof_names = list(self.articulation.dof_names or [])
        if not dof_names:
            return

        manip_names = [n for n in XLEROBOT_EXPECTED_DOF_NAMES if n in dof_names and n not in _BASE_VELOCITY_JOINTS]
        if manip_names:
            subset_m = ArticulationSubset(self.articulation, manip_names)
            ji = subset_m.joint_indices
            if ji is not None:
                self._manip_joint_indices = np.asarray(ji, dtype=np.int64)
                q_sub = subset_m.get_joint_positions()
                if q_sub is not None:
                    self._manip_hold_q = np.asarray(q_sub, dtype=np.float64).reshape(-1)

        base_names = [n for n in _BASE_VELOCITY_JOINTS if n in dof_names]
        if base_names:
            subset_b = ArticulationSubset(self.articulation, base_names)
            if subset_b.joint_indices is not None:
                self._base_joint_indices = np.asarray(subset_b.joint_indices, dtype=np.int64)

    def _apply_manipulator_pose_hold(self) -> None:
        """用 PD 目标角保持上身，避免每步 ``set_joint_positions`` 硬拧与 PhysX/底盘速度叠加导致关节速度爆炸（日志里 arm_|qd|_max 数百）。"""
        if self._manip_hold_q is None or self._manip_joint_indices is None:
            return
        if self._manip_hold_q.size != self._manip_joint_indices.size:
            return
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(self._manip_hold_q, dtype=np.float64).reshape(-1),
                joint_indices=np.asarray(self._manip_joint_indices, dtype=np.int64).reshape(-1),
            )
        )

    def _zero_base_velocities(self) -> None:
        if self._base_joint_indices is None or self._base_joint_indices.size == 0:
            return
        zeros = np.zeros(self._base_joint_indices.size, dtype=np.float64)
        self.articulation.set_joint_velocities(zeros, joint_indices=self._base_joint_indices)

    def apply_action(self, action: dict):
        if not self.articulation.handles_initialized:
            return
        # 先于底盘速度写入上身目标，对齐官方每步 JointPosition + JointVelocity 组合
        self._apply_manipulator_pose_hold()
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warning(f"unknown controller {controller_name} in action")
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self._apply_articulation_control(control)
        if self._holonomic_controller_name and self._holonomic_controller_name not in action:
            self._zero_base_velocities()

    def _apply_articulation_control(self, control) -> None:
        """下发控制到 PhysX。

        Isaac Sim 的 ``SingleArticulation.apply_action`` 对带 ``joint_indices`` 的
        ``ArticulationAction`` 往往不按子集速度生效，底盘会「完全不动」；而若与
        ``set_joint_velocities`` 同一帧各写一遍又会冲突导致炸飞。

        对「仅关节速度 + 显式 joint_indices」走单独的 ``set_joint_velocities``；
        其余情况仍走 ``apply_action``。
        """
        ji = getattr(control, "joint_indices", None)
        jv = getattr(control, "joint_velocities", None)
        jp = getattr(control, "joint_positions", None)
        je = getattr(control, "joint_efforts", None)

        if ji is not None and jv is not None and jp is None and je is None:
            jv_arr = np.asarray(jv, dtype=np.float64).reshape(-1)
            ji_arr = np.asarray(ji, dtype=np.int64).reshape(-1)
            if jv_arr.size == ji_arr.size and jv_arr.size > 0:
                self.articulation.set_joint_velocities(jv_arr, joint_indices=ji_arr)
                return

        self.articulation.apply_action(control)

    def get_obs(self) -> OrderedDict:
        position, orientation = self._robot_base.get_pose()
        arm_base_position, arm_base_orientation = self._arm_base.get_pose()
        eef_base_position, eef_orientation = self._end_effector.get_pose()
        eef_position = self._compute_eef_center(
            base_position=eef_base_position,
            base_orientation=eef_orientation,
        )
        obs = {
            "position": position,
            "orientation": orientation,
            "joint_positions": self.articulation.get_joint_positions(),
            "joint_velocities": self.articulation.get_joint_velocities(),
            "arm_base_position": arm_base_position,
            "arm_base_orientation": arm_base_orientation,
            "eef_position": eef_position,
            "eef_orientation": eef_orientation,
            "controllers": {},
            "sensors": {},
        }

        for controller_name, controller in self.controllers.items():
            obs["controllers"][controller_name] = controller.get_obs()
        for sensor_name, sensor in self.sensors.items():
            obs["sensors"][sensor_name] = sensor.get_data()
        return self._make_ordered(obs)

    @classmethod
    def _compute_eef_center(
        cls,
        base_position: np.ndarray,
        base_orientation: np.ndarray,
    ) -> np.ndarray:
        rotation = cls._quat_to_rotmat(base_orientation)
        return np.array(base_position, dtype=float) + rotation @ cls.EEF_CENTER_OFFSET

    @staticmethod
    def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat_wxyz]
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-8:
            return np.eye(3, dtype=float)
        w /= norm
        x /= norm
        y /= norm
        z /= norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def set_gains(self):
        dof_names = list(self.articulation.dof_names)
        if not dof_names:
            log.warning("xlerobot: articulation.dof_names empty, skip set_gains")
            return

        base_names = [n for n in _BASE_VELOCITY_JOINTS if n in dof_names]
        manip_names = [
            n for n in XLEROBOT_EXPECTED_DOF_NAMES if n in dof_names and n not in _BASE_VELOCITY_JOINTS
        ]

        if manip_names:
            subset_m = ArticulationSubset(self.articulation, manip_names)
            # 与全向底盘同帧时不宜过刚；配合 apply_action 位置目标而非硬 set 关节角
            kps_m = np.array([28.0] * len(manip_names))
            kds_m = np.array([6.0] * len(manip_names))
            self.articulation.set_gains(
                kps=kps_m,
                kds=kds_m,
                joint_indices=subset_m.joint_indices,
            )

        if base_names:
            subset_b = ArticulationSubset(self.articulation, base_names)
            # kp=0：允许 Holonomic / 键盘式 JointVelocity 驱动；kd 提高有利于速度跟踪与接地稳定
            kps_b = np.zeros(len(base_names), dtype=float)
            kds_b = np.array([56.0, 56.0, 22.0][: len(base_names)], dtype=float)
            if len(kds_b) < len(base_names):
                kds_b = np.pad(kds_b, (0, len(base_names) - len(kds_b)), constant_values=22.0)
            self.articulation.set_gains(
                kps=kps_b,
                kds=kds_b,
                joint_indices=subset_b.joint_indices,
            )

        self.articulation.set_solver_position_iteration_count(10)
        # 速度迭代为 0 时接触/摩擦解算易飘、穿模；略增有利于「贴地」
        self.articulation.set_solver_velocity_iteration_count(4)
        # 全向底盘行走 demo：关闭自碰，减轻双臂在保持姿态时的额外内力（与常见导航 demo 一致）
        self.articulation.set_enabled_self_collisions(False)
