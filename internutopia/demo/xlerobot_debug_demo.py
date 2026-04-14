"""
在 InternUtopia / Isaac Sim 中加载 XLeRobot，并在 Merom 场景下按 waypoint 行走。

默认与 `test3_debug.py` 相同：Merom 烘焙场景、灯光、碰撞补丁、路径点与键盘选路。
底盘使用 `HolonomicPlanarMoveToPointController`（root_x/y + 绕 z），与 PiperGo2 的四足速度环不同，
但 action 仍为 `move_to_point: [(gx, gy, gz)]`，主循环与 test3_debug 一致。

将 `USE_EMPTY_SCENE` 设为 True 可改回空场景（仅短步数验证，无键盘路径）。

在未执行 `pip install -e .` 的情况下，需先把仓库根目录加入 `sys.path`。
"""
import select
import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO = Path(__file__).resolve().parents[2]
_repo_str = str(_REPO)
if _repo_str not in sys.path:
    sys.path.insert(0, _repo_str)

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
from internutopia_extension.configs.objects import DynamicCubeCfg
from internutopia_extension.configs.robots.xlerobot import (
    XLEROBOT_USD_PATH,
    XlerobotRobotCfg,
    move_to_point_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

# ================= 场景（与 test3_debug 对齐） =================
EMPTY_SCENE_USD = str(_REPO / "internutopia" / "assets" / "scenes" / "empty.usd")
MEROM_SCENE_USD = str(_REPO / "internutopia" / "demo" / "merom_scene_baked.usd")

USE_EMPTY_SCENE = False

SCENE_USD = EMPTY_SCENE_USD if USE_EMPTY_SCENE else MEROM_SCENE_USD

MANUAL_START_XY = np.array([0.58, 7.50704], dtype=float)
# Merom 世界系地面高度需与烘焙场景一致；过高易悬空、接触差易倒。若 Merom 地台不在 z≈0，请改回 ~0.45–0.52。
MEROM_ROBOT_Z = 0.02

# 空场景地面约 z=0；与 MEROM_ROBOT_Z 对齐便于对照碰撞箱/贴地。
ROBOT_SPAWN_Z = 0.02 if USE_EMPTY_SCENE else float(MEROM_ROBOT_Z)
ROBOT_SPAWN_XY = (
    (0.0, 0.0) if USE_EMPTY_SCENE else (float(MANUAL_START_XY[0]), float(MANUAL_START_XY[1]))
)

DEBUG_GRASP_OBJECT_CFG = DynamicCubeCfg(
    name="debug_grasp_cube",
    prim_path="/World/debug_grasp_cube",
    position=(3.46261, 7.67599, 1.30000),
    scale=(0.05, 0.05, 0.05),
    color=(0.85, 0.2, 0.2),
)

ENABLE_RUNTIME_SCENE_COLLISION_PATCH = not USE_EMPTY_SCENE
# 与 test3_debug 一致：Merom 烘焙地面接触不可靠时在出生点下加隐形静态承托，减轻穿模/飘起
ENABLE_SPAWN_SUPPORT_PLATFORM = not USE_EMPTY_SCENE
USE_DYNAMIC_CHASE_CAMERA = False

# 与 move_to_point_cfg.threshold 对齐的平面到达半径（略收紧便于停稳）
WAYPOINT_REACH_XY = float(move_to_point_cfg.threshold)

# ---------- 走向目标时「飞起」排查：底盘关节 / PhysX 根速度 / 竖直突变 ----------
DEBUG_MOVE_TO_POINT = True
DEBUG_MOVE_PRINT_EVERY = 20
DEBUG_MOVE_Z_STEP_WARN = 0.012
DEBUG_MOVE_Z_ABS_WARN = 0.18

_DEBUG_BASE_JOINT_NAMES = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
)

# 空场景：短跑；Merom：交互循环
MAX_PHYSICS_STEPS_EMPTY = 400

# ================= Waypoints（与 test3_debug 相同） =================
TEST_PATH_WAYPOINT = np.array([2.74051, 7.67599, 0.0], dtype=float)

WAYPOINTS = {
    2: np.array([3.45022, 6.5689, 0.0]),
    4: np.array([1.73975, 5.9176, 0.0]),
    5: np.array([1.73975, 1.09495, 0.0]),
    6: np.array([3.17464, 1.26846, 0.0]),
    7: np.array([1.78321, 3.62602, 0.0]),
    8: np.array([3.09079, 3.62602, 0.0]),
}

PATHS = {
    "1": [WAYPOINTS[2]],
    "2": [WAYPOINTS[4], WAYPOINTS[5], WAYPOINTS[6]],
    "3": [WAYPOINTS[5], WAYPOINTS[7], WAYPOINTS[8]],
    "4": [TEST_PATH_WAYPOINT],
}


def _disable_instances_and_add_collision(stage):
    """手动补丁（参考用）。已注册 Env 时优先用 SingleInferenceTaskCfg.enable_static_scene_mesh_collision_patch。"""
    from pxr import PhysxSchema, Usd, UsdPhysics

    scene_root = stage.GetPrimAtPath("/World/env_0/scene")
    if not scene_root.IsValid():
        print("Scene root not found, collision patch skipped")
        return

    for prim in Usd.PrimRange(scene_root):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    collision_count = 0
    for prim in Usd.PrimRange(scene_root):
        if prim.GetTypeName() != "Mesh":
            continue
        try:
            UsdPhysics.CollisionAPI.Apply(prim)
            physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx.CreateApproximationAttr().Set("convexHull")
            collision_count += 1
        except Exception:
            pass

    print(f"Collision added to {collision_count} meshes")


def ensure_spawn_support_platform(stage, center_xy, top_z: float) -> None:
    """在出生点下方加隐形碰撞立方体，改善 Merom 上机器人『不接地』、打滑乱飞（同 test3_debug 思路）。"""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    platform_path = "/World/debug_spawn_support_xlerobot"
    if stage.GetPrimAtPath(platform_path).IsValid():
        return

    cube = UsdGeom.Cube.Define(stage, platform_path)
    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3f(float(center_xy[0]), float(center_xy[1]), float(top_z - 0.05)))
    xform.AddScaleOp().Set(Gf.Vec3f(1.8, 1.8, 0.06))

    prim = stage.GetPrimAtPath(platform_path)
    UsdGeom.Imageable(prim).MakeInvisible()

    UsdPhysics.CollisionAPI.Apply(prim)
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateRigidBodyEnabledAttr(False)

    # 不显式设 PhysxCollisionAPI.approximation：部分 Kit 无 CreateApproximationAttr，立方体仅 CollisionAPI 即可参与检测
    try:
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        if hasattr(physx_collision, "CreateContactOffsetAttr"):
            physx_collision.CreateContactOffsetAttr().Set(0.02)
        if hasattr(physx_collision, "CreateRestOffsetAttr"):
            physx_collision.CreateRestOffsetAttr().Set(0.0)
    except Exception:
        pass

    print(f"XLeRobot spawn support platform => center_xy=({center_xy[0]:.3f}, {center_xy[1]:.3f}), top_z={top_z:.3f}")


def _setup_stage_after_reset(env, use_merom: bool):
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom, UsdLux

    stage = get_current_stage()
    if use_merom:
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(5000)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2000)
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Sun")).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 35))
        print("Lighting added (Merom, same as test3_debug)")
        if ENABLE_RUNTIME_SCENE_COLLISION_PATCH:
            print(
                "Merom mesh collision handled at task load (enable_static_scene_mesh_collision_patch); "
                "skip post-reset scene USD edits."
            )
        else:
            print("Runtime scene collision patch disabled (empty scene or intentional)")
    else:
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight_XlerobotDemo")
        dome.CreateIntensityAttr(3500)
        print("Lighting added (empty scene)")


try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
    except ImportError:
        set_camera_view = None


def get_robot_obs(obs_data):
    if isinstance(obs_data, dict) and "position" in obs_data:
        return obs_data
    if isinstance(obs_data, dict) and "xlerobot" in obs_data:
        return obs_data["xlerobot"]
    if isinstance(obs_data, dict) and "xlerobot_0" in obs_data:
        return obs_data["xlerobot_0"]
    if isinstance(obs_data, (list, tuple)) and len(obs_data) > 0:
        first = obs_data[0]
        if isinstance(first, dict) and "xlerobot" in first:
            return first["xlerobot"]
        if isinstance(first, dict) and "xlerobot_0" in first:
            return first["xlerobot_0"]
        return first
    raise KeyError(f"Unsupported observation structure: {type(obs_data)}")


def _yaw_wxyz(quat_wxyz) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _try_physx_dof_and_root_debug(robot_view) -> str:
    """从 ArticulationView 底层 physics_view 读关节速度目标、实际速度、根空间速度（版本不兼容时吞异常）。"""
    pv = getattr(robot_view, "_physics_view", None)
    if pv is None:
        return "no _physics_view"
    parts: list[str] = []
    try:
        names = list(robot_view.dof_names or [])
        tgt = np.asarray(pv.get_dof_velocity_targets(), dtype=float).reshape(-1)
        vel = np.asarray(pv.get_dof_velocities(), dtype=float).reshape(-1)
        if tgt.size == len(names) and vel.size == len(names):
            for jn in _DEBUG_BASE_JOINT_NAMES:
                if jn not in names:
                    continue
                i = names.index(jn)
                parts.append(f"{jn}:tgt={tgt[i]:.4f} act={vel[i]:.4f}")
    except Exception as exc:
        parts.append(f"dof_vel_err={exc!r}")
    try:
        rv = np.asarray(pv.get_root_velocities(), dtype=float).reshape(-1)
        if rv.size >= 6:
            lin, ang = rv[:3], rv[3:6]
            parts.append("root_v_lin=" + ",".join(f"{x:.4f}" for x in lin))
            parts.append("root_v_ang=" + ",".join(f"{x:.4f}" for x in ang))
    except Exception as exc:
        parts.append(f"root_vel_err={exc!r}")
    return " ".join(parts)


def debug_xlerobot_move_step(
    robot_view,
    robot_obs: dict,
    *,
    goal: np.ndarray,
    controller_obs: dict,
    prev_pos: Optional[np.ndarray],
    step_idx: int,
    physics_dt: float,
    z_spawn_ref: float,
) -> Tuple[np.ndarray, int]:
    """每步调用一次；返回 (当前位置作下一帧 prev, 递增后的 step_idx)。满足周期或异常竖直运动时打印。"""
    pos = np.array(robot_obs["position"], dtype=float)
    step_idx = int(step_idx) + 1
    dz = float("nan")
    vz_est = float("nan")
    if prev_pos is not None:
        dz = float(pos[2] - prev_pos[2])
        vz_est = dz / max(float(physics_dt), 1e-9)

    dist_xy = float(np.linalg.norm(pos[:2] - goal[:2]))
    finished = controller_obs.get("finished")
    yaw = _yaw_wxyz(robot_obs["orientation"])

    trigger = (
        (step_idx % DEBUG_MOVE_PRINT_EVERY == 0)
        or step_idx <= 3
        or (prev_pos is not None and np.isfinite(dz) and abs(dz) > DEBUG_MOVE_Z_STEP_WARN)
    )
    if np.isfinite(pos[2]) and pos[2] > z_spawn_ref + DEBUG_MOVE_Z_ABS_WARN:
        trigger = True

    if trigger:
        q = np.asarray(robot_obs.get("joint_positions"), dtype=float).reshape(-1)
        qd = np.asarray(robot_obs.get("joint_velocities"), dtype=float).reshape(-1)
        dof_names = list(robot_view.dof_names or [])
        base_line = ""
        if len(dof_names) == q.size == qd.size:
            chunks = []
            for jn in _DEBUG_BASE_JOINT_NAMES:
                if jn not in dof_names:
                    continue
                i = dof_names.index(jn)
                chunks.append(f"{jn}:q={q[i]:.4f} qd={qd[i]:.4f}")
            base_line = " | ".join(chunks)
        arm_qd_max = float("nan")
        if len(dof_names) == qd.size and qd.size > 0:
            base_idx = {dof_names.index(n) for n in _DEBUG_BASE_JOINT_NAMES if n in dof_names}
            others = [float(abs(qd[i])) for i in range(len(qd)) if i not in base_idx]
            if others:
                arm_qd_max = max(others)
        phys = _try_physx_dof_and_root_debug(robot_view)
        arm_qd_s = f"{arm_qd_max:.4f}" if np.isfinite(arm_qd_max) else str(arm_qd_max)
        print(
            f"[move_dbg] step={step_idx} goal_xy=({goal[0]:.3f},{goal[1]:.3f}) "
            f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
            f"dz={dz:.6f} vz_est={vz_est:.3f} yaw={yaw:.3f} "
            f"dist_xy={dist_xy:.3f} finished={finished}"
        )
        if base_line:
            print(f"            base: {base_line}")
        print(f"            arm_|qd|_max={arm_qd_s}  phys: {phys}")

    return pos, step_idx


def focus_default_view_on_robot(robot_xy, robot_z):
    if set_camera_view is None:
        print("Default viewport focus skipped: set_camera_view unavailable")
        return

    eye = [robot_xy[0] - 2.8, robot_xy[1] - 2.2, robot_z + 1.8]
    target = [robot_xy[0], robot_xy[1], max(robot_z - 0.4, 0.2)]

    set_camera_view(
        eye=eye,
        target=target,
        camera_prim_path="/OmniverseKit_Persp",
    )
    print(f"Default viewport focused => eye={eye}, target={target}")


def quat_wxyz_to_rotmat(quat_wxyz):
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


def update_chase_camera(robot_obs):
    if set_camera_view is None or not USE_DYNAMIC_CHASE_CAMERA:
        return

    position = np.array(robot_obs["position"], dtype=float)
    orientation = np.array(robot_obs["orientation"], dtype=float)
    rotation = quat_wxyz_to_rotmat(orientation)
    front_head = rotation @ np.array([0.58, 0.0, 0.34], dtype=float)
    look_forward = rotation @ np.array([1.45, 0.0, 0.34], dtype=float)
    eye = (position + front_head).tolist()
    target = (position + look_forward).tolist()
    set_camera_view(
        eye=eye,
        target=target,
        camera_prim_path="/OmniverseKit_Persp",
    )


def stabilize_robot(env, target_xy, settle_steps=120):
    obs = None
    idle_action = {
        move_to_point_cfg.name: [(float(target_xy[0]), float(target_xy[1]), 0.0)],
    }
    for step in range(settle_steps):
        obs, _, terminated, _, _ = env.step(action=idle_action)
        if step % 30 == 0:
            robot_obs = get_robot_obs(obs)
            update_chase_camera(robot_obs)
            pos = robot_obs["position"]
            print(
                f"stabilize step={step} "
                f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
        episode_terminated = terminated[0] if isinstance(terminated, (list, tuple)) else bool(terminated)
        if episode_terminated:
            print("stabilize terminated early")
            break
    return obs


headless = not has_display()

_task_objects = [] if USE_EMPTY_SCENE else [DEBUG_GRASP_OBJECT_CFG]

config = Config(
    simulator=SimConfig(
        physics_dt=1 / 240,
        rendering_dt=1 / 240,
        use_fabric=False,
        headless=headless,
        webrtc=headless,
    ),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=SCENE_USD,
            robots=[
                XlerobotRobotCfg(
                    position=(float(ROBOT_SPAWN_XY[0]), float(ROBOT_SPAWN_XY[1]), float(ROBOT_SPAWN_Z)),
                )
            ],
            objects=_task_objects,
            # 在 init_robots 之前补 Merom 地面碰撞；切勿仅在 env.reset() 后再改场景（会破坏 PhysX view、地面仍无接触）
            enable_static_scene_mesh_collision_patch=ENABLE_RUNTIME_SCENE_COLLISION_PATCH,
        )
    ],
)

import_extensions()

_usd = Path(XLEROBOT_USD_PATH)
if not _usd.is_file():
    raise FileNotFoundError(
        f"未找到 XLeRobot USD：{_usd} "
        "(期望：xlerobot_isaaclab-/assets/robots/xlerobot/xlerobot.usd)"
    )
if not Path(SCENE_USD).is_file():
    raise FileNotFoundError(f"场景 USD 不存在：{SCENE_USD}")

env = Env(config)
PHYSICS_DT = float(config.simulator.physics_dt)
obs, _ = env.reset()
_setup_stage_after_reset(env, use_merom=not USE_EMPTY_SCENE)

if ENABLE_SPAWN_SUPPORT_PLATFORM:
    from omni.isaac.core.utils.stage import get_current_stage

    _st = get_current_stage()
    # 低出生高度时「base − 0.14」会把承托板沉到地面以下；改为略低于 base 的薄垫片顶面参数
    _mrz = float(MEROM_ROBOT_Z)
    _support_top_z = _mrz - 0.14 if _mrz > 0.15 else max(0.0, _mrz - 0.03)
    ensure_spawn_support_platform(
        _st,
        (float(MANUAL_START_XY[0]), float(MANUAL_START_XY[1])),
        top_z=_support_top_z,
    )

print("obs keys:", list(obs.keys()) if obs is not None else None)
if obs is not None:
    print("base position:", obs.get("position"))

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import get_current_stage

robot_view = ArticulationView(prim_paths_expr="/World/env_0/robots/xlerobot", name="xlerobot_view")
robot_view.initialize()
print("xlerobot dof_names:", robot_view.dof_names)

try:
    if USE_EMPTY_SCENE:
        step = 0
        while env.simulation_app.is_running() and step < MAX_PHYSICS_STEPS_EMPTY:
            obs, _, terminated, _, _ = env.step(action={})
            step += 1
            if step % 80 == 0 and obs is not None:
                p = np.array(obs["position"], dtype=float)
                print(f"step={step} position=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            if terminated:
                break
        print(f"empty-scene demo finished after {step} steps")
    else:
        spawn = np.array([float(MANUAL_START_XY[0]), float(MANUAL_START_XY[1]), 0.0], dtype=float)
        focus_default_view_on_robot(spawn[:2], robot_z=MEROM_ROBOT_Z)
        stabilize_robot(env, spawn[:2], settle_steps=120)

        stage = get_current_stage()
        robot_prim = stage.GetPrimAtPath("/World/env_0/robots/xlerobot")
        print(f"Robot prim valid => {robot_prim.IsValid()}")
        print(
            "Debug grasp object => "
            f"name={DEBUG_GRASP_OBJECT_CFG.name}, "
            f"position={DEBUG_GRASP_OBJECT_CFG.position}, "
            f"scale={DEBUG_GRASP_OBJECT_CFG.scale}"
        )

        current_pos = spawn.copy()
        current_path = []
        current_target_index = 0
        moving = False
        _dbg_move_prev_pos: Optional[np.ndarray] = None
        _dbg_move_step = 0

        print("\n控制说明（与 test3_debug 相同）：")
        print("1 = 沙发")
        print("2 = 卧室")
        print("3 = 浴室")
        print("4 = 测试路径")
        print("q = 退出\n")

        while env.simulation_app.is_running():
            if not moving:
                idle_action = {
                    move_to_point_cfg.name: [tuple(current_pos.tolist())],
                }
                obs, _, _, _, _ = env.step(action=idle_action)
                robot_obs = get_robot_obs(obs)
                update_chase_camera(robot_obs)

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip()

                if cmd == "q":
                    break

                if cmd in PATHS:
                    current_path = PATHS[cmd]
                    current_target_index = 0
                    moving = True
                    _dbg_move_prev_pos = None
                    _dbg_move_step = 0
                    print(f"开始路径 {cmd}")
                    if DEBUG_MOVE_TO_POINT:
                        print(
                            f"  [move_dbg] PHYSICS_DT={PHYSICS_DT:g} 每{DEBUG_MOVE_PRINT_EVERY}步或 "
                            f"单步|dz|>{DEBUG_MOVE_Z_STEP_WARN} 或 z>spawn+{DEBUG_MOVE_Z_ABS_WARN} 时打印；"
                            "phys 行为 PhysX 关节速度目标/根空间速度。"
                        )

            if moving and current_target_index < len(current_path):
                goal = current_path[current_target_index]

                env_action = {
                    move_to_point_cfg.name: [tuple(goal.tolist())],
                }

                obs, _, _, _, _ = env.step(action=env_action)

                robot_obs = get_robot_obs(obs)
                update_chase_camera(robot_obs)
                pos = np.array(robot_obs["position"])
                if not np.isfinite(pos).all():
                    print("检测到机器人位姿 NaN/Inf，停止路径以防 PhysX 持续报错。")
                    moving = False
                    continue
                dist = np.linalg.norm(pos[:2] - goal[:2])

                controller_obs = robot_obs["controllers"][move_to_point_cfg.name]
                if DEBUG_MOVE_TO_POINT:
                    _dbg_move_prev_pos, _dbg_move_step = debug_xlerobot_move_step(
                        robot_view,
                        robot_obs,
                        goal=goal,
                        controller_obs=controller_obs,
                        prev_pos=_dbg_move_prev_pos,
                        step_idx=_dbg_move_step,
                        physics_dt=PHYSICS_DT,
                        z_spawn_ref=float(ROBOT_SPAWN_Z),
                    )
                elif current_target_index == 0:
                    print(
                        f"moving => goal={goal.tolist()} "
                        f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                        f"dist={dist:.3f} finished={controller_obs.get('finished')}"
                    )

                if dist < WAYPOINT_REACH_XY:
                    print(f"到达 waypoint {current_target_index}")
                    current_target_index += 1
                    current_pos = goal
                    _dbg_move_prev_pos = None
                    _dbg_move_step = 0

            elif moving:
                print("路径完成")
                moving = False
                _dbg_move_prev_pos = None
                _dbg_move_step = 0
finally:
    try:
        env.close()
    except Exception as exc:
        print(f"env cleanup failed: {exc}")
