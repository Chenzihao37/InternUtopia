from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
# Old Aliengo import block for rollback:
# from internutopia_extension.configs.robots.aliengo import (
#     AliengoRobotCfg,
#     move_to_point_cfg,
#     camera_cfg,
# )
from internutopia_extension.configs.robots.pipergo2 import (
    PiperGo2RobotCfg,
    move_to_point_cfg,
)
from internutopia_extension.configs.objects import DynamicCubeCfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

import numpy as np
import cv2

# ================= 路径 =================
SCENE_USD = "/home/zyserver/work/my_project/InternUtopia/internutopia/demo/merom_scene_baked.usd"

SPAWN_PRIM = "/World/env_0/scene/MeromScene/carpet_ctclvd_1/Asset/base_link/visuals"
SPAWN_Y_OFFSET = -0.3
# Old manual starts for rollback:
# MANUAL_START_XY = np.array([0.42541398, 9.0150], dtype=float)
# MANUAL_START_XY = np.array([0.06496, 7.50704], dtype=float)
MANUAL_START_XY = np.array([0.58, 7.50704], dtype=float)
TEST_PATH_WAYPOINT = np.array([2.74051, 7.24591, 0.0], dtype=float)
# Old test waypoint for rollback:
# TEST_PATH_WAYPOINT = np.array([2.74051, 7.24591, 0.0], dtype=float)
TEST_PATH_WAYPOINT = np.array([2.74051, 7.67599, 0.0], dtype=float)

DEBUG_GRASP_OBJECT_CFG = DynamicCubeCfg(
    name="debug_grasp_cube",
    prim_path="/World/debug_grasp_cube",
    # Old debug grasp position for rollback:
    # position=(3.32087, 7.24591, 0.29768),
    # position=(3.32087, 7.67599, 0.29768),
    position=(3.46261, 7.67599, 1.30000),
    scale=(0.05, 0.05, 0.05),
    color=(0.85, 0.2, 0.2),
)

ROBOT_Z = 0.55
# Old disabled state for rollback:
# ENABLE_RUNTIME_SCENE_COLLISION_PATCH = False
ENABLE_RUNTIME_SCENE_COLLISION_PATCH = True
USE_DYNAMIC_CHASE_CAMERA = False

# ================= env =================
headless = not has_display()

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
                # Old Aliengo robot block for rollback:
                # AliengoRobotCfg(
                #     position=(0, 0, ROBOT_Z),
                #     controllers=[move_to_point_cfg],
                #     # Debug version:
                #     # The original script mounted the robot camera sensor here.
                #     # We comment it out first to verify whether the reset crash is
                #     # caused purely by RepCamera / syntheticdata initialization.
                #     #
                #     # sensors=[camera_cfg],
                # )
                PiperGo2RobotCfg(
                    # Old reset-at-origin version for rollback:
                    # position=(0, 0, ROBOT_Z),
                    position=(float(MANUAL_START_XY[0]), float(MANUAL_START_XY[1]), ROBOT_Z),
                    controllers=[move_to_point_cfg],
                    # Old default mass behavior for rollback:
                    # arm_mass_scale left at config default
                    arm_mass_scale=0.25,
                )
            ],
            objects=[
                # Debug version:
                # Add a small graspable cube near the new test path so later we
                # can reuse the same scene for manipulation experiments.
                DEBUG_GRASP_OBJECT_CFG,
            ],
        )
    ],
)

# ================= 初始化 =================
import_extensions()

env = Env(config)
# Old single-reset version for rollback:
# obs, _ = env.reset()
obs, _ = env.reset()

# ================= stage =================
from omni.isaac.core.utils.stage import get_current_stage

stage = get_current_stage()

# ================= 灯光 =================
from pxr import Gf, UsdGeom, UsdLux

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(5000)

sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2000)
UsdGeom.Xformable(stage.GetPrimAtPath("/World/Sun")).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 35))

print("Lighting added")


def _disable_instances_and_add_collision(stage):
    from pxr import PhysxSchema, Usd, UsdPhysics

    # Old debug version for rollback:
    # test3_debug.py originally relied on the baked Merom scene to already have
    # usable collision on floors and other static meshes. That assumption is not
    # reliable here, so Aliengo can visually stand on the floor while physically
    # falling through it.

    scene_root = stage.GetPrimAtPath("/World/env_0/scene")
    if not scene_root.IsValid():
        print("Scene root not found, collision patch skipped")
        return

    # Old broad traversal for rollback:
    # for prim in stage.Traverse():
    #     if prim.IsInstance():
    #         prim.SetInstanceable(False)

    for prim in Usd.PrimRange(scene_root):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    collision_count = 0
    # Old broad traversal for rollback:
    # for prim in stage.Traverse():
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


# Old active call for rollback:
# _disable_instances_and_add_collision(stage)
#
# PiperGo2 version:
# Do not mutate scene instances/colliders after env.reset() has already created
# the robot and physics tensor views. That invalidates the simulation view and
# breaks subsequent pose queries. Keep the helper here for reference, but gate
# it off by default.
if ENABLE_RUNTIME_SCENE_COLLISION_PATCH:
    _disable_instances_and_add_collision(stage)
    # Old retry-reset attempt for rollback:
    # Rebuilding the env here consumes the only episode in this debug script and
    # leaves later stage/prim queries pointing at an invalid task.
    #
    # obs, _ = env.reset()
else:
    print("Runtime scene collision patch skipped for PiperGo2")

# ================= 俯视camera =================
# Disabled in debug version for now.
# Keep the whole block commented for easy rollback.
#
# import omni.kit.viewport.utility as vp_utils
#
# top_camera_path = "/World/top_view_camera"
# camera_prim = UsdGeom.Camera.Define(stage, top_camera_path)
#
# xform = UsdGeom.Xformable(camera_prim)
# xform.AddTranslateOp().Set(Gf.Vec3f(2, 2, 18))
# xform.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
#
# env.step({})
#
# viewport = vp_utils.get_active_viewport()
# viewport.set_active_camera(top_camera_path)
#
# print("Top view camera set")

# ================= PiperGo2 跟随camera =================
try:
    import omni.kit.viewport.utility as vp_utils
except ImportError:
    vp_utils = None

# ================= 默认视角对准机器人 =================
try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
    except ImportError:
        set_camera_view = None


# ================= 工具函数 =================
def get_world_pos(stage, prim_path):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Spawn prim not found or invalid: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def ensure_spawn_support_platform(stage, center_xy, top_z=0.02):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    # Old version for rollback:
    # We relied on the baked Merom floor collision to support the robot.
    # In practice PiperGo2 can still fall straight through that floor, so for
    # the debug navigation script we add a hidden static support platform under
    # the spawn area instead of mutating the whole scene.

    platform_path = "/World/debug_spawn_support"
    cube = UsdGeom.Cube.Define(stage, platform_path)
    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3f(float(center_xy[0]), float(center_xy[1]), float(top_z - 0.05)))
    xform.AddScaleOp().Set(Gf.Vec3f(1.6, 1.6, 0.05))

    prim = stage.GetPrimAtPath(platform_path)
    prim.CreateAttribute("visibility", prim.GetAttribute("visibility").GetTypeName() if prim.HasAttribute("visibility") else None)
    UsdGeom.Imageable(prim).MakeInvisible()

    UsdPhysics.CollisionAPI.Apply(prim)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateApproximationAttr().Set("boundingCube")

    print(f"Spawn support platform ready at ({center_xy[0]:.3f}, {center_xy[1]:.3f}, top={top_z:.3f})")


def teleport_robot(stage, pos):
    from omni.isaac.core.articulations import ArticulationView

    # Old Aliengo prim path for rollback:
    # robot = ArticulationView(prim_paths_expr="/World/env_0/robots/aliengo")

    robot = ArticulationView(prim_paths_expr="/World/env_0/robots/pipergo2")
    robot.initialize()

    # Old version for rollback:
    # robot.set_world_poses(
    #     np.array([[pos[0], pos[1], 1.2]]),
    #     np.array([[0, 0, 0, 1]]),
    # )

    # Old debug version for rollback:
    # We previously teleported exactly onto the carpet reference prim and lifted
    # the robot to z=1.2. In the baked Merom scene that can place Aliengo above
    # or beside non-walkable geometry, so the robot appears to "fall out"
    # immediately after startup.
    #
    # robot.set_world_poses(
    #     np.array([[pos[0], pos[1], 1.2]]),
    #     np.array([[1, 0, 0, 0]]),
    # )

    robot.set_world_poses(
        np.array([[pos[0], pos[1], ROBOT_Z]]),
        np.array([[1, 0, 0, 0]]),
    )
    print(f"PiperGo2 teleported to {(pos[0], pos[1], ROBOT_Z)}")
    try:
        pos_w, quat_w = robot.get_world_poses()
        print(f"Robot world pose => pos={pos_w[0]}, quat={quat_w[0]}")
    except Exception as exc:
        print(f"Robot pose query failed: {exc}")


def focus_default_view_on_robot(robot_xy, robot_z=ROBOT_Z):
    if set_camera_view is None:
        print("Default viewport focus skipped: set_camera_view unavailable")
        return

    # Old debug version for rollback:
    # We relied on the removed top-view camera to find the robot in the scene.
    # With that camera disabled, the default perspective camera often stays far
    # away from the teleported robot and makes it look like the robot was not
    # loaded at all.

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


def attach_follow_camera_to_pipergo2(stage):
    # Old no-camera version for rollback:
    # PiperGo2 debug runs previously relied only on the default perspective
    # camera. Here we add a simple viewport-only follow camera under the robot,
    # without touching RepCamera / syntheticdata.
    if vp_utils is None:
        print("Follow camera skipped: viewport utility unavailable")
        return None

    camera_path = "/World/env_0/robots/pipergo2/debug_follow_camera"
    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3f(-2.2, 0.0, 1.35))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(78.0, 0.0, -90.0))

    viewport = vp_utils.get_active_viewport()
    viewport.set_active_camera(camera_path)
    print(f"Follow camera set => {camera_path}")
    return camera_path


def update_chase_camera(robot_obs):
    # Old active chase-camera behavior for rollback:
    # This function used to overwrite /OmniverseKit_Persp every few frames.
    # Keep the implementation intact, but allow the script to fall back to the
    # fixed default view when debugging camera issues.
    if set_camera_view is None or not USE_DYNAMIC_CHASE_CAMERA:
        return

    position = np.array(robot_obs["position"], dtype=float)
    orientation = np.array(robot_obs["orientation"], dtype=float)
    rotation = quat_wxyz_to_rotmat(orientation)

    # Old third-person rear-follow version for rollback:
    # back = rotation @ np.array([-2.8, 0.0, 1.6], dtype=float)
    # target_offset = rotation @ np.array([1.2, 0.0, 0.45], dtype=float)
    # eye = (position + back).tolist()
    # target = (position + target_offset).tolist()

    # Old front-head-but-looking-back version for rollback:
    # front_head = rotation @ np.array([0.95, 0.0, 1.05], dtype=float)
    # look_back = rotation @ np.array([0.15, 0.0, 0.55], dtype=float)

    # Previous front-facing shot for rollback:
    # front_head = rotation @ np.array([0.95, 0.0, 1.05], dtype=float)
    # look_forward = rotation @ np.array([1.85, 0.0, 0.65], dtype=float)

    # Updated closer/lower shot:
    # `position` here is the PiperGo2 base pose (`trunk/base`), so to make the
    # camera feel level with the robot body we should keep the z-offset much
    # closer to that base height. Also pull x forward distance back a bit so the
    # camera sits nearer to the "head" region instead of floating too far out.
    front_head = rotation @ np.array([0.58, 0.0, 0.34], dtype=float)
    look_forward = rotation @ np.array([1.45, 0.0, 0.34], dtype=float)

    eye = (position + front_head).tolist()
    target = (position + look_forward).tolist()
    set_camera_view(
        eye=eye,
        target=target,
        camera_prim_path="/OmniverseKit_Persp",
    )


def get_robot_obs(obs_data):
    # Old direct-access version for rollback:
    # The Aliengo demo assumed obs itself always exposed "position" at top level.
    # PiperGo2 demos in this repo already handle a few nested observation layouts,
    # so we reuse that safer pattern here.
    if isinstance(obs_data, dict) and "position" in obs_data:
        return obs_data
    if isinstance(obs_data, dict) and "pipergo2" in obs_data:
        return obs_data["pipergo2"]
    if isinstance(obs_data, dict) and "pipergo2_0" in obs_data:
        return obs_data["pipergo2_0"]
    if isinstance(obs_data, (list, tuple)) and len(obs_data) > 0:
        first = obs_data[0]
        if isinstance(first, dict) and "pipergo2" in first:
            return first["pipergo2"]
        if isinstance(first, dict) and "pipergo2_0" in first:
            return first["pipergo2_0"]
        return first
    raise KeyError(f"Unsupported observation structure: {type(obs_data)}")


def stabilize_robot(env, target_xy, settle_steps=120):
    # PiperGo2 is much more top-heavy than Aliengo. Give the locomotion
    # controller a short settling window after teleport so it can recover to a
    # stable stance before we ask it to navigate.
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


# ================= Waypoints =================
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
    # New debug path for rollback-friendly testing:
    "4": [TEST_PATH_WAYPOINT],
}

# ================= 初始化位置 =================
# Old spawn-from-scene + teleport route for rollback:
# spawn = get_world_pos(stage, SPAWN_PRIM)
#
# # Old version for rollback:
# # spawn[2] = 0.0
#
# # Follow the safer spawn adjustment used in test.py: step slightly away from the
# # carpet reference point before teleporting the robot. This reduces the chance
# # of spawning on top of visual-only or edge geometry.
# spawn[1] += SPAWN_Y_OFFSET
# spawn[2] = 0.0
#
# env.step({})
# teleport_robot(stage, spawn)
# focus_default_view_on_robot(spawn[:2], robot_z=ROBOT_Z)
# stabilize_robot(env, spawn[:2], settle_steps=120)

# PiperGo2 version:
# Spawn directly near the known scene start area instead of teleporting an
# already-reset articulation. PiperGo2 is much less tolerant of post-reset
# teleports than Aliengo and tends to topple before the locomotion controller
# can recover.
spawn = np.array([float(MANUAL_START_XY[0]), float(MANUAL_START_XY[1]), 0.0], dtype=float)
focus_default_view_on_robot(spawn[:2], robot_z=ROBOT_Z)
# Old static follow-camera call for rollback:
# attach_follow_camera_to_pipergo2(stage)
stabilize_robot(env, spawn[:2], settle_steps=120)

# Old Aliengo prim check for rollback:
# robot_prim = stage.GetPrimAtPath("/World/env_0/robots/aliengo")
robot_prim = stage.GetPrimAtPath("/World/env_0/robots/pipergo2")
print(f"Robot prim valid => {robot_prim.IsValid()}")
print(
    "Debug grasp object => "
    f"name={DEBUG_GRASP_OBJECT_CFG.name}, "
    f"position={DEBUG_GRASP_OBJECT_CFG.position}, "
    f"scale={DEBUG_GRASP_OBJECT_CFG.scale}"
)

current_pos = spawn.copy()

# ================= 状态 =================
current_path = []
current_target_index = 0
moving = False

print("\n控制说明：")
print("1 = 沙发")
print("2 = 卧室")
print("3 = 浴室")
print("4 = 测试路径")
print("q = 退出\n")

# ================= 主循环 =================
import select
import sys

try:
    while env.simulation_app.is_running():
        # Old Aliengo-style loop for rollback:
        # idle_action = {
        #     move_to_point_cfg.name: [tuple(current_pos.tolist())]
        # }
        #
        # obs, _, _, _, _ = env.step(action=idle_action)
        # robot_obs = get_robot_obs(obs)
        # update_chase_camera(robot_obs)
        #
        # PiperGo2 version:
        # Do not send an "idle back to current_pos" command right before the real
        # navigation goal every frame. That can effectively cancel out movement
        # and make path "4" look completely unresponsive.
        if not moving:
            idle_action = {
                move_to_point_cfg.name: [tuple(current_pos.tolist())]
            }
            obs, _, _, _, _ = env.step(action=idle_action)
            robot_obs = get_robot_obs(obs)
            update_chase_camera(robot_obs)

        # ================= 获取camera图像 =================
        # Debug version:
        # Keep the original OpenCV preview path here for rollback, but comment it out
        # while we validate whether the native crash disappears without GUI image
        # presentation.
        #
        # Old Aliengo camera preview for rollback:
        # if "sensors" in obs:
        #     cam_data = obs["sensors"][camera_cfg.name]
        #
        #     if "rgba" in cam_data:
        #         img = cam_data["rgba"]
        #
        #         if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
        #             img = img[:, :, :3]
        #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        #             cv2.imshow("Robot Camera", img)
        #             cv2.waitKey(1)

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip()

            if cmd == "q":
                break

            if cmd in PATHS:
                current_path = PATHS[cmd]
                current_target_index = 0
                moving = True
                print(f"开始路径 {cmd}")

        if moving and current_target_index < len(current_path):
            goal = current_path[current_target_index]

            env_action = {
                move_to_point_cfg.name: [tuple(goal.tolist())],
            }

            obs, _, _, _, _ = env.step(action=env_action)

            robot_obs = get_robot_obs(obs)
            update_chase_camera(robot_obs)
            pos = np.array(robot_obs["position"])
            dist = np.linalg.norm(pos[:2] - goal[:2])

            controller_obs = robot_obs["controllers"][move_to_point_cfg.name]
            if current_target_index == 0:
                print(
                    f"moving => goal={goal.tolist()} "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                    f"dist={dist:.3f} finished={controller_obs.get('finished')}"
                )

            if dist < 0.4:
                print(f"到达 waypoint {current_target_index}")
                current_target_index += 1
                current_pos = goal

        elif moving:
            print("路径完成")
            moving = False
finally:
    # Debug version:
    # Explicit cleanup is added to avoid native crashes during Omniverse/Python
    # shutdown ordering.
    try:
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"cv2 cleanup failed: {exc}")

    try:
        env.close()
    except Exception as exc:
        print(f"env cleanup failed: {exc}")
