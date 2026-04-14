from pathlib import Path

import numpy as np

from internutopia.bridge import (
    FrankaManipulationAPI,
    MultiRobotNavigateAPI,
    PiperGo2ManipulationAPI,
    create_franka_robot_cfg,
    create_pipergo2_robot_cfg,
)
from internutopia.core.config import Config, SimConfig
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions
from internutopia_extension.configs.objects import DynamicCubeCfg, VisualCubeCfg
from internutopia_extension.configs.robots.pipergo2 import move_to_point_cfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg


SCENE_ASSET_PATH = "/home/zyserver/work/my_project/InternUtopia/internutopia/demo/merom_scene_baked.usd"
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"

ROBOT_START = (0.58, 7.50704, 0.55)
FRANKA_START = (3.46261, 8.00000, 0.41172)
STAGING_WAYPOINT = (2.74051, 7.67599, 0.0)

PLACE_PEDESTAL_POS = (1.02, 7.08000, 0.16)
PLACE_PEDESTAL_SCALE = (0.14, 0.14, 0.32)
CUBE_POS = (3.46261, 7.67599, 0.41172)
# Previous relative-offset Franka target attempt. This matched the geometry in
# franka_manipulation.py more literally, but in this Merom scene it made Franka
# miss the object entirely.
# FRANKA_PICK_OFFSET = (0.40, 0.00, 0.10)
# FRANKA_PRE_OFFSET = (0.40, 0.00, 0.30)
# FRANKA_CUBE_POS = (
#     FRANKA_START[0] + FRANKA_PICK_OFFSET[0],
#     FRANKA_START[1] + FRANKA_PICK_OFFSET[1],
#     FRANKA_START[2] + FRANKA_PICK_OFFSET[2],
# )
FRANKA_CUBE_POS = (3.86261, 8.00000, 0.41172)

NEARBY_CLUTTER = [
    {
        "name": "nearby_block_tall",
        "position": (3.58500, 7.73600, CUBE_POS[2]),
        "scale": (0.04, 0.04, 0.09),
        "color": (0.2, 0.55, 0.9),
    },
    {
        "name": "nearby_block_flat",
        "position": (3.40200, 7.74400, CUBE_POS[2]),
        "scale": (0.08, 0.05, 0.03),
        "color": (0.95, 0.78, 0.18),
    },
    {
        "name": "nearby_block_small",
        "position": (3.56000, 7.59400, CUBE_POS[2]),
        "scale": (0.035, 0.035, 0.05),
        "color": (0.28, 0.82, 0.46),
    },
]

FORCE_GUI = True
# Old slower demo pacing for rollback:
# SCENE_PREVIEW_STEPS = 240
SCENE_PREVIEW_STEPS = 90
# STABILIZE_STEPS = 120
STABILIZE_STEPS = 60
# Old faster-but-too-tight navigation limit:
# STAGING_MAX_STEPS = 900
STAGING_MAX_STEPS = 1500
# Old lighter Franka cube mass for rollback:
# FRANKA_CUBE_MASS = 0.01
FRANKA_CUBE_MASS = 0.02
# Old x-adjusted place target for rollback:
# FRANKA_PLACE_POS = (3.74261, 7.70000, 0.41172)
# Keep x aligned with the successful grasp geometry and only pull the place site
# closer along y.
# Old faster place target tuning for rollback:
# FRANKA_PLACE_POS = (3.86261, 7.72000, 0.41172)
FRANKA_PLACE_POS = (3.86261, 7.84000, 0.41172)
FRANKA_RELEASE_PAUSE_STEPS = 90
FRANKA_PLACE_LIFT_Z = FRANKA_PLACE_POS[2] + 0.28
FRANKA_PLACE_XY = (FRANKA_PLACE_POS[0], FRANKA_PLACE_POS[1])
FRANKA_PLACE_SURFACE_POS = (FRANKA_PLACE_XY[0], FRANKA_PLACE_XY[1], FRANKA_PLACE_POS[2])
FRANKA_PLACE_LIFT_POS = (FRANKA_PLACE_XY[0], FRANKA_PLACE_XY[1], FRANKA_PLACE_LIFT_Z)
# Old slower-than-default but still too fast release lift:
# FRANKA_RELEASE_WAYPOINT_COUNT = 24
FRANKA_RELEASE_WAYPOINT_COUNT = 20

try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
    except ImportError:
        set_camera_view = None


def _disable_instances_and_add_collision(stage):
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


def focus_default_view_on_robot(robot_xy, robot_z=ROBOT_START[2]):
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


def get_robot_obs(env_obs, robot_name):
    if isinstance(env_obs, list):
        env_obs = env_obs[0]
    if not isinstance(env_obs, dict):
        raise KeyError(f"Unsupported observation structure: {type(env_obs)}")
    if robot_name not in env_obs:
        raise KeyError(f"Robot {robot_name} not found in obs keys={list(env_obs.keys())}")
    return env_obs[robot_name]


def stabilize_robot(env, robot_name, target_xy, settle_steps=STABILIZE_STEPS):
    idle_action = [
        {
            robot_name: {
                move_to_point_cfg.name: [(float(target_xy[0]), float(target_xy[1]), 0.0)],
            },
            "franka": {},
        }
    ]
    obs = None
    for step in range(settle_steps):
        obs, _, terminated, _, _ = env.step(action=idle_action)
        if step % 30 == 0:
            robot_obs = get_robot_obs(obs, robot_name)
            pos = robot_obs["position"]
            print(
                f"stabilize step={step} "
                f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
        if terminated[0]:
            print("stabilize terminated early")
            break
    return obs


def navigate_to_waypoint(env, robot_name, waypoint_xy, max_steps=STAGING_MAX_STEPS, threshold=0.10, label="staging"):
    goal_action = [
        {
            robot_name: {
                move_to_point_cfg.name: [(float(waypoint_xy[0]), float(waypoint_xy[1]), 0.0)],
            },
            "franka": {},
        }
    ]
    dist = float("inf")
    for step in range(max_steps):
        obs, _, terminated, _, _ = env.step(action=goal_action)
        robot_obs = get_robot_obs(obs, robot_name)
        pos = np.array(robot_obs["position"], dtype=float)
        dist = float(np.linalg.norm(pos[:2] - np.array(waypoint_xy[:2], dtype=float)))
        if step % 60 == 0:
            print(
                f"{label} step={step} "
                f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"dist={dist:.4f}"
            )
        if dist <= threshold:
            print(f"{label} reached => step={step}, dist={dist:.4f}")
            return True, obs
        if terminated[0]:
            print(f"{label} terminated early")
            return False, obs
    print(f"{label} failed => dist={dist:.4f}")
    return False, obs


def set_mass_if_present(env, object_name: str, mass: float) -> None:
    try:
        obj = env.runner.get_obj(object_name)
    except Exception:
        return
    obj.set_mass(mass)
    print(f"set_mass => object={object_name}, mass={mass}")


class SingleRobotEnvAdapter:
    def __init__(self, multi_env: Env, robot_name: str, peer_robot_names: list[str]) -> None:
        self._multi_env = multi_env
        self.robot_name = robot_name
        self.peer_robot_names = list(peer_robot_names)

    def step(self, action):
        env_action = {}
        env_action[self.robot_name] = action
        for peer_name in self.peer_robot_names:
            env_action[peer_name] = {}
        env_obs, reward, terminated, truncated, info = self._multi_env.step([env_action])
        obs = env_obs[0][self.robot_name]
        reward_value = reward[0] if reward else 0.0
        terminated_value = terminated[0] if terminated else False
        truncated_value = truncated[0] if truncated else False
        info_value = info[0] if info else None
        return obs, reward_value, terminated_value, truncated_value, info_value

    def get_observations(self):
        observations = self._multi_env.get_observations()
        if not observations:
            return {}
        return observations[0][self.robot_name]

    @property
    def runner(self):
        return self._multi_env.runner

    def close(self):
        return None


def build_scene_objects():
    cube_cfg = DynamicCubeCfg(
        name="pick_cube",
        prim_path="/World/pick_cube",
        position=CUBE_POS,
        scale=(0.05, 0.05, 0.05),
        color=(0.85, 0.2, 0.2),
    )
    franka_cube_cfg = DynamicCubeCfg(
        name="franka_pick_cube",
        prim_path="/World/franka_pick_cube",
        position=FRANKA_CUBE_POS,
        scale=(0.05, 0.05, 0.05),
        color=(0.18, 0.45, 0.95),
    )
    place_pedestal_cfg = DynamicCubeCfg(
        name="place_pedestal",
        prim_path="/World/place_pedestal",
        position=PLACE_PEDESTAL_POS,
        scale=PLACE_PEDESTAL_SCALE,
        color=(0.42, 0.42, 0.46),
    )
    clutter_cfgs = [
        VisualCubeCfg(
            name=item["name"],
            prim_path=f"/World/{item['name']}",
            position=item["position"],
            scale=item["scale"],
            color=list(item["color"]),
        )
        for item in NEARBY_CLUTTER
    ]
    return cube_cfg, franka_cube_cfg, place_pedestal_cfg, clutter_cfgs


def build_robot_cfgs():
    piper_cfg = create_pipergo2_robot_cfg(position=ROBOT_START, arm_mass_scale=0.25)
    piper_cfg.name = "pipergo2"
    piper_cfg.prim_path = "/pipergo2"

    franka_cfg = create_franka_robot_cfg(position=FRANKA_START)
    franka_cfg.name = "franka"
    franka_cfg.prim_path = "/franka"
    return piper_cfg, franka_cfg


def build_env_config(robot_cfgs, objects):
    return Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            rendering_interval=0,
            headless=not FORCE_GUI,
            native=not FORCE_GUI,
            webrtc=not FORCE_GUI,
        ),
        env_num=1,
        metrics_save_path="none",
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=SCENE_ASSET_PATH,
                robots=list(robot_cfgs),
                objects=list(objects),
            )
        ],
    )


def start_multi_robot_env(config):
    import_extensions()
    return Env(config)


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cube_cfg, franka_cube_cfg, place_pedestal_cfg, clutter_cfgs = build_scene_objects()
    piper_cfg, franka_cfg = build_robot_cfgs()

    # Old inline scene/object/robot construction for rollback:
    # cube_cfg = DynamicCubeCfg(
    #     name="pick_cube",
    #     prim_path="/World/pick_cube",
    #     position=CUBE_POS,
    #     scale=(0.05, 0.05, 0.05),
    #     color=(0.85, 0.2, 0.2),
    # )
    # franka_cube_cfg = DynamicCubeCfg(
    #     name="franka_pick_cube",
    #     prim_path="/World/franka_pick_cube",
    #     position=FRANKA_CUBE_POS,
    #     scale=(0.05, 0.05, 0.05),
    #     color=(0.18, 0.45, 0.95),
    # )
    # place_pedestal_cfg = DynamicCubeCfg(
    #     name="place_pedestal",
    #     prim_path="/World/place_pedestal",
    #     position=PLACE_PEDESTAL_POS,
    #     scale=PLACE_PEDESTAL_SCALE,
    #     color=(0.42, 0.42, 0.46),
    # )
    # clutter_cfgs = [
    #     VisualCubeCfg(
    #         name=item["name"],
    #         prim_path=f"/World/{item['name']}",
    #         position=item["position"],
    #         scale=item["scale"],
    #         color=list(item["color"]),
    #     )
    #     for item in NEARBY_CLUTTER
    # ]
    # piper_cfg = create_pipergo2_robot_cfg(position=ROBOT_START, arm_mass_scale=0.25)
    # piper_cfg.name = "pipergo2"
    # piper_cfg.prim_path = "/pipergo2"
    # franka_cfg = create_franka_robot_cfg(position=FRANKA_START)
    # franka_cfg.name = "franka"
    # franka_cfg.prim_path = "/franka"

    # Keep this object-aware vec_env startup because MultiRobotNavigateAPI does
    # not currently accept scene objects. We still import it above on purpose so
    # this file stays aligned with the bridge-oriented demo style.
    bridge_api_placeholder = MultiRobotNavigateAPI(
        scene_asset_path=SCENE_ASSET_PATH,
        robot_cfgs=[piper_cfg, franka_cfg],
        headless=not FORCE_GUI,
    )
    # Old direct env startup for rollback:
    # config = Config(
    #     simulator=SimConfig(
    #         physics_dt=1 / 240,
    #         rendering_dt=1 / 240,
    #         use_fabric=False,
    #         rendering_interval=0,
    #         headless=not FORCE_GUI,
    #         native=not FORCE_GUI,
    #         webrtc=not FORCE_GUI,
    #     ),
    #     env_num=1,
    #     metrics_save_path="none",
    #     task_configs=[
    #         SingleInferenceTaskCfg(
    #             scene_asset_path=SCENE_ASSET_PATH,
    #             robots=[piper_cfg, franka_cfg],
    #             objects=[place_pedestal_cfg, cube_cfg, *clutter_cfgs],
    #         )
    #     ],
    # )
    # import_extensions()
    # env = Env(config)
    config = build_env_config(
        robot_cfgs=[piper_cfg, franka_cfg],
        objects=[place_pedestal_cfg, cube_cfg, franka_cube_cfg, *clutter_cfgs],
    )
    env = start_multi_robot_env(config)

    try:
        env_obs, _ = env.reset()
        obs = env_obs[0]
        print(
            "multi robot room demo => "
            f"pipergo2_start={ROBOT_START}, "
            f"franka_start={FRANKA_START}, "
            f"cube={CUBE_POS}, "
            f"franka_cube={FRANKA_CUBE_POS}, "
            f"place_pedestal={PLACE_PEDESTAL_POS}, "
            f"staging={STAGING_WAYPOINT}, "
            f"bridge_api={type(bridge_api_placeholder).__name__}"
        )
        print(f"initial observation keys => {list(obs.keys())}")

        stage = env.runner._world.stage
        focus_default_view_on_robot(ROBOT_START[:2], robot_z=ROBOT_START[2])
        _disable_instances_and_add_collision(stage)

        set_mass_if_present(env, "pick_cube", 0.05)
        # Old Franka cube mass for rollback:
        # set_mass_if_present(env, "franka_pick_cube", 0.05)
        set_mass_if_present(env, "franka_pick_cube", FRANKA_CUBE_MASS)
        set_mass_if_present(env, "place_pedestal", 100.0)

        from omni.isaac.core.articulations import ArticulationView

        piper_view = ArticulationView(prim_paths_expr="/World/env_0/robots/pipergo2", name="pipergo2_view")
        franka_view = ArticulationView(prim_paths_expr="/World/env_0/robots/franka", name="franka_view")
        piper_view.initialize()
        franka_view.initialize()
        print(f"pipergo2 joints => {piper_view.dof_names}")
        print(f"franka joints => {franka_view.dof_names}")

        if SCENE_PREVIEW_STEPS > 0:
            print(
                "scene preview => "
                f"steps={SCENE_PREVIEW_STEPS}, "
                "window should stay open before navigation starts"
            )
            for _ in range(SCENE_PREVIEW_STEPS):
                env.step([{"pipergo2": {}, "franka": {}}])

        piper_env = SingleRobotEnvAdapter(env, "pipergo2", peer_robot_names=["franka"])
        franka_env = SingleRobotEnvAdapter(env, "franka", peer_robot_names=["pipergo2"])

        piper_api = PiperGo2ManipulationAPI(
            scene_asset_path=SCENE_ASSET_PATH,
            robot_cfg=piper_cfg,
            headless=not FORCE_GUI,
            # Old slower motion values for rollback:
            # pause_steps=90,
            # arm_settle_steps=90,
            # arm_motion_steps=150,
            pause_steps=45,
            arm_settle_steps=45,
            arm_motion_steps=90,
            navigation_offset=0.42,
            enable_arm_ik=True,
            allow_arm_heuristic_fallback=False,
        )
        piper_api._env = piper_env
        piper_api._robot_view = piper_view
        piper_api._joint_indices = {name: idx for idx, name in enumerate(piper_view.dof_names)}
        piper_api._active_grasp_joint_path = None
        piper_api._active_grasp_object_path = None

        franka_api = FrankaManipulationAPI(
            scene_asset_path=SCENE_ASSET_PATH,
            robot_cfg=franka_cfg,
            headless=not FORCE_GUI,
            # Old smoother-but-looser defaults for rollback:
            # pause_steps=60,
            # arm_waypoint_count=6,
            # Franka in the Merom room needs a little more time to finish
            # closing and settle before retreating, otherwise the cube is
            # often lifted and then immediately dropped.
            # Old slower values for rollback:
            # pause_steps=90,
            # gripper_settle_steps=60,
            # arm_waypoint_count=12,
            pause_steps=45,
            gripper_settle_steps=30,
            arm_waypoint_count=8,
        )
        franka_api._env = franka_env

        stabilize_robot(env, "pipergo2", ROBOT_START[:2], settle_steps=STABILIZE_STEPS)
        staged, obs = navigate_to_waypoint(
            env,
            "pipergo2",
            STAGING_WAYPOINT[:2],
            max_steps=STAGING_MAX_STEPS,
            label="staging",
        )
        if not staged:
            print("multi robot room demo aborted before next phase: staging navigation failed")
            return

        piper_pick_target = {
            "name": "piper_pick_site",
            "position": CUBE_POS,
            "pre_position": (CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + 0.08),
            "post_position": (CUBE_POS[0] - 0.18, CUBE_POS[1] - 0.05, CUBE_POS[2] + 0.18),
            "orientation": (1.0, 0.0, 0.0, 0.0),
            "metadata": {
                "object_name": "pick_cube",
            },
        }
        quat_down = (0.0, 0.0, 1.0, 0.0)
        franka_pick_target = {
            "name": "franka_pick_site",
            "position": FRANKA_CUBE_POS,
            # Relative-offset version kept for rollback:
            # "pre_position": (
            #     FRANKA_START[0] + FRANKA_PRE_OFFSET[0],
            #     FRANKA_START[1] + FRANKA_PRE_OFFSET[1],
            #     FRANKA_START[2] + FRANKA_PRE_OFFSET[2],
            # ),
            # "post_position": (
            #     FRANKA_START[0] + FRANKA_PRE_OFFSET[0],
            #     FRANKA_START[1] + FRANKA_PRE_OFFSET[1],
            #     FRANKA_START[2] + FRANKA_PRE_OFFSET[2],
            # ),
            # Old pre/post for rollback:
            # "pre_position": (FRANKA_CUBE_POS[0], FRANKA_CUBE_POS[1], FRANKA_CUBE_POS[2] + 0.20),
            # "post_position": (FRANKA_CUBE_POS[0], FRANKA_CUBE_POS[1], FRANKA_CUBE_POS[2] + 0.20),
            "pre_position": (FRANKA_CUBE_POS[0], FRANKA_CUBE_POS[1], FRANKA_CUBE_POS[2] + 0.18),
            "post_position": (FRANKA_CUBE_POS[0], FRANKA_CUBE_POS[1], FRANKA_CUBE_POS[2] + 0.12),
            "orientation": quat_down,
        }
        franka_place_target = {
            "name": "franka_place_site",
            # Old direct tuple usage for rollback:
            # "position": FRANKA_PLACE_POS,
            "position": FRANKA_PLACE_SURFACE_POS,
            # Old no-release flow for rollback:
            # Franka used to stop after pick only.
            # Old pre-position for rollback:
            # "pre_position": (FRANKA_PLACE_POS[0], FRANKA_PLACE_POS[1], FRANKA_PLACE_POS[2] + 0.18),
            # Keep pre/target/post on the same x/y column so release reads as a
            # purely vertical place-and-lift motion.
            "pre_position": FRANKA_PLACE_LIFT_POS,
            # Old split-lift behavior for rollback:
            # "post_position": FRANKA_PLACE_POS,
            # Use the in-release vertical lift again so place+lift stay inside a
            # single Franka release() motion.
            "post_position": FRANKA_PLACE_LIFT_POS,
            "orientation": quat_down,
        }

        piper_pose = get_robot_obs(obs, "pipergo2")["position"]
        franka_pose = get_robot_obs(obs, "franka")["position"]
        print(
            "staging summary => "
            f"pipergo2=({piper_pose[0]:.3f}, {piper_pose[1]:.3f}, {piper_pose[2]:.3f}), "
            f"franka=({franka_pose[0]:.3f}, {franka_pose[1]:.3f}, {franka_pose[2]:.3f})"
        )

        piper_pick_result = piper_api.pick(
            piper_pick_target,
            dump_path=OUTPUT_DIR / "test3_multi_piper_pick.json",
        )
        print(
            "piper_pick => "
            f"success={piper_pick_result.success}, "
            f"steps={piper_pick_result.steps}, "
            f"log={OUTPUT_DIR / 'test3_multi_piper_pick.json'}"
        )

        franka_pick_result = franka_api.pick(
            franka_pick_target,
            dump_path=OUTPUT_DIR / "test3_multi_franka_pick.json",
        )
        print(
            "franka_pick => "
            f"success={franka_pick_result.success}, "
            f"steps={franka_pick_result.steps}, "
            f"log={OUTPUT_DIR / 'test3_multi_franka_pick.json'}"
        )
        if franka_pick_result.success:
            # Old behavior for rollback:
            # reuse the same pause_steps for both pick and release
            original_franka_pause_steps = franka_api.pause_steps
            original_franka_waypoint_count = franka_api.arm_waypoint_count
            franka_api.pause_steps = FRANKA_RELEASE_PAUSE_STEPS
            # Old release speed for rollback:
            # use the same arm_waypoint_count for pick and release
            franka_api.arm_waypoint_count = FRANKA_RELEASE_WAYPOINT_COUNT
            franka_place_result = franka_api.release(
                franka_place_target,
                dump_path=OUTPUT_DIR / "test3_multi_franka_release.json",
            )
            franka_api.pause_steps = original_franka_pause_steps
            franka_api.arm_waypoint_count = original_franka_waypoint_count
            print(
                "franka_release => "
                f"success={franka_place_result.success}, "
                f"steps={franka_place_result.steps}, "
                f"log={OUTPUT_DIR / 'test3_multi_franka_release.json'}"
            )

        # Old final hold for rollback:
        # for _ in range(240):
        for _ in range(90):
            env.step([{"pipergo2": {}, "franka": {}}])
    finally:
        env.close()


if __name__ == "__main__":
    main()
