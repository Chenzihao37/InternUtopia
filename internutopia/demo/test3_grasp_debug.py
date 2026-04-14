from pathlib import Path

import numpy as np

from internutopia.bridge import (
    PiperGo2ManipulationAPI,
    create_pipergo2_robot_cfg,
)
from internutopia_extension.configs.objects import DynamicCubeCfg, VisualCubeCfg
from internutopia_extension.configs.robots.pipergo2 import move_to_point_cfg


SCENE_ASSET_PATH = "/home/zyserver/work/my_project/InternUtopia/internutopia/demo/merom_scene_baked.usd"
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"

ROBOT_START = (0.58, 7.50704, 0.55)
STAGING_WAYPOINT = (2.74051, 7.67599, 0.0)
# Old nearby pedestal position for rollback:
# PLACE_PEDESTAL_POS = (1.02, 7.28500, 0.16)
PLACE_PEDESTAL_POS = (1.02, 7.08000, 0.16)
PLACE_PEDESTAL_SCALE = (0.14, 0.14, 0.32)
# Old cube positions for rollback:
# CUBE_POS = (3.46261, 7.67599, 0.29768)
# CUBE_POS = (3.46261, 7.67599, 0.37768)
CUBE_POS = (3.46261, 7.67599, 0.41172)
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
# Old pedestal settings for rollback:
# PICK_PEDESTAL_POS = (CUBE_POS[0], CUBE_POS[1], 0.14884)
# PICK_PEDESTAL_SCALE = (0.22, 0.22, 0.29768)
FORCE_GUI = True
SCENE_PREVIEW_STEPS = 240
STABILIZE_STEPS = 120
STAGING_MAX_STEPS = 1200

try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
    except ImportError:
        set_camera_view = None


def make_marker(name: str, position: tuple[float, float, float], color: tuple[float, float, float]) -> VisualCubeCfg:
    return VisualCubeCfg(
        name=name,
        prim_path=f"/World/{name}",
        position=position,
        scale=(0.04, 0.04, 0.04),
        color=list(color),
    )


def _disable_instances_and_add_collision(stage):
    from pxr import PhysxSchema, Usd, UsdPhysics

    # Match the Merom-scene collision patching strategy we already validated in
    # test3_debug.py: only patch the scene subtree, and do not trigger a second
    # env.reset() afterwards.
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

    # Match the room-view framing strategy from test3_debug.py so the user
    # immediately sees the scene and the robot in the main viewport.
    eye = [robot_xy[0] - 2.8, robot_xy[1] - 2.2, robot_z + 1.8]
    target = [robot_xy[0], robot_xy[1], max(robot_z - 0.4, 0.2)]

    set_camera_view(
        eye=eye,
        target=target,
        camera_prim_path="/OmniverseKit_Persp",
    )
    print(f"Default viewport focused => eye={eye}, target={target}")


def get_robot_obs(obs_data):
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


def stabilize_robot(env, target_xy, settle_steps=STABILIZE_STEPS):
    idle_action = {
        move_to_point_cfg.name: [(float(target_xy[0]), float(target_xy[1]), 0.0)],
    }
    obs = None
    for step in range(settle_steps):
        obs, _, terminated, _, _ = env.step(action=idle_action)
        if step % 30 == 0:
            robot_obs = get_robot_obs(obs)
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


def navigate_to_waypoint(env, waypoint_xy, max_steps=STAGING_MAX_STEPS, threshold=0.10, label="staging"):
    goal_action = {
        move_to_point_cfg.name: [(float(waypoint_xy[0]), float(waypoint_xy[1]), 0.0)],
    }
    for step in range(max_steps):
        obs, _, terminated, _, _ = env.step(action=goal_action)
        robot_obs = get_robot_obs(obs)
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
        episode_terminated = terminated[0] if isinstance(terminated, (list, tuple)) else bool(terminated)
        if episode_terminated:
            print(f"{label} terminated early")
            return False, obs
    print(f"{label} failed => dist={dist:.4f}")
    return False, obs


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    cube_cfg = DynamicCubeCfg(
        name="pick_cube",
        prim_path="/World/pick_cube",
        position=CUBE_POS,
        scale=(0.05, 0.05, 0.05),
        color=(0.85, 0.2, 0.2),
    )
    place_pedestal_cfg = DynamicCubeCfg(
        name="place_pedestal",
        prim_path="/World/place_pedestal",
        position=PLACE_PEDESTAL_POS,
        scale=PLACE_PEDESTAL_SCALE,
        color=(0.42, 0.42, 0.46),
    )
    # Old dynamic clutter version for rollback:
    # clutter_cfgs = [
    #     DynamicCubeCfg(
    #         name=item["name"],
    #         prim_path=f"/World/{item['name']}",
    #         position=item["position"],
    #         scale=item["scale"],
    #         color=item["color"],
    #     )
    #     for item in NEARBY_CLUTTER
    # ]
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
    # Old support-platform version for rollback:
    # spawn_support_cfg = DynamicCubeCfg(
    #     name="spawn_support",
    #     prim_path="/World/spawn_support",
    #     position=SPAWN_SUPPORT_POS,
    #     scale=SPAWN_SUPPORT_SCALE,
    #     color=(0.25, 0.25, 0.25),
    # )
    # Old pedestal version for rollback:
    # pick_pedestal_cfg = DynamicCubeCfg(
    #     name="pick_pedestal",
    #     prim_path="/World/pick_pedestal",
    #     position=PICK_PEDESTAL_POS,
    #     scale=PICK_PEDESTAL_SCALE,
    #     color=(0.5, 0.5, 0.5),
    # )

    pick_target = {
        "name": "room_pick_site",
        "position": CUBE_POS,
        "pre_position": (CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + 0.08),
        "post_position": (CUBE_POS[0] - 0.18, CUBE_POS[1] - 0.05, CUBE_POS[2] + 0.18),
        "orientation": (1.0, 0.0, 0.0, 0.0),
        "metadata": {
            "object_name": "pick_cube",
        },
    }
    place_surface_z = PLACE_PEDESTAL_POS[2] + PLACE_PEDESTAL_SCALE[2] * 0.5
    cube_half_z = cube_cfg.scale[2] * 0.5
    place_target = {
        "name": "home_place_site",
        "position": (
            PLACE_PEDESTAL_POS[0],
            PLACE_PEDESTAL_POS[1],
            place_surface_z + cube_half_z,
        ),
        # Old higher pre-position for rollback:
        # "pre_position": (
        #     PLACE_PEDESTAL_POS[0],
        #     PLACE_PEDESTAL_POS[1],
        #     place_surface_z + cube_half_z + 0.10,
        # ),
        "pre_position": (
            PLACE_PEDESTAL_POS[0],
            PLACE_PEDESTAL_POS[1],
            place_surface_z + cube_half_z + 0.07,
        ),
        # Old short retreat for rollback:
        # "post_position": (
        #     PLACE_PEDESTAL_POS[0] - 0.12,
        #     PLACE_PEDESTAL_POS[1] + 0.02,
        #     place_surface_z + cube_half_z + 0.16,
        # ),
        "post_position": (
            PLACE_PEDESTAL_POS[0] - 0.18,
            PLACE_PEDESTAL_POS[1] + 0.07,
            place_surface_z + cube_half_z + 0.18,
        ),
        "orientation": (1.0, 0.0, 0.0, 0.0),
        "metadata": {
            "object_name": "pick_cube",
        },
    }

    # Old debug marker cubes for rollback:
    # marker_cfgs = [
    #     make_marker("room_pick_pre_marker", pick_target["pre_position"], (0.2, 0.8, 1.0)),
    #     make_marker("room_pick_target_marker", pick_target["position"], (0.1, 0.95, 0.2)),
    #     make_marker("room_pick_post_marker", pick_target["post_position"], (1.0, 0.75, 0.15)),
    # ]
    #
    # pick_target["metadata"]["debug_markers"] = [
    #     {"name": "room_pick_pre_marker", "position": pick_target["pre_position"]},
    #     {"name": "room_pick_target_marker", "position": pick_target["position"]},
    #     {"name": "room_pick_post_marker", "position": pick_target["post_position"]},
    # ]

    robot_cfg = create_pipergo2_robot_cfg(position=ROBOT_START, arm_mass_scale=0.25)
    api = PiperGo2ManipulationAPI(
        scene_asset_path=SCENE_ASSET_PATH,
        robot_cfg=robot_cfg,
        # Old multi-cube debug version for rollback:
        # objects=[pick_pedestal_cfg, cube_cfg, *marker_cfgs],
        # objects=[cube_cfg, *marker_cfgs],
        objects=[place_pedestal_cfg, cube_cfg, *clutter_cfgs],
        # Old behavior kept for rollback:
        # headless=None,
        # Let PiperGo2ManipulationAPI auto-detect DISPLAY/headless state.
        # In the user's current launch environment this often resolved to a
        # non-visual run, so we explicitly force GUI mode for this room demo.
        headless=not FORCE_GUI,
        pause_steps=90,
        arm_settle_steps=90,
        arm_motion_steps=150,
        navigation_offset=0.42,
        enable_arm_ik=True,
        allow_arm_heuristic_fallback=False,
    )

    obs = api.start()
    print(f"======== INIT OBS {obs} =========")
    print(
        "room grasp demo => "
        f"robot_start={ROBOT_START}, cube={CUBE_POS}, "
        f"place_pedestal={PLACE_PEDESTAL_POS}, "
        f"staging={STAGING_WAYPOINT}, "
        f"pick_pre={pick_target['pre_position']}, "
        f"pick_post={pick_target['post_position']}, "
        f"place={place_target['position']}"
    )

    # Old no-patch version for rollback:
    # The first Merom-room grasp attempt failed before pick because PiperGo2
    # fell straight through the room floor. Reuse the scene-only collision patch
    # from test3_debug.py here before starting navigation.
    stage = api._env.runner._world.stage
    focus_default_view_on_robot(ROBOT_START[:2], robot_z=ROBOT_START[2])
    _disable_instances_and_add_collision(stage)

    def set_mass_if_present(object_name: str, mass: float) -> None:
        try:
            obj = api._env.runner.get_obj(object_name)
        except Exception:
            return
        obj.set_mass(mass)
        print(f"set_mass => object={object_name}, mass={mass}")

    set_mass_if_present("pick_cube", 0.05)
    set_mass_if_present("place_pedestal", 100.0)
    # Old dynamic clutter mass tuning for rollback:
    # for clutter in NEARBY_CLUTTER:
    #     set_mass_if_present(clutter["name"], 0.08)
    # Old pedestal mass tuning for rollback:
    # set_mass_if_present("pick_pedestal", 100.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Give the room scene a moment to render before the robot starts moving so
    # the GUI window has time to come up and the user can actually see it.
    if SCENE_PREVIEW_STEPS > 0:
        print(
            "scene preview => "
            f"steps={SCENE_PREVIEW_STEPS}, "
            "window should stay open before pick starts"
        )
        for _ in range(SCENE_PREVIEW_STEPS):
            api._env.step({})

    stabilize_robot(api._env, ROBOT_START[:2], settle_steps=STABILIZE_STEPS)
    staged, _ = navigate_to_waypoint(
        api._env,
        STAGING_WAYPOINT[:2],
        max_steps=STAGING_MAX_STEPS,
        label="staging",
    )
    if not staged:
        print("room grasp demo aborted before pick: staging navigation failed")
        api.close()
        return

    try:
        pick_result = api.pick(
            pick_target,
            dump_path=OUTPUT_DIR / "test3_room_pick.json",
        )
        print(
            "pick => "
            f"success={pick_result.success}, "
            f"steps={pick_result.steps}, "
            f"log={OUTPUT_DIR / 'test3_room_pick.json'}"
        )
        if pick_result.success:
            place_result = api.release(
                place_target,
                dump_path=OUTPUT_DIR / "test3_room_place.json",
            )
            print(
                "place => "
                f"success={place_result.success}, "
                f"steps={place_result.steps}, "
                f"log={OUTPUT_DIR / 'test3_room_place.json'}"
            )
            # Old direct-return-after-pick behavior for rollback:
            # returned, _ = navigate_to_waypoint(
            #     api._env,
            #     ROBOT_START[:2],
            #     max_steps=STAGING_MAX_STEPS,
            #     label="return_home",
            # )
            returned, _ = navigate_to_waypoint(
                api._env,
                ROBOT_START[:2],
                max_steps=STAGING_MAX_STEPS,
                label="return_home",
            )
            print(f"return_home => success={returned}, target={ROBOT_START[:2]}")
    finally:
        api.close()


if __name__ == "__main__":
    main()
