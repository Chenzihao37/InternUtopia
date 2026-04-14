from pathlib import Path
import time

from internutopia.bridge import PiperGo2ManipulationAPI, create_pipergo2_robot_cfg
from internutopia_extension.configs.objects import VisualCubeCfg


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")


def make_marker(name: str, position: tuple[float, float, float], color: tuple[float, float, float]) -> VisualCubeCfg:
    return VisualCubeCfg(
        name=name,
        prim_path=f"/World/{name}",
        position=position,
        scale=(0.035, 0.035, 0.035),
        color=list(color),
    )


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    target = {
        "name": "ik_loop_site",
        "position": (0.30, 0.00, 0.80),
        "pre_position": (0.30, 0.00, 0.90),
        "post_position": (0.24, 0.00, 0.88),
        "orientation": (1.0, 0.0, 0.0, 0.0),
        "metadata": {
            "debug_markers": [
                {"name": "loop_pre_marker", "position": (0.30, 0.00, 0.90)},
                {"name": "loop_target_marker", "position": (0.30, 0.00, 0.80)},
                {"name": "loop_post_marker", "position": (0.24, 0.00, 0.88)},
            ],
        },
    }

    marker_cfgs = [
        make_marker("loop_pre_marker", target["pre_position"], (0.2, 0.8, 1.0)),
        make_marker("loop_target_marker", target["position"], (0.1, 0.95, 0.2)),
        make_marker("loop_post_marker", target["post_position"], (1.0, 0.75, 0.15)),
    ]

    api = PiperGo2ManipulationAPI(
        scene_asset_path=SCENE_ASSET_PATH,
        robot_cfg=create_pipergo2_robot_cfg(position=(0.0, 0.0, 0.55), arm_mass_scale=0.25),
        objects=marker_cfgs,
        pause_steps=30,
        arm_settle_steps=90,
        arm_motion_steps=120,
        navigation_offset=0.42,
        enable_arm_ik=True,
        allow_arm_heuristic_fallback=False,
    )

    try:
        obs = api.start()
        print(f"======== INIT OBS {obs} =========")

        base_position = obs.get("position", (0.0, 0.0, 0.0))
        target["metadata"]["base_position"] = (
            float(base_position[0]),
            float(base_position[1]),
            0.0,
        )
        print(f"locked base => {target['metadata']['base_position']}")
        print("Looping forever. Press Ctrl+C to stop.")

        cycle_index = 0
        while True:
            cycle_index += 1
            print(f"\n===== cycle {cycle_index} =====")

            pre_pose = api._plan_arm_pose(
                target["pre_position"],
                gripper_open=True,
                orientation=target["orientation"],
            )
            ok, _, final_obs = api._run_arm_motion(pre_pose, "approach", api._coerce_manipulation_target(target))
            if not ok:
                raise RuntimeError("approach failed")

            target_pose = api._plan_arm_pose(
                target["position"],
                gripper_open=True,
                orientation=target["orientation"],
            )
            ok, _, final_obs = api._run_arm_motion(target_pose, "target", api._coerce_manipulation_target(target))
            if not ok:
                raise RuntimeError("target failed")

            post_pose = api._plan_arm_pose(
                target["post_position"],
                gripper_open=True,
                orientation=target["orientation"],
            )
            ok, _, final_obs = api._run_arm_motion(post_pose, "retreat", api._coerce_manipulation_target(target))
            if not ok:
                raise RuntimeError("retreat failed")

            rest_pose = api._get_rest_arm_pose(gripper_open=True)
            ok, _, final_obs = api._run_arm_motion(rest_pose, "rest", api._coerce_manipulation_target(target))
            if not ok:
                raise RuntimeError("rest failed")

            print(f"cycle {cycle_index} final eef => {final_obs.get('eef_position')}")
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nmanual stop")
    finally:
        print("leaving simulation open ended by process exit")


if __name__ == "__main__":
    main()
