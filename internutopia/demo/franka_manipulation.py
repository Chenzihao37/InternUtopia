from pathlib import Path

from internutopia.bridge import (
    FrankaManipulationAPI,
    create_franka_robot_cfg,
)


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
WAREHOUSE_PATH = "/home/zyserver/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    robot_cfg = create_franka_robot_cfg(position=(0.0, 0.0, 0.0))
    api = FrankaManipulationAPI(
        # scene_asset_path=SCENE_ASSET_PATH,
        scene_asset_path=WAREHOUSE_PATH,
        robot_cfg=robot_cfg,
        pause_steps=60,
        arm_waypoint_count=6,
    )

    obs = api.start()
    print(f"======== INIT OBS {obs} =========")

    # 末端朝下的抓取姿态，数值沿用你原 demo 的用法。
    quat_down = (0.0, 0.0, 1.0, 0.0)

    pick_target = {
        "name": "pick_site",
        "position": (0.40, 0.00, 0.10),
        "pre_position": (0.40, 0.00, 0.30),
        "post_position": (0.40, 0.00, 0.30),
        "orientation": quat_down,
    }
    place_target = {
        "name": "place_site",
        "position": (0.40, -0.30, 0.10),
        "pre_position": (0.40, -0.30, 0.30),
        "post_position": (0.40, -0.30, 0.30),
        "orientation": quat_down,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pick_result = api.pick(
            pick_target,
            dump_path=OUTPUT_DIR / "franka_pick.json",
        )
        print(
            "pick => "
            f"success={pick_result.success}, "
            f"steps={pick_result.steps}, "
            f"log={OUTPUT_DIR / 'franka_pick.json'}"
        )

        if pick_result.success:
            place_result = api.release(
                place_target,
                dump_path=OUTPUT_DIR / "franka_release.json",
            )
            print(
                "release => "
                f"success={place_result.success}, "
                f"steps={place_result.steps}, "
                f"log={OUTPUT_DIR / 'franka_release.json'}"
            )
    finally:
        api.close()


if __name__ == "__main__":
    main()
