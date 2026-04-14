from pathlib import Path

from internutopia.bridge import (
    PiperGo2ManipulationAPI,
    create_pipergo2_robot_cfg,
)
from internutopia_extension.configs.objects import DynamicCubeCfg, VisualCubeCfg


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"


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

    cube_cfg = DynamicCubeCfg(
        name="pick_cube",
        prim_path="/World/pick_cube",
        position=(1.18, 0.00, 0.675),
        scale=(0.05, 0.05, 0.05),
        color=(0.85, 0.2, 0.2),
    )
    pedestal_cfg = DynamicCubeCfg(
        name="pick_pedestal",
        prim_path="/World/pick_pedestal",
        position=(1.18, 0.00, 0.325),
        scale=(0.18, 0.18, 0.65),
        color=(0.5, 0.5, 0.5),
    )

    pick_target = {
        "name": "pick_site",
        "position": (1.18, 0.00, 0.675),
        "pre_position": (1.18, 0.00, 0.745),
        "post_position": (1.02, 0.00, 0.86),
        "orientation": (1.0, 0.0, 0.0, 0.0),
        "metadata": {
            "object_name": "pick_cube",
        },
    }
    place_target = {
        "name": "place_site",
        "position": (1.18, -0.25, 0.675),
        "pre_position": (1.18, -0.25, 0.745),
        "post_position": (1.00, -0.18, 0.86),
        "orientation": (1.0, 0.0, 0.0, 0.0),
    }

    marker_cfgs = [
        # ===== Pick 阶段 =====
        
        # 浅蓝色（偏青色）- 表示抓取前的预位置（pre）
        make_marker("pick_pre_marker", pick_target["pre_position"], (0.2, 0.8, 1.0)),
        
        # # 亮绿色 - 表示实际抓取目标点（target）
        # make_marker("pick_target_marker", pick_target["position"], (0.1, 0.95, 0.2)),
        
        # 橙黄色 - 表示抓取后的离开位置（post）
        make_marker("pick_post_marker", pick_target["post_position"], (1.0, 0.75, 0.15)),

        
        # ===== Place 阶段 =====
        
        # 蓝紫色（偏冷色）- 表示放置前的预位置（pre）
        make_marker("place_pre_marker", place_target["pre_position"], (0.35, 0.55, 1.0)),
        
        # 粉紫色（洋红）- 表示放置目标点（target）
        make_marker("place_target_marker", place_target["position"], (0.95, 0.25, 0.95)),
        
        # 橙红色 - 表示放置后的离开位置（post）
        make_marker("place_post_marker", place_target["post_position"], (1.0, 0.45, 0.2)),
    ]

    pick_target["metadata"]["debug_markers"] = [
        {"name": "pick_pre_marker", "position": pick_target["pre_position"]},
        # {"name": "pick_target_marker", "position": pick_target["position"]},
        {"name": "pick_post_marker", "position": pick_target["post_position"]},
        {"name": "place_pre_marker", "position": place_target["pre_position"]},
        {"name": "place_target_marker", "position": place_target["position"]},
        {"name": "place_post_marker", "position": place_target["post_position"]},
    ]
    place_target["metadata"] = {
        "debug_markers": pick_target["metadata"]["debug_markers"],
    }

    robot_cfg = create_pipergo2_robot_cfg(position=(0.0, 0.0, 0.55), arm_mass_scale=0.25) # 位置+机械臂重量
    api = PiperGo2ManipulationAPI(
        scene_asset_path=SCENE_ASSET_PATH,
        robot_cfg=robot_cfg,
        objects=[pedestal_cfg, cube_cfg, *marker_cfgs],
        pause_steps=90, # 小阶段停留
        arm_settle_steps=90, # 大阶段停留
        arm_motion_steps=150, # 两点间切分移动步数
        navigation_offset=0.42, # 底盘安全距离
        enable_arm_ik=True,
        allow_arm_heuristic_fallback=False,
    )

    obs = api.start()
    print(f"======== INIT OBS {obs} =========")
    print("markers => pick: pre=blue target=green post=orange; place: pre=blue-purple target=magenta post=orange-red")

    def set_mass_if_present(object_name: str, mass: float) -> None:
        try:
            obj = api._env.runner.get_obj(object_name)
        except Exception:
            return
        obj.set_mass(mass)
        print(f"set_mass => object={object_name}, mass={mass}")

    # Keep the grasped cube light enough to carry, while making support pillars
    # heavy so incidental contacts are less likely to launch them away.
    set_mass_if_present("pick_cube", 0.05)
    set_mass_if_present("pick_pedestal", 100.0)
    set_mass_if_present("place_pedestal", 100.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pick_result = api.pick(
            pick_target,
            dump_path=OUTPUT_DIR / "pipergo2_pick.json",
        )
        print(
            "pick => "
            f"success={pick_result.success}, "
            f"steps={pick_result.steps}, "
            f"log={OUTPUT_DIR / 'pipergo2_pick.json'}"
        )

        if pick_result.success:
            place_result = api.release(
                place_target,
                dump_path=OUTPUT_DIR / "pipergo2_release.json",
            )
            print(
                "release => "
                f"success={place_result.success}, "
                f"steps={place_result.steps}, "
                f"log={OUTPUT_DIR / 'pipergo2_release.json'}"
            )
    finally:
        api.close()


if __name__ == "__main__":
    main()
