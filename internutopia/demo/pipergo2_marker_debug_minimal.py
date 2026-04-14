from pathlib import Path
import traceback

from internutopia.bridge import PiperGo2ManipulationAPI, create_pipergo2_robot_cfg
from internutopia_extension.configs.objects import VisualCubeCfg
import time


# ======================
# 场景路径
# ======================
ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")


# ======================
# 创建调试 marker
# ======================
def make_marker(
    name: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> VisualCubeCfg:
    return VisualCubeCfg(
        name=name,
        prim_path=f"/World/{name}",
        position=position,
        scale=(0.035, 0.035, 0.035),
        color=list(color),
    )


def spawn_eef_marker(prim_path: str, position: tuple[float, float, float], color: tuple[float, float, float], radius: float = 0.025):
    import omni
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(radius)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(tuple(float(v) for v in position))
    return sphere.GetPrim()


# ======================
# 主函数
# ======================
def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    # ======================
    # ✅ IK测试目标
    # ======================
    target = {
        "name": "ik_debug_site",

        # ✅ 合理测试点
        "position": (0.30, 0.00, 0.60),

        "pre_position": (0.30, 0.00, 0.70),
        "post_position": (0.25, 0.00, 0.70),

        # 先用单位四元数，表示不额外旋转
        "orientation": (1.0, 0.0, 0.0, 0.0),

        "metadata": {
            "debug_markers": [
                {"name": "debug_pre_marker", "position": (0.30, 0.00, 0.70)},
                {"name": "debug_target_marker", "position": (0.30, 0.00, 0.60)},
                {"name": "debug_post_marker", "position": (0.25, 0.00, 0.70)},
            ],
        },
    }

    # ======================
    # Debug 可视化方块
    # ======================
    marker_cfgs = [
        make_marker(
            "debug_pre_marker",
            target["pre_position"],
            (0.2, 0.8, 1.0)   # 浅蓝色 / 青蓝色 � 表示「前置位置」（pre）
        ),
        make_marker(
            "debug_target_marker",
            target["position"],
            (0.1, 0.95, 0.2)  # 亮绿色 � 表示「目标位置」（target）
        ),
        make_marker(
            "debug_post_marker",
            target["post_position"],
            (1.0, 0.75, 0.15) # 橙黄色 � 表示「后置位置」（post）
        ),
    ]

    # ======================
    # 初始化 API
    # ======================
    api = PiperGo2ManipulationAPI(
        scene_asset_path=SCENE_ASSET_PATH,

        robot_cfg=create_pipergo2_robot_cfg(
            position=(0.0, 0.0, 0.55),
            arm_mass_scale=0.25,
        ),

        objects=marker_cfgs,

        # 时间控制
        pause_steps=60,
        arm_settle_steps=90,
        arm_motion_steps=120,

        # � 锁 base（IK-only）
        navigation_offset=0.42,

        enable_arm_ik=True,
        allow_arm_heuristic_fallback=False,
    )

    # ======================
    # 执行
    # ======================
    try:
        obs = api.start()
        print(f"\n======== INIT OBS =========\n{obs}\n")

        init_eef_position = tuple(float(v) for v in obs.get("eef_position", (0.0, 0.0, 0.0)))
        spawn_eef_marker(
            "/World/debug_eef_init_marker",
            init_eef_position,
            (0.85, 0.1, 0.1),
            radius=0.022,
        )
        print(f"initial eef marker => {init_eef_position}")
        print("live eef marker => /World/debug_eef_live_marker (bright green sphere following code eef)")

        # � 锁底盘
        base_position = obs.get("position", (0.0, 0.0, 0.0))
        locked_base = (
            float(base_position[0]),
            float(base_position[1]),
            0.0,
        )
        target["metadata"]["base_position"] = locked_base

        print(f"locked base => {locked_base}")
        print("� IK-only test (no navigation)\n")

        # 执行
        result = api.pick(target)

        print("\n======== RESULT =========")
        print(f"success = {result.success}")
        print(f"steps   = {result.steps}")
        final_eef_position = tuple(float(v) for v in result.final_observation.get("eef_position", (0.0, 0.0, 0.0)))
        spawn_eef_marker(
            "/World/debug_eef_final_marker",
            final_eef_position,
            (1.0, 0.95, 0.1),
            radius=0.024,
        )
        print(f"final eef marker => {final_eef_position}")

    except Exception:
        print("\n� DEMO EXCEPTION:")
        traceback.print_exc()

    finally:
        print("\nleaving simulation open for inspection... (Ctrl+C to exit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nmanual exit")


# ======================
# 入口
# ======================
if __name__ == "__main__":
    main()
