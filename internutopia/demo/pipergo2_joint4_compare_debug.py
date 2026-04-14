from pathlib import Path
import traceback
import time

from internutopia.bridge import PiperGo2ManipulationAPI, create_pipergo2_robot_cfg
from internutopia_extension.configs.objects import VisualCubeCfg


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")


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


def coerce_target(api: PiperGo2ManipulationAPI, target_dict: dict):
    return api._coerce_manipulation_target(target_dict)


def run_pose_sequence(
    api: PiperGo2ManipulationAPI,
    target_dict: dict,
    *,
    label: str,
    flip_joint4: bool,
):
    target = coerce_target(api, target_dict)
    print(f"\n===== {label} =====")

    pre_pose = api._plan_arm_pose(
        target.pre_position or target.position,
        gripper_open=True,
        orientation=target.orientation,
    )
    target_pose = api._plan_arm_pose(
        target.position,
        gripper_open=True,
        orientation=target.orientation,
    )
    post_pose = api._plan_arm_pose(
        target.post_position or target.position,
        gripper_open=True,
        orientation=target.orientation,
    )

    if flip_joint4:
        pre_pose = pre_pose.copy()
        target_pose = target_pose.copy()
        post_pose = post_pose.copy()
        pre_pose[3] = -pre_pose[3]
        target_pose[3] = -target_pose[3]
        post_pose[3] = -post_pose[3]

    print(
        "joint targets => "
        f"pre_j4={pre_pose[3]:+.4f}, "
        f"target_j4={target_pose[3]:+.4f}, "
        f"post_j4={post_pose[3]:+.4f}"
    )

    ok, _, final_obs = api._run_arm_motion(pre_pose, f"{label}_approach", target)
    if not ok:
        raise RuntimeError(f"{label} approach failed")
    ok, _, final_obs = api._run_arm_motion(target_pose, f"{label}_target", target)
    if not ok:
        raise RuntimeError(f"{label} target failed")
    ok, _, final_obs = api._run_arm_motion(post_pose, f"{label}_retreat", target)
    if not ok:
        raise RuntimeError(f"{label} retreat failed")
    return final_obs


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    target = {
        "name": "ik_debug_site",
        "position": (0.30, 0.00, 0.60),
        "pre_position": (0.30, 0.00, 0.70),
        "post_position": (0.25, 0.00, 0.70),
        "orientation": (1.0, 0.0, 0.0, 0.0),
        "metadata": {
            "debug_markers": [
                {"name": "debug_pre_marker", "position": (0.30, 0.00, 0.70)},
                {"name": "debug_target_marker", "position": (0.30, 0.00, 0.60)},
                {"name": "debug_post_marker", "position": (0.25, 0.00, 0.70)},
            ],
        },
    }

    marker_cfgs = [
        make_marker("debug_pre_marker", target["pre_position"], (0.2, 0.8, 1.0)),
        make_marker("debug_target_marker", target["position"], (0.1, 0.95, 0.2)),
        make_marker("debug_post_marker", target["post_position"], (1.0, 0.75, 0.15)),
    ]

    api = PiperGo2ManipulationAPI(
        scene_asset_path=SCENE_ASSET_PATH,
        robot_cfg=create_pipergo2_robot_cfg(position=(0.0, 0.0, 0.55), arm_mass_scale=0.25),
        objects=marker_cfgs,
        pause_steps=60,
        arm_settle_steps=90,
        arm_motion_steps=120,
        navigation_offset=0.42,
        enable_arm_ik=True,
        allow_arm_heuristic_fallback=False,
    )

    try:
        obs = api.start()
        print(f"\n======== INIT OBS =========\n{obs}\n")

        init_eef_position = tuple(float(v) for v in obs.get("eef_position", (0.0, 0.0, 0.0)))
        spawn_eef_marker("/World/debug_eef_init_marker", init_eef_position, (0.85, 0.1, 0.1), radius=0.022)
        print(f"initial eef marker => {init_eef_position}")
        print("live eef marker => /World/debug_eef_live_marker (bright green sphere following code eef)")

        base_position = obs.get("position", (0.0, 0.0, 0.0))
        locked_base = (float(base_position[0]), float(base_position[1]), 0.0)
        target["metadata"]["base_position"] = locked_base

        print(f"locked base => {locked_base}")
        print("IK-only compare test: original IK vs joint4-sign-flipped IK")
        print("base should stay fixed; this script does not intentionally move the quadruped.")

        target_obj = coerce_target(api, target)
        rest_pose = api._get_rest_arm_pose(gripper_open=True)

        final_obs = run_pose_sequence(api, target, label="original_ik", flip_joint4=False)
        api._run_arm_motion(rest_pose, "return_to_rest", target_obj)
        final_obs = run_pose_sequence(api, target, label="joint4_flipped", flip_joint4=True)

        print("\n======== RESULT =========")
        print("Compare the two phases in the viewport:")
        print("1. original_ik")
        print("2. joint4_flipped")
        print("If only the flipped one looks geometrically correct, joint4 sign is very likely the issue.")
        final_eef_position = tuple(float(v) for v in final_obs.get("eef_position", (0.0, 0.0, 0.0)))
        spawn_eef_marker("/World/debug_eef_final_marker", final_eef_position, (1.0, 0.95, 0.1), radius=0.024)
        print(f"final eef marker => {final_eef_position}")

    except Exception:
        print("\nDEMO EXCEPTION:")
        traceback.print_exc()

    finally:
        print("\nleaving simulation open for inspection... (Ctrl+C to exit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nmanual exit")


if __name__ == "__main__":
    main()
