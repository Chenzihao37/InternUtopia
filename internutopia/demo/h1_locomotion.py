import json
from pathlib import Path

from internutopia.bridge import (
    MultiRobotNavigateAPI,
    create_aliengo_robot_cfg,
    create_g1_robot_cfg,
    create_h1_robot_cfg,
)


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
USE_EMPTY_SCENE = False
EMPTY_SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")
GRSCENES_USD_DIR = "scenes/GRScenes-100/commercial_scenes/scenes/MV5M25QKTKJZ2AABAAAAAAA8_usd"
GRSCENES_USD_FILE = "start_result_navigation.usd"
WAREHOUSE_PATH = "/home/zyserver/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"
GRSCENES_ASSET_PATH = str(ASSET_ROOT / GRSCENES_USD_DIR / GRSCENES_USD_FILE)

OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
JOINT_LOG_PATH = OUTPUT_DIR / "multi_robot_joint_log.jsonl"


def _to_builtin(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def main():
    # scene_asset_path = EMPTY_SCENE_ASSET_PATH if USE_EMPTY_SCENE else GRSCENES_ASSET_PATH
    scene_asset_path = WAREHOUSE_PATH
    print(f"Using scene: {scene_asset_path}")

    if not Path(scene_asset_path).exists():
        raise FileNotFoundError(f"Scene file not found: {scene_asset_path}")

    robot_cfgs = [
        create_aliengo_robot_cfg(position=(-4.0, 0.0, 1.05)),
        create_h1_robot_cfg(position=(0.0, 0.0, 1.05), include_camera=False),
        create_g1_robot_cfg(position=(4.0, 0.0, 0.78)),
    ]

    robot_cfgs[0].name = "aliengo"
    robot_cfgs[0].prim_path = "/aliengo"
    robot_cfgs[1].name = "h1"
    robot_cfgs[1].prim_path = "/h1"
    robot_cfgs[2].name = "g1"
    robot_cfgs[2].prim_path = "/g1"

    api = MultiRobotNavigateAPI(
        scene_asset_path=scene_asset_path,
        robot_cfgs=robot_cfgs,
    )
    obs = api.start()
    print(f"Initial observations keys: {list(obs.keys())}")

    from omni.isaac.core.articulations import ArticulationView

    robot_views = {
        "aliengo": ArticulationView(prim_paths_expr="/World/env_0/robots/aliengo", name="aliengo_view"),
        "h1": ArticulationView(prim_paths_expr="/World/env_0/robots/h1", name="h1_view"),
        "g1": ArticulationView(prim_paths_expr="/World/env_0/robots/g1", name="g1_view"),
    }
    for robot_name, robot_view in robot_views.items():
        robot_view.initialize()
        print(f"{robot_name} joints: {robot_view.dof_names}")

    api.register_target("aliengo", "goal_a", (-2.5, 1.5, 0.0))
    api.register_target("h1", "goal_a", (1.5, 1.5, 0.0))
    api.register_target("g1", "goal_a", (5.5, 1.5, 0.0))

    api.register_target("aliengo", "goal_b", (-2.5, -1.5, 0.0))
    api.register_target("h1", "goal_b", (1.5, -1.5, 0.0))
    api.register_target("g1", "goal_b", (5.5, -1.5, 0.0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joint_log_file = open(JOINT_LOG_PATH, "w", encoding="utf-8")
    step_counter = 0

    def record_step(step, current_obs, current_targets):
        nonlocal step_counter
        step_counter += 1

        record = {
            "step": step_counter,
            "targets": {
                robot_name: target.name or target.position
                for robot_name, target in current_targets.items()
            },
            "robots": {},
        }

        for robot_name, robot_view in robot_views.items():
            joint_pos = robot_view.get_joint_positions()[0]
            joint_vel = robot_view.get_joint_velocities()[0]
            robot_obs = current_obs[robot_name]
            record["robots"][robot_name] = {
                "joint_names": list(robot_view.dof_names),
                "joint_positions": _to_builtin(joint_pos),
                "joint_velocities": _to_builtin(joint_vel),
                "base_position": _to_builtin(robot_obs["position"]),
                "base_orientation": _to_builtin(robot_obs["orientation"]),
            }

        joint_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step_counter % 200 == 0:
            print(
                f"step={step_counter} "
                f"aliengo={current_obs['aliengo']['position']} "
                f"h1={current_obs['h1']['position']} "
                f"g1={current_obs['g1']['position']}"
            )

    try:
        for stage_name, stage_targets in (
            (
                "stage_a",
                {
                    "aliengo": "goal_a",
                    "h1": "goal_a",
                    "g1": "goal_a",
                },
            ),
            (
                "stage_b",
                {
                    "aliengo": "goal_b",
                    "h1": "goal_b",
                    "g1": "goal_b",
                },
            ),
        ):
            result = api.navigate_all(
                stage_targets,
                dump_path=OUTPUT_DIR / f"{stage_name}_navigate.json",
                step_callback=record_step,
            )
            print(
                f"{stage_name} => success={result.success}, "
                f"steps={result.steps}, log={OUTPUT_DIR / f'{stage_name}_navigate.json'}"
            )
            if not result.success:
                print(f"{stage_name} did not finish normally, stop demo.")
                break
    finally:
        joint_log_file.close()
        api.close()

    print(f"Saved joint log to {JOINT_LOG_PATH}")


if __name__ == "__main__":
    main()
