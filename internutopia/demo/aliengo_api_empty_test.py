import json
from pathlib import Path

from internutopia.bridge import MultiRobotNavigateAPI, create_aliengo_robot_cfg


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
EMPTY_SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
JOINT_LOG_PATH = OUTPUT_DIR / "aliengo_api_empty_joint_log.jsonl"


def _to_builtin(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def main():
    scene_asset_path = EMPTY_SCENE_ASSET_PATH
    print(f"Using scene: {scene_asset_path}")

    if not Path(scene_asset_path).exists():
        raise FileNotFoundError(f"Scene file not found: {scene_asset_path}")

    robot_cfg = create_aliengo_robot_cfg(position=(0.0, 0.0, 1.05))
    robot_cfg.name = "aliengo"
    robot_cfg.prim_path = "/aliengo"

    api = MultiRobotNavigateAPI(
        scene_asset_path=scene_asset_path,
        robot_cfgs=[robot_cfg],
        headless=False,
    )
    obs = api.start()
    print(f"Initial observation keys: {list(obs.keys())}")

    from omni.isaac.core.articulations import ArticulationView

    robot_view = ArticulationView(
        prim_paths_expr="/World/env_0/robots/aliengo",
        name="aliengo_view",
    )
    robot_view.initialize()
    print(f"Aliengo joints: {robot_view.dof_names}")

    targets = {
        "goal_a": (1.0, 0.0, 0.0),
        "goal_b": (1.0, 1.0, 0.0),
        "goal_c": (0.0, 1.0, 0.0),
    }
    for target_name, target_position in targets.items():
        api.register_target("aliengo", target_name, target_position)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joint_log_file = open(JOINT_LOG_PATH, "w", encoding="utf-8")
    step_counter = 0

    def record_step(step, current_obs, current_targets):
        nonlocal step_counter
        step_counter += 1

        robot_obs = current_obs["aliengo"]
        record = {
            "step": step_counter,
            "target": current_targets["aliengo"].name or current_targets["aliengo"].position,
            "joint_names": list(robot_view.dof_names),
            "joint_positions": _to_builtin(robot_view.get_joint_positions()[0]),
            "joint_velocities": _to_builtin(robot_view.get_joint_velocities()[0]),
            "base_position": _to_builtin(robot_obs["position"]),
            "base_orientation": _to_builtin(robot_obs["orientation"]),
        }
        joint_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step_counter % 100 == 0:
            print(f"step={step_counter} aliengo={robot_obs['position']}")

    try:
        for target_name in ("goal_a", "goal_b", "goal_c"):
            log_path = OUTPUT_DIR / f"aliengo_api_empty_{target_name}.json"
            result = api.navigate_all(
                {"aliengo": target_name},
                dump_path=log_path,
                step_callback=record_step,
            )
            print(
                f"navigate_all({target_name}) => success={result.success}, "
                f"steps={result.steps}, log={log_path}"
            )
            if not result.success:
                print(f"Navigation to {target_name} did not finish normally, stop demo.")
                break
    finally:
        joint_log_file.close()
        api.close()

    print(f"Saved joint log to {JOINT_LOG_PATH}")


if __name__ == "__main__":
    main()
