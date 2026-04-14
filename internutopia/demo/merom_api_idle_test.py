import json
from pathlib import Path

from internutopia.bridge import MultiRobotNavigateAPI, create_aliengo_robot_cfg


OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
ASSEMBLED_SCENE_USD_PATH = OUTPUT_DIR / "merom_manual_scene.usd"
JOINT_LOG_PATH = OUTPUT_DIR / "aliengo_merom_idle_joint_log.jsonl"
ALIENGO_START_POSITION = (2.8, 1.3, 1.05)
HOLD_STEPS = 600
SUPPORT_FLOOR_PRIM_PATH = "/World/env_0/support_floor"
ENABLE_NAVIGATION_DEMO = True
FORWARD_DISTANCE = 1.0
NAV_LOG_PATH = OUTPUT_DIR / "aliengo_merom_forward_1m.json"
FINAL_HOLD_STEPS = 600


def _to_builtin(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _add_runtime_support_floor():
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    from omni.usd import get_context

    stage = get_context().get_stage()
    if stage.GetPrimAtPath(SUPPORT_FLOOR_PRIM_PATH).IsValid():
        return

    cube = UsdGeom.Cube.Define(stage, SUPPORT_FLOOR_PRIM_PATH)
    cube.GetSizeAttr().Set(1.0)
    prim = cube.GetPrim()
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetScale((40.0, 40.0, 0.1))
    xform.SetTranslate((0.0, 7.0, -0.05))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.24, 0.28)])

    UsdPhysics.CollisionAPI.Apply(prim)
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateRigidBodyEnabledAttr(False)

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateContactOffsetAttr(0.02)
    physx_collision.CreateRestOffsetAttr(0.0)


def main():
    if not ASSEMBLED_SCENE_USD_PATH.exists():
        raise FileNotFoundError(
            f"Assembled Merom scene not found: {ASSEMBLED_SCENE_USD_PATH}. "
            f"Run merom_scene_manual.py once to export it first."
        )

    print(f"Using scene: {ASSEMBLED_SCENE_USD_PATH}")

    robot_cfg = create_aliengo_robot_cfg(position=ALIENGO_START_POSITION)
    robot_cfg.name = "aliengo"
    robot_cfg.prim_path = "/aliengo"

    api = MultiRobotNavigateAPI(
        scene_asset_path=str(ASSEMBLED_SCENE_USD_PATH),
        robot_cfgs=[robot_cfg],
        headless=False,
    )
    obs = api.start()
    print(f"Initial observation keys: {list(obs.keys())}")
    _add_runtime_support_floor()

    from omni.isaac.core.articulations import ArticulationView

    robot_view = ArticulationView(
        prim_paths_expr="/World/env_0/robots/aliengo",
        name="aliengo_view",
    )
    robot_view.initialize()
    print(f"Aliengo joints: {robot_view.dof_names}")

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
        if ENABLE_NAVIGATION_DEMO:
            start_position = obs["aliengo"]["position"]
            target_x = float(start_position[0])
            target_y = float(start_position[1] + FORWARD_DISTANCE)
            api.register_target("aliengo", "forward_1m", (target_x, target_y, 0.0))
            print(f"Navigate target forward_1m => ({target_x:.3f}, {target_y:.3f}, 0.0)")
            result = api.navigate_all(
                {"aliengo": "forward_1m"},
                dump_path=NAV_LOG_PATH,
                step_callback=record_step,
            )
            print(
                f"navigate_all(forward_1m) => success={result.success}, "
                f"steps={result.steps}, log={NAV_LOG_PATH}"
            )
            print("Holding final scene...")
            for hold_step in range(FINAL_HOLD_STEPS):
                env_obs, _, _, _, _ = api.env.step([{}])
                current_obs = env_obs[0]
                record_step(step_counter + 1, current_obs, {"aliengo": type('Target', (), {'name': 'hold'})()})
        else:
            for _ in range(HOLD_STEPS):
                env_obs, _, _, _, _ = api.env.step([{}])
                current_obs = env_obs[0]
                record_step(step_counter + 1, current_obs, {"aliengo": type('Target', (), {'name': 'idle'})()})
    finally:
        joint_log_file.close()
        api.close()

    print(f"Saved idle joint log to {JOINT_LOG_PATH}")


if __name__ == "__main__":
    main()
