import json
from pathlib import Path

import numpy as np

from internutopia.bridge import (
    create_aliengo_robot_cfg,
    create_franka_robot_cfg,
    create_g1_robot_cfg,
    create_h1_robot_cfg,
)
from internutopia.core.config import Config, SimConfig
from internutopia.core.util import has_display
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
USE_EMPTY_SCENE = False
EMPTY_SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")
GRSCENES_USD_DIR = "scenes/GRScenes-100/commercial_scenes/scenes/MV5M25QKTKJZ2AABAAAAAAA8_usd"
GRSCENES_USD_FILE = "start_result_navigation.usd"
WAREHOUSE_PATH = "/home/zyserver/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"
GRSCENES_ASSET_PATH = str(ASSET_ROOT / GRSCENES_USD_DIR / GRSCENES_USD_FILE)

OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
JOINT_LOG_PATH = OUTPUT_DIR / "four_robot_joint_log.jsonl"
MAX_STEPS = 5000
HOLD_STEPS_AFTER_DONE = 600


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
        create_franka_robot_cfg(position=(8.0, 0.0, 0.0)),
    ]

    robot_cfgs[0].name = "aliengo"
    robot_cfgs[0].prim_path = "/aliengo"
    robot_cfgs[1].name = "h1"
    robot_cfgs[1].prim_path = "/h1"
    robot_cfgs[2].name = "g1"
    robot_cfgs[2].prim_path = "/g1"
    robot_cfgs[3].name = "franka"
    robot_cfgs[3].prim_path = "/franka"

    headless = not has_display()
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            rendering_interval=0,
            headless=headless,
            native=headless,
            webrtc=headless,
        ),
        env_num=1,
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=scene_asset_path,
                robots=robot_cfgs,
            )
        ],
    )

    import_extensions()
    env = Env(config)

    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    env_obs, _ = env.reset()
    obs = env_obs[0]
    print(f"Initial observations keys: {list(obs.keys())}")

    robot_views = {
        "aliengo": ArticulationView(prim_paths_expr="/World/env_0/robots/aliengo", name="aliengo_view"),
        "h1": ArticulationView(prim_paths_expr="/World/env_0/robots/h1", name="h1_view"),
        "g1": ArticulationView(prim_paths_expr="/World/env_0/robots/g1", name="g1_view"),
        "franka": ArticulationView(prim_paths_expr="/World/env_0/robots/franka", name="franka_view"),
    }
    for robot_name, robot_view in robot_views.items():
        robot_view.initialize()
        print(f"{robot_name} joints: {robot_view.dof_names}")

    locomotion_targets = {
        "stage_a": {
            "aliengo": (-2.5, 1.5, 0.0),
            "h1": (1.5, 1.5, 0.0),
            "g1": (5.5, 1.5, 0.0),
        },
        "stage_b": {
            "aliengo": (-2.5, -1.5, 0.0),
            "h1": (1.5, -1.5, 0.0),
            "g1": (5.5, -1.5, 0.0),
        },
    }

    franka_pick_pos = np.array([8.40, 0.00, 0.10], dtype=np.float32)
    franka_place_pos = np.array([8.40, -0.30, 0.10], dtype=np.float32)
    franka_hover_offset = np.array([0.0, 0.0, 0.20], dtype=np.float32)
    franka_quat_down = euler_angles_to_quat((np.pi, 0.0, np.pi))

    def franka_arm_action(pos, quat=franka_quat_down):
        return {"arm_ik_controller": [np.array(pos, dtype=np.float32), quat]}

    def franka_gripper_action(cmd: str):
        return {"gripper_controller": [cmd]}

    franka_phase = 0
    franka_phase_step = 0
    franka_phase_duration = {
        0: 80,
        1: 220,
        2: 180,
        3: 100,
        4: 220,
        5: 260,
        6: 180,
        7: 100,
        8: 180,
    }

    stage_name = "stage_a"
    stable_count = {name: 0 for name in locomotion_targets[stage_name]}
    stage_done = False
    hold_steps_remaining = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joint_log_file = open(JOINT_LOG_PATH, "w", encoding="utf-8")

    try:
        for step in range(1, MAX_STEPS + 1):
            if hold_steps_remaining > 0:
                hold_steps_remaining -= 1
                env_action = [{"aliengo": {}, "h1": {}, "g1": {}, "franka": {}}]
                env_obs, _, _, _, _ = env.step(action=env_action)
                obs = env_obs[0]
                if step % 120 == 0 or hold_steps_remaining == 0:
                    print(f"holding final scene... remaining={hold_steps_remaining}")
                if hold_steps_remaining == 0:
                    print("Final hold completed, closing demo.")
                    break
                continue

            if franka_phase == 0:
                franka_action = franka_gripper_action("open")
            elif franka_phase == 1:
                franka_action = franka_arm_action(franka_pick_pos + franka_hover_offset)
            elif franka_phase == 2:
                franka_action = franka_arm_action(franka_pick_pos)
            elif franka_phase == 3:
                franka_action = franka_gripper_action("close")
            elif franka_phase == 4:
                franka_action = franka_arm_action(franka_pick_pos + franka_hover_offset)
            elif franka_phase == 5:
                franka_action = franka_arm_action(franka_place_pos + franka_hover_offset)
            elif franka_phase == 6:
                franka_action = franka_arm_action(franka_place_pos)
            elif franka_phase == 7:
                franka_action = franka_gripper_action("open")
            elif franka_phase == 8:
                franka_action = franka_arm_action(franka_place_pos + franka_hover_offset)
            else:
                franka_action = {}

            env_action = [
                {
                    "aliengo": {"move_to_point": [locomotion_targets[stage_name]["aliengo"]]},
                    "h1": {"move_to_point": [locomotion_targets[stage_name]["h1"]]},
                    "g1": {"move_to_point": [locomotion_targets[stage_name]["g1"]]},
                    "franka": franka_action,
                }
            ]

            env_obs, _, _, _, _ = env.step(action=env_action)
            obs = env_obs[0]
            if not obs:
                print("No observations returned, exiting.")
                break

            franka_phase_step += 1
            if franka_phase_step >= franka_phase_duration.get(franka_phase, 100):
                franka_phase += 1
                franka_phase_step = 0
            if franka_phase > 8:
                franka_phase = 0

            record = {
                "step": step,
                "stage": stage_name,
                "robots": {},
            }
            for robot_name, robot_view in robot_views.items():
                joint_pos = robot_view.get_joint_positions()[0]
                joint_vel = robot_view.get_joint_velocities()[0]
                robot_obs = obs[robot_name]
                record["robots"][robot_name] = {
                    "joint_names": list(robot_view.dof_names),
                    "joint_positions": _to_builtin(joint_pos),
                    "joint_velocities": _to_builtin(joint_vel),
                    "base_position": _to_builtin(robot_obs["position"]),
                    "base_orientation": _to_builtin(robot_obs["orientation"]),
                }
                if robot_name == "franka":
                    record["robots"][robot_name]["eef_position"] = _to_builtin(robot_obs["eef_position"])
                    record["robots"][robot_name]["eef_orientation"] = _to_builtin(robot_obs["eef_orientation"])
            joint_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            for robot_name in ("aliengo", "h1", "g1"):
                finished = obs[robot_name]["controllers"]["move_to_point"]["finished"]
                stable_count[robot_name] = stable_count[robot_name] + 1 if finished else 0

            if not stage_done and all(value >= 20 for value in stable_count.values()):
                stage_done = True
                print(f"{stage_name} finished at step {step}")
                with open(OUTPUT_DIR / f"{stage_name}_navigate.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "stage": stage_name,
                            "step": step,
                            "targets": locomotion_targets[stage_name],
                            "success": True,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                if stage_name == "stage_a":
                    stage_name = "stage_b"
                    stable_count = {name: 0 for name in locomotion_targets[stage_name]}
                    stage_done = False
                else:
                    hold_steps_remaining = HOLD_STEPS_AFTER_DONE
                    print(
                        "Both locomotion stages completed. "
                        f"Holding the final scene for {HOLD_STEPS_AFTER_DONE} steps."
                    )

            if step % 200 == 0:
                print(
                    f"step={step} "
                    f"aliengo={obs['aliengo']['position']} "
                    f"h1={obs['h1']['position']} "
                    f"g1={obs['g1']['position']} "
                    f"franka_eef={obs['franka']['eef_position']}"
                )
    finally:
        joint_log_file.close()
        env.close()

    print(f"Saved joint log to {JOINT_LOG_PATH}")


if __name__ == "__main__":
    main()
