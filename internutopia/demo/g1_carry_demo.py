from pathlib import Path

import numpy as np

from internutopia.bridge import create_g1_robot_cfg
from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
from internutopia_extension.configs.controllers import JointControllerCfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
WAREHOUSE_PATH = "/home/zyserver/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"
EMPTY_SCENE_ASSET_PATH = str(ASSET_ROOT / "scenes" / "empty.usd")
USE_WAREHOUSE = True
SCENE_ASSET_PATH = WAREHOUSE_PATH if USE_WAREHOUSE else EMPTY_SCENE_ASSET_PATH

ARM_CONTROLLER_NAME = "arm_joint_controller"
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def make_env():
    robot_cfg = create_g1_robot_cfg(position=(0.0, 0.0, 0.78))
    robot_cfg.controllers.append(
        JointControllerCfg(
            name=ARM_CONTROLLER_NAME,
            joint_names=ARM_JOINT_NAMES,
        )
    )

    headless = not has_display()
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            rendering_interval=0,
            headless=headless,
            webrtc=headless,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=SCENE_ASSET_PATH,
                robots=[robot_cfg],
            )
        ],
    )
    import_extensions()
    return Env(config)


def step_until_reached(env: Env, target, arm_pose=None, max_steps=1500):
    stable = 0
    for step in range(max_steps):
        action = {"move_to_point": [target]}
        if arm_pose is not None:
            action[ARM_CONTROLLER_NAME] = [arm_pose]

        obs, _, _, _, _ = env.step(action=action)
        finished = obs["controllers"]["move_to_point"]["finished"]
        stable = stable + 1 if finished else 0

        if step % 100 == 0:
            print(f"navigate step={step}, pos={obs['position']}, target={target}")

        if stable >= 20:
            return obs
    raise RuntimeError(f"Failed to reach target {target} within {max_steps} steps")


def hold_pose(env: Env, arm_pose, hold_steps=120):
    for step in range(hold_steps):
        obs, _, _, _, _ = env.step(action={ARM_CONTROLLER_NAME: [arm_pose]})
        if step % 60 == 0:
            print(f"hold step={step}, pos={obs['position']}")


def get_arm_pose(view, target_map):
    indices = {name: idx for idx, name in enumerate(view.dof_names)}
    current = view.get_joint_positions()[0]
    pose = np.array([current[indices[name]] for name in ARM_JOINT_NAMES], dtype=float)
    for joint_name, value in target_map.items():
        pose[ARM_JOINT_NAMES.index(joint_name)] = value
    return pose


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    env = make_env()
    obs, _ = env.reset()
    print(f"Initial obs keys: {list(obs.keys())}")

    from omni.isaac.core.articulations import ArticulationView

    robot = ArticulationView(prim_paths_expr="/World/env_0/robots/g1", name="g1_view")
    robot.initialize()

    rest_pose = get_arm_pose(robot, {})
    carry_pose = get_arm_pose(
        robot,
        {
            "left_shoulder_pitch_joint": 0.85,
            "right_shoulder_pitch_joint": 0.85,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.25,
            "right_elbow_joint": 1.25,
            "left_wrist_pitch_joint": -0.2,
            "right_wrist_pitch_joint": -0.2,
        },
    )

    pickup_target = (1.5, 0.0, 0.0)
    drop_target = (3.0, -1.0, 0.0)

    print(f"Step 1: walk to pickup target {pickup_target}")
    step_until_reached(env, pickup_target)

    print("Step 2: raise both arms into a carry pose")
    hold_pose(env, carry_pose, hold_steps=180)

    print(f"Step 3: keep carry pose and walk to drop target {drop_target}")
    step_until_reached(env, drop_target, arm_pose=carry_pose)

    print("Step 4: lower arms to place the carried object")
    hold_pose(env, rest_pose, hold_steps=180)

    env.close()


if __name__ == "__main__":
    main()
