from pathlib import Path
import time

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
from internutopia_extension.configs.controllers import JointControllerCfg
from internutopia_extension.configs.robots.pipergo2 import PiperGo2RobotCfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg


ARM_CONTROLLER_NAME = "arm_joint_controller"
ARM_JOINT_NAMES = [
    "piper_j1",
    "piper_j2",
    "piper_j3",
    "piper_j4",
    "piper_j5",
    "piper_j6",
    "piper_j7",
    "piper_j8",
]
SCENE_ASSET_PATH = "/home/zyserver/work/my_project/InternUtopia/internutopia/assets/scenes/empty.usd"


def make_env() -> Env:
    robot_cfg = PiperGo2RobotCfg(
        position=(0.0, 0.0, 0.55),
        arm_mass_scale=0.25,
        controllers=[
            JointControllerCfg(
                name=ARM_CONTROLLER_NAME,
                joint_names=ARM_JOINT_NAMES,
            ),
        ],
    )

    headless = not has_display()
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            webrtc=headless,
        ),
        metrics_save_path="none",
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=SCENE_ASSET_PATH,
                robots=[robot_cfg],
            )
        ],
    )
    import_extensions()
    return Env(config)


def hold_pose(env: Env, pose: np.ndarray, steps: int, label: str) -> dict:
    obs = {}
    for step in range(steps):
        obs, _, terminated, _, _ = env.step(action={ARM_CONTROLLER_NAME: [pose]})
        if step % 60 == 0:
            joint4 = float(obs["joint_positions"][15])
            eef = tuple(round(float(v), 4) for v in obs.get("eef_position", (0.0, 0.0, 0.0)))
            print(f"[{label}] step={step:03d} joint4={joint4:+.4f} eef={eef}")
        if terminated:
            raise RuntimeError(f"{label} terminated early")
    return obs


def get_current_arm_pose(env: Env) -> np.ndarray:
    obs = env.get_observations()
    joint_positions = np.array(obs["joint_positions"], dtype=float)
    # Whole robot order in current asset: 12 legs + 8 arm joints.
    return joint_positions[12:20].copy()


def main():
    if not Path(SCENE_ASSET_PATH).exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_ASSET_PATH}")

    env = make_env()
    try:
        obs, _ = env.reset()
        print(f"======== INIT OBS {obs} =========")
        print("This test does not use IK and does not move the quadruped base.")
        print("Watch piper_link4 only:")
        print("1. Neutral hold")
        print("2. piper_j4 = +0.6 rad")
        print("3. Back to neutral")
        print("4. piper_j4 = -0.6 rad")
        print("If the visually correct 'convex up' pose happens only when the sign is flipped, joint4 is likely reversed.")
        print("If both directions look reasonable and IK still chooses the wrong one, it is more likely an IK branch issue.")

        neutral_pose = get_current_arm_pose(env)
        neutral_pose[6] = 0.032
        neutral_pose[7] = -0.032

        pos_pose = neutral_pose.copy()
        pos_pose[3] = 0.6

        neg_pose = neutral_pose.copy()
        neg_pose[3] = -0.6

        hold_pose(env, neutral_pose, steps=180, label="neutral")
        hold_pose(env, pos_pose, steps=240, label="joint4_positive")
        hold_pose(env, neutral_pose, steps=180, label="back_to_neutral")
        hold_pose(env, neg_pose, steps=240, label="joint4_negative")

        print("Experiment finished. Simulation stays open for inspection. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    finally:
        pass


if __name__ == "__main__":
    main()
