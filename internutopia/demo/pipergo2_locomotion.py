from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.pipergo2 import (
    PiperGo2RobotCfg,
    move_along_path_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

headless = not has_display()

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless, webrtc=headless),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path='/home/zyserver/work/my_project/InternUtopia/internutopia/assets/scenes/empty.usd',
            robots=[
                PiperGo2RobotCfg(
                    position=(0.0, 0.0, 0.55),
                    controllers=[move_along_path_cfg],
                )
            ],
        ),
    ],
)

import_extensions()

env = Env(config)
obs, _ = env.reset()


def get_robot_obs(obs_data):
    if isinstance(obs_data, dict) and 'position' in obs_data:
        return obs_data
    if isinstance(obs_data, dict) and 'pipergo2' in obs_data:
        return obs_data['pipergo2']
    if isinstance(obs_data, dict) and 'pipergo2_0' in obs_data:
        return obs_data['pipergo2_0']
    if isinstance(obs_data, (list, tuple)) and len(obs_data) > 0:
        first = obs_data[0]
        if isinstance(first, dict) and 'pipergo2' in first:
            return first['pipergo2']
        if isinstance(first, dict) and 'pipergo2_0' in first:
            return first['pipergo2_0']
        return first
    raise KeyError(f'Unsupported observation structure: {type(obs_data)}')

env_action = {
    move_along_path_cfg.name: [
        [
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ]
    ],
}

print(f'actions: {env_action}')

step_count = 0
while env.simulation_app.is_running():
    step_count += 1
    obs, _, terminated, _, _ = env.step(action=env_action)

    if step_count % 240 == 0:
        robot_obs = get_robot_obs(obs)
        position = robot_obs['position']
        controller_obs = robot_obs['controllers'][move_along_path_cfg.name]
        print(
            f'step={step_count} '
            f'position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) '
            f'path_index={controller_obs.get("current_index")} '
            f'finished={controller_obs.get("finished")}'
        )

    episode_terminated = terminated[0] if isinstance(terminated, (list, tuple)) else bool(terminated)
    if episode_terminated:
        print('episode terminated')
        break

env.close()
