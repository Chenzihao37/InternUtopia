from pathlib import Path

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.core.util.joint import create_joint
from internutopia_extension import import_extensions
from internutopia_extension.configs.controllers import JointControllerCfg
from internutopia_extension.configs.objects import DynamicCubeCfg
from internutopia_extension.configs.robots.pipergo2 import PiperGo2RobotCfg, move_to_point_cfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

ARM_CONTROLLER_NAME = 'arm_joint_controller'
ARM_JOINT_NAMES = [
    'piper_j1',
    'piper_j2',
    'piper_j3',
    'piper_j4',
    'piper_j5',
    'piper_j6',
    'piper_j7',
    'piper_j8',
]

PEDESTAL_CFG = DynamicCubeCfg(
    name='grasp_pedestal',
    prim_path='/World/grasp_pedestal',
    position=(1.18, 0.0, 0.325),
    scale=(0.18, 0.18, 0.65),
    color=(0.5, 0.5, 0.5),
)

GRASP_CUBE_CFG = DynamicCubeCfg(
    name='grasp_cube',
    prim_path='/World/grasp_cube',
    position=(1.18, 0.0, 0.675),
    scale=(0.05, 0.05, 0.05),
    color=(0.85, 0.2, 0.2),
)


def make_env():
    robot_cfg = PiperGo2RobotCfg(
        position=(0.0, 0.0, 0.55),
        controllers=[
            move_to_point_cfg,
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
            rendering_interval=0,
            headless=headless,
            webrtc=headless,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path='/home/zyserver/work/my_project/InternUtopia/internutopia/assets/scenes/empty.usd',
                robots=[robot_cfg],
                objects=[PEDESTAL_CFG, GRASP_CUBE_CFG],
            )
        ],
    )
    import_extensions()
    return Env(config)


def step_until_reached(env: Env, target, arm_pose=None, max_steps=1800):
    stable = 0
    for step in range(max_steps):
        action = {move_to_point_cfg.name: [target]}
        if arm_pose is not None:
            action[ARM_CONTROLLER_NAME] = [arm_pose]
        obs, _, _, _, _ = env.step(action=action)
        finished = obs['controllers'][move_to_point_cfg.name]['finished']
        stable = stable + 1 if finished else 0
        if step % 120 == 0:
            print(f'navigate step={step}, pos={obs["position"]}, target={target}')
        if stable >= 20:
            return obs
    raise RuntimeError(f'Failed to reach target {target} within {max_steps} steps')


def hold_pose(env: Env, arm_pose, hold_steps=120, label='hold'):
    obs = None
    for step in range(hold_steps):
        obs, _, _, _, _ = env.step(action={ARM_CONTROLLER_NAME: [arm_pose]})
        if step % 60 == 0:
            print(f'{label} step={step}, pos={obs["position"]}')
    return obs


def get_arm_pose(view, target_map):
    indices = {name: idx for idx, name in enumerate(view.dof_names)}
    current = view.get_joint_positions()[0]
    pose = np.array([current[indices[name]] for name in ARM_JOINT_NAMES], dtype=float)
    for joint_name, value in target_map.items():
        pose[ARM_JOINT_NAMES.index(joint_name)] = value
    return pose


def resolve_prim_path(rigid_body):
    raw = rigid_body.unwrap()
    for attr in ('prim_path', '_prim_path'):
        value = getattr(raw, attr, None)
        if value is not None:
            return value
    return rigid_body._param['prim_path']


def main():
    scene_path = Path('/home/zyserver/work/my_project/InternUtopia/internutopia/assets/scenes/empty.usd')
    if not scene_path.exists():
        raise FileNotFoundError(f'Scene file not found: {scene_path}')

    env = make_env()
    obs, _ = env.reset()
    print(f'Initial obs keys: {list(obs.keys())}')

    from omni.isaac.core.articulations import ArticulationView

    robot = ArticulationView(prim_paths_expr='/World/env_0/robots/pipergo2', name='pipergo2_view')
    robot.initialize()

    grasp_cube = env.runner.get_obj(GRASP_CUBE_CFG.name)
    grasp_cube.set_mass(0.05)
    pedestal = env.runner.get_obj(PEDESTAL_CFG.name)
    pedestal.set_mass(100.0)

    rest_pose = get_arm_pose(robot, {})
    open_reach_pose = get_arm_pose(
        robot,
        {
            'piper_j1': 0.0,
            'piper_j2': 1.20,
            'piper_j3': -2.20,
            'piper_j4': 0.0,
            'piper_j5': 1.05,
            'piper_j6': 0.0,
            'piper_j7': 0.032,
            'piper_j8': -0.032,
        },
    )
    close_grasp_pose = get_arm_pose(
        robot,
        {
            'piper_j1': 0.0,
            'piper_j2': 1.20,
            'piper_j3': -2.20,
            'piper_j4': 0.0,
            'piper_j5': 1.05,
            'piper_j6': 0.0,
            'piper_j7': 0.010,
            'piper_j8': -0.010,
        },
    )
    retract_pose = get_arm_pose(
        robot,
        {
            'piper_j1': 0.0,
            'piper_j2': 0.70,
            'piper_j3': -1.50,
            'piper_j4': 0.0,
            'piper_j5': 0.80,
            'piper_j6': 0.0,
            'piper_j7': 0.010,
            'piper_j8': -0.010,
        },
    )

    walk_target = (0.78, 0.0, 0.0)

    print(f'Step 1: walk to grasp approach point {walk_target}')
    step_until_reached(env, walk_target, arm_pose=rest_pose)

    print('Step 2: keep stable before manipulation')
    hold_pose(env, rest_pose, hold_steps=120, label='stabilize')

    print('Step 3: extend arm with open gripper')
    hold_pose(env, open_reach_pose, hold_steps=180, label='extend')

    print('Step 4: close gripper to grasp the cube')
    hold_pose(env, close_grasp_pose, hold_steps=120, label='close_gripper')

    print('Step 5: attach cube to gripper base for a stable carry demo')
    create_joint(
        prim_path='/World/pipergo2_grasp_joint',
        joint_type='FixedJoint',
        body0='/World/env_0/robots/pipergo2/piper_gripper_base',
        body1=resolve_prim_path(grasp_cube),
        enabled=True,
    )

    print('Step 6: retract arm with the grasped cube')
    hold_pose(env, retract_pose, hold_steps=180, label='retract')

    print('Step 7: return arm to rest pose')
    hold_pose(env, rest_pose, hold_steps=180, label='rest')

    env.close()


if __name__ == '__main__':
    main()
