import cv2
import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.macros import gm
from internutopia.core.util import has_display

from internutopia_extension import import_extensions

from internutopia_extension.configs.robots.h1 import (
    H1RobotCfg,
    h1_camera_cfg,
    move_along_path_cfg,
)

from internutopia_extension.configs.tasks import FiniteStepTaskCfg


print("===== Starting InternUtopia Camera Demo =====")


# -----------------------------
# Robot config
# -----------------------------

h1_robot = H1RobotCfg(
    position=(0, 0, 1.05),
    controllers=[move_along_path_cfg],
    sensors=[
        h1_camera_cfg.update(
            name="camera",
            resolution=(320, 240)
        )
    ]
)


# -----------------------------
# Simulation config
# -----------------------------

headless = False
if not has_display():
    headless = True


config = Config(
    simulator=SimConfig(
        physics_dt=1/240,
        rendering_dt=1/240,
        headless=headless,
        webrtc=False
    ),
    task_configs=[
        FiniteStepTaskCfg(
            max_steps=1000,
            scene_asset_path=gm.ASSET_PATH + "/scenes/empty.usd",
            scene_scale=(0.01,0.01,0.01),
            robots=[h1_robot]
        )
    ]
)


# -----------------------------
# Start simulator
# -----------------------------

import_extensions()

env = Env(config)

obs,_ = env.reset()

print("Sensors:", obs["sensors"])


# -----------------------------
# Video recorder
# -----------------------------

video = cv2.VideoWriter(
    "h1_camera.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (320,240)
)


# -----------------------------
# Robot path
# -----------------------------

path = [
    (1.0,0.0,0.0),
    (1.0,1.0,0.0),
    (3.0,4.0,0.0)
]

move_action = {
    move_along_path_cfg.name:[path]
}


# -----------------------------
# Simulation loop
# -----------------------------

step = 0

while env.simulation_app.is_running():

    step += 1

    obs,_,terminated,_,_ = env.step(action=move_action)

    # episode结束
    if obs is None:
        break

    try:

        camera = obs["sensors"]["camera"]

        # camera可能没准备好
        if "rgba" not in camera:
            continue

        frame = camera["rgba"]

        # 前几帧为空
        if frame.size == 0:
            continue

        # RGBA → RGB
        frame = frame[:,:,:3]
        frame = frame.astype(np.uint8)

        video.write(frame)

    except Exception as e:

        print("Camera read error:", e)

    if step % 100 == 0:
        print("step:", step)

    if terminated:

        obs,info = env.reset()

        if info is None:
            break


# -----------------------------
# Close
# -----------------------------

video.release()

env.close()

print("Video saved: h1_camera.mp4")
