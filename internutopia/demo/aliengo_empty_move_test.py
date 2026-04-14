import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "internutopia" / "assets"
ALIENGO_USD_PATH = ASSET_ROOT / "robots" / "aliengo" / "aliengo_camera.usd"
ALIENGO_PRIM_PATH = "/World/Robots/aliengo"
ALIENGO_MARKER_PATH = "/World/Robots/aliengo_marker"
ALIENGO_SCALE = [1.15, 1.15, 1.15]
ALIENGO_START_POSITION = [0.0, 0.0, 0.55]
ALIENGO_START_QUAT_XYZW = [0.0, 0.0, 0.0, 1.0]
WAYPOINTS = [
    [0.0, 0.0],
    [0.8, 0.0],
    [0.8, 0.8],
    [0.0, 0.8],
    [0.0, 0.0],
]
MOVE_STEP = 0.03
WAYPOINT_TOLERANCE = 0.05
HOLD_STEPS = 300


def _apply_transform(stage, prim_path: str, position, orientation_xyzw, scale):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()

    translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    scale_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    if scale_op is None:
        scale_op = xformable.AddScaleOp()

    translate_op.Set(Gf.Vec3d(*position))
    orient_op.Set(Gf.Quatd(orientation_xyzw[3], Gf.Vec3d(*orientation_xyzw[:3])))
    scale_op.Set(Gf.Vec3d(*scale))


def _spawn_aliengo(stage, add_reference_to_stage):
    from pxr import Gf, UsdGeom

    if not ALIENGO_USD_PATH.exists():
        raise FileNotFoundError(f"Aliengo USD not found: {ALIENGO_USD_PATH}")

    add_reference_to_stage(usd_path=str(ALIENGO_USD_PATH), prim_path=ALIENGO_PRIM_PATH)
    _apply_transform(
        stage=stage,
        prim_path=ALIENGO_PRIM_PATH,
        position=ALIENGO_START_POSITION,
        orientation_xyzw=ALIENGO_START_QUAT_XYZW,
        scale=ALIENGO_SCALE,
    )

    marker = UsdGeom.Sphere.Define(stage, ALIENGO_MARKER_PATH)
    marker.GetRadiusAttr().Set(0.12)
    marker.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.1, 0.1)])
    UsdGeom.XformCommonAPI(marker.GetPrim()).SetTranslate(
        (
            ALIENGO_START_POSITION[0],
            ALIENGO_START_POSITION[1],
            ALIENGO_START_POSITION[2] + 0.9,
        )
    )


def _set_aliengo_pose(stage, position_xy, yaw):
    quat_xyzw = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    _apply_transform(
        stage=stage,
        prim_path=ALIENGO_PRIM_PATH,
        position=(position_xy[0], position_xy[1], ALIENGO_START_POSITION[2]),
        orientation_xyzw=quat_xyzw,
        scale=ALIENGO_SCALE,
    )
    _apply_transform(
        stage=stage,
        prim_path=ALIENGO_MARKER_PATH,
        position=(position_xy[0], position_xy[1], ALIENGO_START_POSITION[2] + 0.9),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
    )


def _add_basic_lighting(stage):
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Looks/DomeLight")
    dome.CreateIntensityAttr(1200.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.96))

    distant = UsdLux.DistantLight.Define(stage, "/World/Looks/DistantLight")
    distant.CreateIntensityAttr(900.0)
    distant.CreateAngleAttr(1.0)
    distant.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Looks/DistantLight"))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))


def main():
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    try:
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
        except Exception:
            from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage

        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except Exception:
            from omni.isaac.core.utils.viewports import set_camera_view

        from pxr import UsdGeom

        create_new_stage()
        for _ in range(2):
            simulation_app.update()
        stage = get_current_stage()

        UsdGeom.Xform.Define(stage, "/World/Looks")
        UsdGeom.Xform.Define(stage, "/World/Robots")
        _add_basic_lighting(stage)
        _spawn_aliengo(stage, add_reference_to_stage)

        set_camera_view(
            eye=[3.0, 3.0, 2.2],
            target=[0.0, 0.0, 0.5],
            camera_prim_path="/OmniverseKit_Persp",
        )

        for _ in range(60):
            simulation_app.update()

        current_xy = np.array(WAYPOINTS[0], dtype=float)
        print(f"Aliengo spawned at {current_xy.tolist()}, starting empty-scene move test")

        for idx, waypoint in enumerate(WAYPOINTS[1:], start=1):
            target_xy = np.array(waypoint, dtype=float)
            print(f"Moving to waypoint_{idx}: {target_xy.tolist()}")
            while np.linalg.norm(target_xy - current_xy) > WAYPOINT_TOLERANCE and simulation_app.is_running():
                delta = target_xy - current_xy
                dist = np.linalg.norm(delta)
                direction = delta / max(dist, 1e-8)
                step_size = min(MOVE_STEP, dist)
                current_xy = current_xy + direction * step_size
                yaw = math.atan2(direction[1], direction[0])
                _set_aliengo_pose(stage, current_xy, yaw)
                simulation_app.update()

        print("Empty-scene waypoint test finished. Holding final frame...")
        for _ in range(HOLD_STEPS):
            if not simulation_app.is_running():
                break
            simulation_app.update()

        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
