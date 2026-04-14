import json
import heapq
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "internutopia" / "assets"
OG_ROOT = PROJECT_ROOT.parent / "COHERENT" / "OmniGibson"
OG_DATA_ROOT = OG_ROOT / "omnigibson" / "data"
OG_DATASET_ROOT = OG_DATA_ROOT / "og_dataset"
SCENE_JSON_PATH = OG_DATASET_ROOT / "scenes" / "Merom_1_int" / "json" / "Merom_1_int_best.json"
KEY_PATH = OG_DATA_ROOT / "omnigibson.key"
DECRYPT_CACHE_DIR = Path(__file__).resolve().parent / "logs" / "merom_manual_cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
MAX_OBJECTS = None
ALIENGO_USD_PATH = ASSET_ROOT / "robots" / "aliengo" / "aliengo_camera.usd"
ALIENGO_PRIM_PATH = "/World/Robots/aliengo"
ALIENGO_POSITION = [2.8, 1.8, 0.55]
ALIENGO_ORIENTATION_XYZW = [0.0, 0.0, 0.70710678, 0.70710678]
ALIENGO_SCALE = [1.15, 1.15, 1.15]
SKIP_CATEGORIES = {
    "ceilings",
}
TRAV_MAP_PATH = OG_DATASET_ROOT / "scenes" / "Merom_1_int" / "layout" / "floor_trav_0.png"
TRAV_MAP_RESOLUTION = 0.1
TRAV_MAP_DEFAULT_RESOLUTION = 0.01
TRAV_MAP_EROSION = 4
ALIENGO_MOVE_STEP = 0.05
WAYPOINT_TOLERANCE = 0.08
POST_NAV_HOLD_STEPS = 180
FINAL_HOLD_STEPS = 600
DEMO_WAYPOINTS = [
    "start_anchor",
    "point_1",
    "point_3",
]
WAYPOINT_DISPLAY_Z = 0.08
ENABLE_MOVEMENT_DEMO = True
REFERENCE_POINTS = {
    "p1": [2.4, 2.6, WAYPOINT_DISPLAY_Z],
    "p2": [2.0, 4.8, WAYPOINT_DISPLAY_Z],
    "p3": [1.6, 6.6, WAYPOINT_DISPLAY_Z],
    "goal": [1.2, 7.4, WAYPOINT_DISPLAY_Z],
    "goal_x_plus_0_5": [1.7, 7.4, WAYPOINT_DISPLAY_Z],
    "goal_y_plus_0_5": [1.2, 7.9, WAYPOINT_DISPLAY_Z],
}
DEBUG_CONTROL_POINTS = [
    {"name": "p1", "position": REFERENCE_POINTS["p1"], "color": [1.0, 0.8, 0.2]},
    {"name": "p2", "position": REFERENCE_POINTS["p2"], "color": [1.0, 0.6, 0.2]},
    {"name": "p3", "position": REFERENCE_POINTS["p3"], "color": [1.0, 0.4, 0.2]},
    {"name": "goal", "position": REFERENCE_POINTS["goal"], "color": [1.0, 0.0, 0.0]},
    {"name": "goal_x_plus_0_5", "position": REFERENCE_POINTS["goal_x_plus_0_5"], "color": [1.0, 1.0, 0.0]},
    {"name": "goal_y_plus_0_5", "position": REFERENCE_POINTS["goal_y_plus_0_5"], "color": [0.0, 0.0, 1.0]},
]
WAYPOINT_ANCHORS = [
    {
        "name": "start_anchor",
        "mode": "offset_from_prim",
        "asset_prim_path": "/World/MeromScene/carpet_ctclvd_1/base_link/visuals",
        "offset": [0.0, -0.5, 0.08],
        "color": [0.0, 1.0, 0.0],
        "neighbors": ["point_1", "point_2"],
    },
    {
        "name": "point_1",
        "mode": "offset_from_prim",
        "asset_prim_path": "/World/MeromScene/door_lvgliq_1/link_2/visuals",
        "offset": [0.0, 0.3, 0.0],
        "color": [1.0, 0.85, 0.2],
        "neighbors": ["start_anchor", "point_3"],
    },
    {
        "name": "point_2",
        "mode": "compose_axes",
        "x": {"source": "value", "value": REFERENCE_POINTS["p3"][0], "offset": 0.1},
        "y": {"source": "anchor", "name": "start_anchor"},
        "z": {"source": "anchor", "name": "start_anchor"},
        "color": [0.2, 0.9, 1.0],
        "neighbors": ["start_anchor", "point_4"],
    },
    {
        "name": "point_3",
        "mode": "compose_axes",
        "x": {"source": "prim", "path": "/World/MeromScene/armchair_qplklw_2/base_link/visuals", "offset": -0.1},
        "y": {"source": "anchor", "name": "point_1"},
        "z": {"source": "anchor", "name": "point_1"},
        "color": [1.0, 0.45, 0.15],
        "neighbors": ["point_1"],
    },
    {
        "name": "point_4",
        "mode": "compose_axes",
        "x": {"source": "anchor", "name": "point_2"},
        "y": {"source": "value", "value": REFERENCE_POINTS["goal_x_plus_0_5"][1]},
        "z": {"source": "anchor", "name": "point_2"},
        "color": [0.65, 0.4, 1.0],
        "neighbors": ["point_2", "point_5"],
    },
    {
        "name": "point_5",
        "mode": "compose_axes",
        "x": {"source": "anchor", "name": "point_4"},
        "y": {"source": "prim", "path": "/World/MeromScene/door_lvgliq_2/link_2/visuals"},
        "z": {"source": "anchor", "name": "point_4"},
        "color": [1.0, 0.2, 0.7],
        "neighbors": ["point_4", "point_6"],
    },
    {
        "name": "point_6",
        "mode": "offset_from_prim",
        "asset_prim_path": "/World/MeromScene/bottom_cabinet_jrhgeu_1/link_1/visuals",
        "offset": [0.2, 0.0, 0.0],
        "color": [0.3, 1.0, 0.55],
        "neighbors": ["point_5"],
    },
]

# 室内补光参数。位置是 [x, y, z]，建议把 z 放在天花板下方一点。
INTERIOR_LIGHTS = [
    # 入门区域
    {"name": "EntryLight_A", "position": [0.0, 1.0, 2.35]},
    {"name": "EntryLight_B", "position": [0.0, 2.5, 2.35]},

    # 客厅（多灯覆盖）
    {"name": "LivingLight_A", "position": [0.6, 4.0, 2.35]},
    {"name": "LivingLight_B", "position": [1.5, 5.5, 2.35]},
    {"name": "LivingLight_C", "position": [-0.5, 5.5, 2.35]},

    # 厨房（重点提亮）
    {"name": "KitchenLight_A", "position": [1.8, 7.8, 2.30]},
    {"name": "KitchenLight_B", "position": [2.5, 9.0, 2.30]},

    # 侧房
    {"name": "SideRoomLight_A", "position": [-1.4, 7.0, 2.30]},
    {"name": "SideRoomLight_B", "position": [-2.2, 8.5, 2.30]},
]


@dataclass
class NavigateResult:
    success: bool
    steps: int
    path_world: list[list[float]]
    final_position: list[float]
    target_position: list[float]

    def dump_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "success": self.success,
                    "steps": self.steps,
                    "path_world": self.path_world,
                    "final_position": self.final_position,
                    "target_position": self.target_position,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _get_object_asset_path(category: str, model: str) -> Path:
    usd_dir = OG_DATASET_ROOT / "objects" / category / model / "usd"
    plain_usd = usd_dir / f"{model}.usd"
    encrypted_usd = usd_dir / f"{model}.encrypted.usd"
    if plain_usd.exists():
        return plain_usd
    if encrypted_usd.exists():
        return encrypted_usd
    raise FileNotFoundError(f"Asset not found for {category}/{model}: {plain_usd} or {encrypted_usd}")


def _decrypt_usd_if_needed(asset_path: Path) -> Path:
    if asset_path.suffix == ".usd" and not asset_path.name.endswith(".encrypted.usd"):
        return asset_path

    # Preserve the whole usd directory layout so relative texture references like
    # "materials/foo-albedo.png" continue to resolve correctly.
    relative_dir = asset_path.relative_to(OG_DATASET_ROOT / "objects").parent
    out_dir = DECRYPT_CACHE_DIR / relative_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / asset_path.name.replace(".encrypted.usd", ".usd")

    for sibling in asset_path.parent.iterdir():
        target = out_dir / sibling.name
        if sibling.is_dir():
            shutil.copytree(sibling, target, dirs_exist_ok=True)
        elif sibling.name != asset_path.name and not target.exists():
            shutil.copy2(sibling, target)

    if out_path.exists() and out_path.stat().st_mtime >= asset_path.stat().st_mtime:
        return out_path

    with open(KEY_PATH, "rb") as key_file:
        key = key_file.read()
    with open(asset_path, "rb") as encrypted_file:
        encrypted = encrypted_file.read()

    decrypted = Fernet(key).decrypt(encrypted)
    with open(out_path, "wb") as usd_file:
        usd_file.write(decrypted)
    return out_path


def _freeze_physics_subtree(stage, root_prim_path: str):
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return

    for prim in Usd.PrimRange(root):
        rigid_enabled_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if rigid_enabled_attr.IsValid():
            rigid_enabled_attr.Set(False)

        kinematic_attr = prim.GetAttribute("physics:kinematicEnabled")
        if kinematic_attr.IsValid():
            kinematic_attr.Set(True)

        # Also strip velocity-like settings if present so imported props stay put.
        for attr_name, value in (
            ("physics:velocity", (0.0, 0.0, 0.0)),
            ("physics:angularVelocity", (0.0, 0.0, 0.0)),
        ):
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(value)


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
        position=ALIENGO_POSITION,
        orientation_xyzw=ALIENGO_ORIENTATION_XYZW,
        scale=ALIENGO_SCALE,
    )

    # Add a visible marker above the robot so it is easy to spot during scene debugging.
    marker = UsdGeom.Sphere.Define(stage, "/World/Robots/aliengo_marker")
    marker.GetRadiusAttr().Set(0.12)
    marker_prim = marker.GetPrim()
    UsdGeom.XformCommonAPI(marker_prim).SetTranslate(
        (
            ALIENGO_POSITION[0],
            ALIENGO_POSITION[1],
            ALIENGO_POSITION[2] + 0.9,
        )
    )
    marker.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])


def _world_to_map(xy, map_size, map_resolution):
    return np.flip((np.array(xy) / map_resolution + map_size / 2.0)).astype(int)


def _map_to_world(xy, map_size, map_resolution):
    axis = 0 if len(np.array(xy).shape) == 1 else 1
    return np.flip((np.array(xy) - map_size / 2.0) * map_resolution, axis=axis)


def _load_trav_map():
    trav_map = np.array(Image.open(TRAV_MAP_PATH))
    map_size = int(trav_map.shape[0] * TRAV_MAP_DEFAULT_RESOLUTION / TRAV_MAP_RESOLUTION)
    trav_map = np.array(Image.fromarray(trav_map).resize((map_size, map_size), Image.NEAREST))
    try:
        import cv2

        if TRAV_MAP_EROSION > 0:
            trav_map = cv2.erode(trav_map, np.ones((TRAV_MAP_EROSION, TRAV_MAP_EROSION), np.uint8))
    except Exception:
        pass
    trav_map[trav_map < 255] = 0
    trav_map[trav_map >= 255] = 255
    return trav_map, map_size


def _nearest_free_cell(binary_map, cell):
    h, w = binary_map.shape
    r0, c0 = int(cell[0]), int(cell[1])
    if 0 <= r0 < h and 0 <= c0 < w and binary_map[r0, c0] > 0:
        return (r0, c0)

    visited = set()
    queue = [(r0, c0)]
    while queue:
        r, c = queue.pop(0)
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if 0 <= r < h and 0 <= c < w and binary_map[r, c] > 0:
            return (r, c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and -1 < nr < h + 1 and -1 < nc < w + 1:
                    queue.append((nr, nc))
    raise RuntimeError(f"No traversable cell found near {cell}")


def _astar(binary_map, start, goal):
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < binary_map.shape[0] and 0 <= nc < binary_map.shape[1]):
                continue
            if binary_map[nr, nc] == 0:
                continue

            nxt = (nr, nc)
            step_cost = math.hypot(dr, dc)
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f_score = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, nxt))

    raise RuntimeError(f"No path found from {start} to {goal}")


class ManualAliengoNavigateAPI:
    def __init__(self):
        self._simulation_app = None
        self._stage = None
        self._set_camera_view = None
        self._add_reference_to_stage = None
        self._trav_map = None
        self._map_size = None
        self._resolved_waypoint_anchor_map = {}

    def _render_steps(self, count=1):
        for _ in range(count):
            self._simulation_app.update()

    def start(self):
        if not SCENE_JSON_PATH.exists():
            raise FileNotFoundError(f"Scene json not found: {SCENE_JSON_PATH}")
        if not KEY_PATH.exists():
            raise FileNotFoundError(f"OmniGibson key not found: {KEY_PATH}")

        from omni.isaac.kit import SimulationApp

        self._simulation_app = SimulationApp({"headless": False})

        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except Exception:
            try:
                from omni.isaac.core.utils.viewports import set_camera_view
            except Exception:
                set_camera_view = None

        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
        except Exception:
            from omni.isaac.core.utils.stage import add_reference_to_stage

        try:
            from isaacsim.core.utils.stage import create_new_stage, get_current_stage
        except Exception:
            from omni.isaac.core.utils.stage import create_new_stage, get_current_stage

        from pxr import UsdGeom

        self._set_camera_view = set_camera_view
        self._add_reference_to_stage = add_reference_to_stage
        create_new_stage()
        self._render_steps(2)
        self._stage = get_current_stage()

        UsdGeom.Xform.Define(self._stage, "/World/MeromScene")
        UsdGeom.Xform.Define(self._stage, "/World/Looks")
        UsdGeom.Xform.Define(self._stage, "/World/Robots")
        UsdGeom.Xform.Define(self._stage, "/World/DebugPoints")
        _add_basic_lighting(self._stage)

        if self._set_camera_view is not None:
            self._set_camera_view(
                eye=[3.0, 9.0, 6.0],
                target=[0.0, 7.0, 0.8],
                camera_prim_path="/OmniverseKit_Persp",
            )

        entries = _load_scene_entries()
        for entry in entries:
            prim_path = f"/World/MeromScene/{entry['name']}"
            asset_path = _get_object_asset_path(entry["category"], entry["model"])
            usd_path = _decrypt_usd_if_needed(asset_path)
            self._add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)
            _apply_transform(
                stage=self._stage,
                prim_path=prim_path,
                position=entry["position"],
                orientation_xyzw=entry["orientation"],
                scale=entry["scale"],
            )
            _freeze_physics_subtree(self._stage, prim_path)

        _spawn_aliengo(self._stage, self._add_reference_to_stage)
        _spawn_control_markers(self._stage)
        self._resolved_waypoint_anchors = _spawn_waypoint_anchors(self._stage)
        self._resolved_waypoint_anchor_map = {
            anchor["name"]: anchor for anchor in self._resolved_waypoint_anchors
        }
        self._trav_map, self._map_size = _load_trav_map()

        self._render_steps(60)

        return {
            "aliengo_position": self.get_aliengo_position().tolist(),
            "scene_json": str(SCENE_JSON_PATH),
            "trav_map": str(TRAV_MAP_PATH),
            "waypoint_anchors": self._resolved_waypoint_anchors,
        }

    def get_aliengo_position(self):
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(ALIENGO_PRIM_PATH)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        translation = matrix.ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]], dtype=float)

    def get_anchor_xy(self, anchor_name: str):
        anchor = self._resolved_waypoint_anchor_map[anchor_name]
        return np.array(anchor["position"][:2], dtype=float)

    def _set_aliengo_pose(self, position_xy, yaw):
        quat_xyzw = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        _apply_transform(
            stage=self._stage,
            prim_path=ALIENGO_PRIM_PATH,
            position=(position_xy[0], position_xy[1], ALIENGO_POSITION[2]),
            orientation_xyzw=quat_xyzw,
            scale=ALIENGO_SCALE,
        )
        _apply_transform(
            stage=self._stage,
            prim_path="/World/Robots/aliengo_marker",
            position=(position_xy[0], position_xy[1], ALIENGO_POSITION[2] + 0.9),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        )

    def plan_path(self, target_xy):
        start_xy = self.get_aliengo_position()[:2]
        start_map = _nearest_free_cell(self._trav_map, _world_to_map(start_xy, self._map_size, TRAV_MAP_RESOLUTION))
        target_map = _nearest_free_cell(self._trav_map, _world_to_map(target_xy, self._map_size, TRAV_MAP_RESOLUTION))
        path_map = _astar(self._trav_map, start_map, target_map)
        path_world = _map_to_world(np.array(path_map), self._map_size, TRAV_MAP_RESOLUTION)
        return np.asarray(path_world, dtype=float)

    def navigate_to(self, target_xy, dump_path=None):
        path_world = self.plan_path(target_xy)
        steps = 0
        current_xy = self.get_aliengo_position()[:2]

        for waypoint in path_world[1:]:
            waypoint = np.asarray(waypoint, dtype=float)
            while np.linalg.norm(waypoint - current_xy) > WAYPOINT_TOLERANCE:
                delta = waypoint - current_xy
                dist = np.linalg.norm(delta)
                direction = delta / max(dist, 1e-8)
                step_size = min(ALIENGO_MOVE_STEP, dist)
                current_xy = current_xy + direction * step_size
                yaw = math.atan2(direction[1], direction[0])
                self._set_aliengo_pose(current_xy, yaw)
                self._render_steps(1)
                steps += 1

        self._render_steps(POST_NAV_HOLD_STEPS)

        result = NavigateResult(
            success=True,
            steps=steps,
            path_world=np.asarray(path_world).tolist(),
            final_position=self.get_aliengo_position().tolist(),
            target_position=[float(target_xy[0]), float(target_xy[1]), ALIENGO_POSITION[2]],
        )
        if dump_path is not None:
            result.dump_json(Path(dump_path))
        return result

    def navigate_to_anchor(self, anchor_name: str, dump_path=None):
        return self.navigate_to(self.get_anchor_xy(anchor_name), dump_path=dump_path)

    def close(self):
        if self._simulation_app is not None:
            self._simulation_app.close()


def _spawn_control_markers(stage):
    from pxr import Gf, UsdGeom

    for point in DEBUG_CONTROL_POINTS:
        safe_name = (
            point["name"]
            .replace(".", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        prim_path = f"/World/DebugPoints/{safe_name}"
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.GetRadiusAttr().Set(0.10)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*point["color"])])
        UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(tuple(point["position"]))


def _get_prim_world_position(stage, prim_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim not found for waypoint anchor: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    translation = matrix.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]], dtype=float)


def _safe_prim_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace(" ", "_")


def _get_debug_point_position(stage, name: str):
    prim_path = f"/World/DebugPoints/{_safe_prim_name(name)}"
    return _get_prim_world_position(stage, prim_path)


def _resolve_axis_value(stage, resolved_positions, axis_spec: dict, axis_index: int) -> float:
    source = axis_spec["source"]
    offset = float(axis_spec.get("offset", 0.0))

    if source == "anchor":
        value = resolved_positions[axis_spec["name"]][axis_index]
    elif source == "prim":
        value = _get_prim_world_position(stage, axis_spec["path"])[axis_index]
    elif source == "value":
        value = float(axis_spec["value"])
    else:
        raise ValueError(f"Unsupported axis source: {source}")

    return float(value) + offset


def _spawn_waypoint_anchors(stage):
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World/WaypointAnchors")
    resolved_points = []
    resolved_positions = {}

    for anchor in WAYPOINT_ANCHORS:
        mode = anchor.get("mode", "offset_from_prim")
        if mode == "offset_from_prim":
            base_position = _get_prim_world_position(stage, anchor["asset_prim_path"])
            position = base_position + np.array(anchor.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        elif mode == "compose_axes":
            position = np.array(
                [
                    _resolve_axis_value(stage, resolved_positions, anchor["x"], 0),
                    _resolve_axis_value(stage, resolved_positions, anchor["y"], 1),
                    _resolve_axis_value(stage, resolved_positions, anchor["z"], 2),
                ],
                dtype=float,
            )
        else:
            raise ValueError(f"Unsupported waypoint anchor mode: {mode}")

        # For path planning we only care about XY. Keep the displayed waypoint spheres
        # on a consistent height so the graph is easier to inspect visually.
        position[2] = float(anchor.get("display_z", WAYPOINT_DISPLAY_Z))

        safe_name = _safe_prim_name(anchor["name"])
        prim_path = f"/World/WaypointAnchors/{safe_name}"
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.GetRadiusAttr().Set(0.14)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*anchor["color"])])
        UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(tuple(position.tolist()))
        resolved_positions[anchor["name"]] = position.tolist()
        resolved_points.append(
            {
                "name": anchor["name"],
                "position": position.tolist(),
                "mode": mode,
                "neighbors": anchor.get("neighbors", []),
            }
        )

    return resolved_points


def _load_scene_entries():
    with open(SCENE_JSON_PATH, "r", encoding="utf-8") as scene_file:
        scene_data = json.load(scene_file)

    entries = []
    for index, (name, init_info) in enumerate(scene_data["objects_info"]["init_info"].items()):
        if MAX_OBJECTS is not None and index >= MAX_OBJECTS:
            break

        args = init_info["args"]
        category = args["category"]
        if category in SKIP_CATEGORIES:
            continue
        state = scene_data["state"]["object_registry"].get(name)
        if state is None or "root_link" not in state:
            continue

        entries.append(
            {
                "name": name,
                "category": category,
                "model": args["model"],
                "scale": args.get("scale", [1.0, 1.0, 1.0]),
                "position": state["root_link"]["pos"],
                "orientation": state["root_link"]["ori"],
                "joint_state": state.get("joints", {}),
            }
        )
    return entries


def _add_basic_lighting(stage):
    from pxr import Gf, UsdGeom, UsdLux

    # =========================
    # 🌤 全局环境光（避免黑背景）
    # =========================
    dome = UsdLux.DomeLight.Define(stage, "/World/Looks/DomeLight")
    dome.CreateIntensityAttr(1800.0)     # ↑ 提高环境亮度
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.96))  # 微暖色

    # =========================
    # ☀️ 太阳光（柔一点）
    # =========================
    distant = UsdLux.DistantLight.Define(stage, "/World/Looks/DistantLight")
    distant.CreateIntensityAttr(800.0)   # ↓ 降低避免过曝
    distant.CreateAngleAttr(1.0)         # ↑ 更柔和阴影
    distant.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))

    xform = stage.GetPrimAtPath("/World/Looks/DistantLight")
    light_xform = UsdGeom.Xformable(xform)
    rotate = light_xform.AddRotateXYZOp()
    rotate.Set(Gf.Vec3f(315.0, 0.0, 35.0))

    # =========================
    # 💡 室内大面积顶灯（核心）
    # =========================
    for light_cfg in INTERIOR_LIGHTS:
        light_path = f"/World/Looks/{light_cfg['name']}"
        rect = UsdLux.RectLight.Define(stage, light_path)

        # 🔥 关键参数（已经调优）
        rect.CreateIntensityAttr(42000.0)   # 高亮
        rect.CreateExposureAttr(1.2)        # 再提升一档（≈2.3倍）
        rect.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.92))  # 室内暖光

        # 🔥 灯变大（核心！！）
        rect.CreateWidthAttr(2.8)
        rect.CreateHeightAttr(2.8)

        # 🔧 位置 + 朝下
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(light_path))
        translate = xform.AddTranslateOp()
        rotate = xform.AddRotateXYZOp()

        translate.Set(Gf.Vec3d(*light_cfg["position"]))
        rotate.Set(Gf.Vec3f(180.0, 0.0, 0.0))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = ManualAliengoNavigateAPI()
    obs = api.start()
    print(f"Scene ready: {obs}")
    print("Debug control points:")
    for point in DEBUG_CONTROL_POINTS:
        print(f"  - {point['name']}: {point['position']}")
    print("Resolved waypoint anchors:")
    for anchor in obs.get("waypoint_anchors", []):
        print(
            f"  - {anchor['name']}: {anchor['position']} "
            f"neighbors={anchor.get('neighbors', [])}"
        )

    try:
        if ENABLE_MOVEMENT_DEMO:
            for waypoint_index, anchor_name in enumerate(DEMO_WAYPOINTS[1:], start=1):
                result = api.navigate_to_anchor(
                    anchor_name,
                    dump_path=OUTPUT_DIR / f"aliengo_merom_waypoint_{waypoint_index}.json",
                )
                print(
                    f"waypoint_{waypoint_index} => "
                    f"success={result.success}, "
                    f"steps={result.steps}, "
                    f"target_anchor={anchor_name}, "
                    f"final={result.final_position}"
                )
            print("All waypoints finished. Holding final frame...")
            api._render_steps(FINAL_HOLD_STEPS)
        else:
            print("Movement demo disabled. Only scene, robot, and debug markers are loaded.")
            while api._simulation_app.is_running():
                api._simulation_app.update()
    finally:
        api.close()


if __name__ == "__main__":
    main()
