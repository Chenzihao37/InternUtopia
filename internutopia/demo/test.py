import os
import json
import difflib
from pathlib import Path
from urllib import request, error

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.aliengo import AliengoRobotCfg, move_to_point_cfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg


os.environ["OMNI_USD_RESOLVER_SEARCH_PATH"] = "/home/zyserver/work/my_project/COHERENT/OmniGibson/omnigibson/data/og_dataset"

SCENE_USD_PATH = "/home/zyserver/work/my_project/InternUtopia/internutopia/demo/merom_scene_baked.usd"
SCENE_JSON_PATH = "/home/zyserver/work/my_project/COHERENT/OmniGibson/omnigibson/data/og_dataset/scenes/Merom_1_int/json/Merom_1_int_best.json"
SCENE_ROOT_PRIM_PATH = "/World/env_0/scene/MeromScene"
ROBOT_PRIM_PATH = "/World/env_0/robots/aliengo"
SPAWN_REFERENCE_PRIM_PATH = "/World/env_0/scene/MeromScene/carpet_ctclvd_1/Asset/base_link/visuals"
SPAWN_Y_OFFSET = -0.3
ROBOT_Z_HEIGHT = 1.05
NAV_COMMAND = "导航到 卧室"
NAV_TARGET_CLEARANCE = 0.8
OCCUPANCY_CELL_SIZE = 0.15
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
ROOM_ALIASES = {
    "卧室": "bedroom_0",
    "主卧": "bedroom_0",
    "bedroom": "bedroom_0",
    "儿童房": "childs_room_0",
    "孩子房间": "childs_room_0",
    "child room": "childs_room_0",
    "childs room": "childs_room_0",
    "客厅": "living_room_0",
    "living room": "living_room_0",
    "餐厅": "dining_room_0",
    "dining room": "dining_room_0",
    "厨房": "kitchen_0",
    "kitchen": "kitchen_0",
    "卫生间": "bathroom_0",
    "浴室": "bathroom_0",
    "bathroom": "bathroom_0",
}


def add_basic_lighting(stage):
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(5000)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500)
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Sun")).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 35))

    fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(500)
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/FillLight")).AddRotateXYZOp().Set(Gf.Vec3f(45, 0, -30))


def _get_world_center(stage, prim_path):
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim not found: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_pt = world_bbox.GetMin()
    max_pt = world_bbox.GetMax()
    center = (min_pt + max_pt) * 0.5
    return np.array([center[0], center[1], center[2]], dtype=float)


def _disable_instances_and_add_collision(stage):
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)

    collision_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        try:
            UsdPhysics.CollisionAPI.Apply(prim)
            physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx.CreateApproximationAttr().Set("convexHull")
            collision_count += 1
        except Exception:
            pass

    print(f"✅ Collision added to {collision_count} meshes")


def _parse_command(command: str) -> str:
    normalized = command.strip()
    if normalized.startswith("导航到"):
        return normalized.replace("导航到", "", 1).strip()
    return normalized


def _load_scene_room_map():
    with open(SCENE_JSON_PATH, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    room_map = {}
    for object_name, init_info in scene_data["objects_info"]["init_info"].items():
        rooms = init_info["args"].get("in_rooms") or []
        for room_name in rooms:
            room_map.setdefault(room_name, []).append(object_name)

    return room_map


SCENE_ROOM_MAP = _load_scene_room_map()


def _list_scene_object_names(stage):
    scene_root = stage.GetPrimAtPath(SCENE_ROOT_PRIM_PATH)
    if not scene_root.IsValid():
        return []
    return sorted(child.GetName() for child in scene_root.GetChildren())


def _extract_response_text(response_json):
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _resolve_object_name_with_openai(command: str, candidates: list[str]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "You map a Chinese or English navigation command to exactly one scene object name. "
                    "Choose only from the provided candidates. "
                    "If no candidate is plausible, return the closest candidate with low confidence and explain briefly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Command: {command}\n"
                    f"Candidates: {json.dumps(candidates, ensure_ascii=False)}\n"
                    "Return JSON with keys object_name, confidence, reason."
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scene_object_match",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["object_name", "confidence", "reason"],
                    "additionalProperties": False,
                },
            }
        },
    }

    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            response_json = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(f"⚠️ OpenAI API request failed: {e.code} {detail}")
        return None
    except Exception as e:
        print(f"⚠️ OpenAI API unavailable, fallback to local matcher. reason={e}")
        return None

    text = _extract_response_text(response_json)
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"⚠️ OpenAI returned non-JSON text: {text}")
        return None

    object_name = parsed.get("object_name")
    if object_name in candidates:
        print(
            f"✅ OpenAI command grounding: command={command}, object={object_name}, "
            f"confidence={parsed.get('confidence')}, reason={parsed.get('reason')}"
        )
        return object_name

    return None


def _resolve_room_name(command: str) -> str | None:
    parsed = _parse_command(command).strip().lower()
    if parsed in ROOM_ALIASES:
        return ROOM_ALIASES[parsed]

    for alias, room_name in ROOM_ALIASES.items():
        if alias in parsed:
            return room_name

    return None


def _resolve_object_name_locally(command: str, candidates: list[str]) -> str:
    parsed = _parse_command(command)
    if parsed in candidates:
        return parsed

    normalized = parsed.lower()
    lower_map = {name.lower(): name for name in candidates}
    if normalized in lower_map:
        return lower_map[normalized]

    contains = [name for name in candidates if normalized in name.lower() or name.lower() in normalized]
    if contains:
        return contains[0]

    matches = difflib.get_close_matches(parsed, candidates, n=1, cutoff=0.2)
    if matches:
        return matches[0]

    raise ValueError(f"Could not resolve command '{command}' to any scene object.")


def _build_target_prim_path(object_name: str) -> str:
    return f"{SCENE_ROOT_PRIM_PATH}/{object_name}"


def _fallback_target_from_object(stage, object_name: str, clearance: float):
    object_center = _get_world_center(stage, _build_target_prim_path(object_name))
    return np.array([object_center[0], object_center[1] - clearance, 0.0], dtype=float)


def _fallback_target_from_room(stage, room_name: str, clearance: float):
    object_names = SCENE_ROOM_MAP.get(room_name, [])
    if not object_names:
        raise ValueError(f"Could not resolve room '{room_name}' to any scene objects.")

    centers = []
    valid_object_names = []
    for object_name in object_names:
        prim_path = _build_target_prim_path(object_name)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        centers.append(_get_world_center(stage, prim_path))
        valid_object_names.append(object_name)

    if not centers:
        raise ValueError(f"Room '{room_name}' has no valid prims in the loaded stage.")

    room_center = np.mean(np.asarray(centers, dtype=float), axis=0)
    target = np.array([room_center[0], room_center[1] - clearance, 0.0], dtype=float)
    return valid_object_names[0], target


def _try_find_nav_target_with_occupancy(stage, object_name: str, clearance: float):
    try:
        import omni
        from isaacsim.asset.gen.omap.bindings import _omap

        scene_prim = stage.GetPrimAtPath(SCENE_ROOT_PRIM_PATH)
        scene_center = _get_world_center(stage, SCENE_ROOT_PRIM_PATH)
        target_center = _get_world_center(stage, _build_target_prim_path(object_name))

        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        world_bbox = bbox_cache.ComputeWorldBound(scene_prim).ComputeAlignedBox()
        min_pt = world_bbox.GetMin()
        max_pt = world_bbox.GetMax()

        generator = _omap.Generator(
            omni.physx.acquire_physx_interface(),
            omni.usd.get_context().get_stage_id(),
        )
        generator.update_settings(OCCUPANCY_CELL_SIZE, 4, 0, 255)
        generator.set_transform(
            (float(scene_center[0]), float(scene_center[1]), 0.0),
            float(min_pt[0]),
            float(max_pt[0]),
            float(min_pt[1]),
            float(max_pt[1]),
        )
        generator.generate2d()

        free_positions = np.asarray(generator.get_free_positions(), dtype=float)
        if free_positions.size == 0:
            return None

        free_xy = free_positions[:, :2]
        target_xy = np.array([target_center[0], target_center[1]], dtype=float)

        deltas = free_xy - target_xy[None, :]
        dists = np.linalg.norm(deltas, axis=1)
        valid_ids = np.where(dists >= clearance)[0]
        if valid_ids.size == 0:
            valid_ids = np.arange(len(free_positions))

        best_id = valid_ids[np.argmin(dists[valid_ids])]
        best_xy = free_xy[best_id]
        return np.array([best_xy[0], best_xy[1], 0.0], dtype=float)

    except Exception as e:
        print(f"⚠️ Occupancy map unavailable, fallback to heuristic target. reason={e}")
        return None


def _resolve_navigation_target(stage, command: str):
    candidates = _list_scene_object_names(stage)
    object_name = _resolve_object_name_with_openai(command, candidates)
    if object_name is None:
        try:
            object_name = _resolve_object_name_locally(command, candidates)
        except ValueError:
            room_name = _resolve_room_name(command)
            if room_name is None:
                raise
            object_name, nav_target = _fallback_target_from_room(stage, room_name, NAV_TARGET_CLEARANCE)
            print(f"✅ Room grounding fallback: command={command}, room={room_name}, anchor={object_name}")
            return object_name, nav_target

    nav_target = _try_find_nav_target_with_occupancy(stage, object_name, NAV_TARGET_CLEARANCE)
    if nav_target is None:
        nav_target = _fallback_target_from_object(stage, object_name, NAV_TARGET_CLEARANCE)

    return object_name, nav_target


def _teleport_aliengo_to_spawn(stage):
    from omni.isaac.core.articulations import ArticulationView

    spawn_center = _get_world_center(stage, SPAWN_REFERENCE_PRIM_PATH)
    spawn_position = np.array([spawn_center[0], spawn_center[1] + SPAWN_Y_OFFSET, ROBOT_Z_HEIGHT], dtype=float)

    robot_view = ArticulationView(prim_paths_expr=ROBOT_PRIM_PATH, name="aliengo_view")
    robot_view.initialize()

    current_positions, current_orientations = robot_view.get_world_poses()
    current_positions[0] = spawn_position
    robot_view.set_world_poses(current_positions, current_orientations)
    robot_view.set_linear_velocities(np.zeros((1, 3), dtype=np.float32))
    robot_view.set_angular_velocities(np.zeros((1, 3), dtype=np.float32))

    print(f"✅ Aliengo spawn set to {spawn_position.tolist()} from {SPAWN_REFERENCE_PRIM_PATH}")


headless = not has_display()

config = Config(
    simulator=SimConfig(
        physics_dt=1 / 240,
        rendering_dt=1 / 240,
        use_fabric=False,
        headless=headless,
        webrtc=headless,
    ),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=SCENE_USD_PATH,
            robots=[
                AliengoRobotCfg(
                    position=(2.8, 1.8, ROBOT_Z_HEIGHT),
                    controllers=[move_to_point_cfg],
                )
            ],
        ),
    ],
)


import_extensions()

env = Env(config)
obs, _ = env.reset()

from omni.isaac.core.utils.stage import get_current_stage

stage = get_current_stage()
add_basic_lighting(stage)
print("✅ Lighting added")

_disable_instances_and_add_collision(stage)
_teleport_aliengo_to_spawn(stage)

target_object_name, env_target = _resolve_navigation_target(stage, NAV_COMMAND)
print(f"✅ Command parsed: {NAV_COMMAND} -> object={target_object_name}, target={env_target.tolist()}")

env_action = {
    move_to_point_cfg.name: [tuple(env_target)],
}
print(f"✅ Action: {env_action}")


# ====== 参数 ======
TARGET_POS = np.array(env_target, dtype=float)
REACH_THRESHOLD = 0.2        # 到达判定（米）← 可以调
STUCK_THRESHOLD = 1e-3       # 位移变化阈值
STUCK_STEPS = 10             # 连续多少次没动算卡住
MAX_STEPS = 5000             # 最大步数保护

prev_pos = None
stuck_counter = 0

i = 0
while env.simulation_app.is_running():
    i += 1

    obs, _, terminated, _, _ = env.step(action=env_action)

    # 获取机器人位置
    robot_pos = obs.get("position") if isinstance(obs, dict) else None
    if robot_pos is None:
        continue

    robot_pos = np.array(robot_pos, dtype=float)

    # ====== 计算距离 ======
    dist = np.linalg.norm(robot_pos[:2] - TARGET_POS[:2])

    # ====== 打印 ======
    if i % 200 == 0:
        print(f"Step: {i}, robot={robot_pos}, dist={dist:.4f}")

    # ====== ✅ 到达判断 ======
    if dist < REACH_THRESHOLD:
        print(f"✅ Reached target! dist={dist:.4f}, step={i}")
        break

    # ====== ✅ 卡住检测 ======
    if prev_pos is not None:
        move_dist = np.linalg.norm(robot_pos[:2] - prev_pos[:2])

        if move_dist < STUCK_THRESHOLD:
            stuck_counter += 1
        else:
            stuck_counter = 0

        if stuck_counter >= STUCK_STEPS:
            print(f"⚠️ Robot seems STUCK! step={i}, dist={dist:.4f}")
            break

    prev_pos = robot_pos.copy()

    # ====== ✅ 终止条件 ======
    if terminated:
        print("⚠️ Episode terminated")
        break

    if i >= MAX_STEPS:
        print("⚠️ Max steps reached")
        break

env.close()
