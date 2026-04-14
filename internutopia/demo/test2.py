import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.aliengo import AliengoRobotCfg, move_to_point_cfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

# ===== 场景路径（改成你的）=====
SCENE_USD_PATH = "/home/zyserver/work/my_project/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/scenes/MVUHLWYKTKJ5EAABAAAAADA8_usd/start_result_navigation.usd"

ROBOT_PRIM_PATH = "/World/env_0/robots/aliengo"

# ===== 手动目标点（你可以随便改）=====
TARGET_XY = np.array([3.0, 3.0])   # ← 可以改
TARGET_Z = 0.0

OCCUPANCY_CELL_SIZE = 0.15
REACH_THRESHOLD = 0.4
WAYPOINT_REACH_THRESHOLD = 0.3


# ================= 可视化 =================

def create_sphere(stage, path, pos, color):
    from pxr import UsdGeom, Gf, UsdShade, Sdf

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(0.25)

    xform = UsdGeom.Xformable(sphere)
    xform.AddTranslateOp().Set(Gf.Vec3f(float(pos[0]), float(pos[1]), 0.3))

    material_path = path + "_mat"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/shader")

    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)


def hide_ceiling(stage):
    from pxr import UsdGeom
    count = 0
    for prim in stage.Traverse():
        if "ceil" in prim.GetName().lower():
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                count += 1
            except:
                pass
    print(f"✅ Hidden ceiling: {count}")


# ================= 机器人 =================

def get_robot_pos():
    from omni.isaac.core.articulations import ArticulationView
    view = ArticulationView(prim_paths_expr=ROBOT_PRIM_PATH)
    view.initialize()
    pos, _ = view.get_world_poses()
    return np.array(pos[0])


# ================= occupancy =================

def generate_occupancy(stage):
    import omni, carb
    from isaacsim.asset.gen.omap.bindings import _omap
    from pxr import UsdGeom, Usd

    scene = stage.GetPrimAtPath("/World")

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(scene).ComputeAlignedBox()

    min_pt = bbox.GetMin()
    max_pt = bbox.GetMax()

    gen = _omap.Generator(
        omni.physx.acquire_physx_interface(),
        omni.usd.get_context().get_stage_id()
    )

    gen.update_settings(OCCUPANCY_CELL_SIZE, 4, 0, 255)

    origin = carb.Float3(min_pt[0], min_pt[1], 0.0)
    lower = carb.Float3(0, 0, 0)
    upper = carb.Float3(max_pt[0]-min_pt[0], max_pt[1]-min_pt[1], 0)

    gen.set_transform(origin, lower, upper)
    gen.generate2d()

    free = np.array(gen.get_free_positions())
    origin_xy = np.array([min_pt[0], min_pt[1]])

    return free, origin_xy


def build_grid(free, origin_xy):
    grid = {}
    pts = []

    for p in free:
        wx = p[0] + origin_xy[0]
        wy = p[1] + origin_xy[1]

        gx = int(round(wx / OCCUPANCY_CELL_SIZE))
        gy = int(round(wy / OCCUPANCY_CELL_SIZE))

        key = (gx, gy)
        if key not in grid:
            grid[key] = np.array([wx, wy])
            pts.append((key, grid[key]))

    return grid, pts


# ================= A* =================

def astar(start, goal, free_keys):
    import heapq

    def h(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    open_set = [(0, start)]
    came = {}
    g = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]

        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nxt = (cur[0]+dx, cur[1]+dy)
                if nxt not in free_keys:
                    continue

                cost = g[cur] + np.hypot(dx, dy)
                if cost < g.get(nxt, 1e9):
                    g[nxt] = cost
                    came[nxt] = cur
                    heapq.heappush(open_set, (cost + h(nxt, goal), nxt))

    return None

# def simplify_scene(stage):
#     from pxr import UsdGeom

#     removed = 0
#     kept = 0

#     for prim in list(stage.Traverse()):

#         path = str(prim.GetPath()).lower()

#         # ===== 保留这些 =====
#         if any(k in path for k in [
#             "floor",
#             "ground",
#             "wall",
#             "room",
#             "doorframe",
#             "structure"
#         ]):
#             kept += 1
#             continue

#         # ===== 删除这些（关键优化）=====
#         if any(k in path for k in [
#             "furniture",
#             "towel",
#             "tv",
#             "decoration",
#             "props",
#             "clutter",
#             "small"
#         ]):
#             try:
#                 stage.RemovePrim(prim.GetPath())
#                 removed += 1
#             except:
#                 pass

#     print(f"� Scene simplified: removed={removed}, kept={kept}")

def simplify_scene(stage):
    removed = 0
    kept = 0

    SCENE_PREFIX = "/World/env_0/scene"
    ROBOT_PREFIX = "/World/env_0/robots"

    KEEP_KEYWORDS = [
        "floor", "ground",
        "wall", "walls",
        "doorframe", "door_frame"
    ]

    to_remove = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        path_l = path.lower()

        # ===== 1. robot整棵树必须保留 =====
        if path.startswith(ROBOT_PREFIX):
            kept += 1
            continue

        # ===== 2. scene外层结构不动 =====
        if not path.startswith(SCENE_PREFIX):
            kept += 1
            continue

        # ===== 3. scene根节点不删 =====
        if path in [
            "/World/env_0/scene",
            "/World/env_0/scene/Meshes",
        ]:
            kept += 1
            continue

        # ===== 4. 只保留 floor / wall =====
        if any(k in path_l for k in KEEP_KEYWORDS):
            kept += 1
            continue

        # ===== 5. 其他一律删除（但只删叶子）=====
        if not prim.GetChildren():
            to_remove.append(prim.GetPath())
        else:
            # 如果是中间节点但不重要，也删
            to_remove.append(prim.GetPath())

    # 执行删除
    for p in to_remove:
        try:
            stage.RemovePrim(p)
            removed += 1
        except:
            pass

    print(f"� Simplified HARD: removed={removed}, kept={kept}")


# ================= 初始化 =================

headless = not has_display()

config = Config(
    simulator=SimConfig(headless=headless),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=SCENE_USD_PATH,
            robots=[AliengoRobotCfg(controllers=[move_to_point_cfg])]
        )
    ],
)

import_extensions()
env = Env(config)
env.reset()

from omni.isaac.core.utils.stage import get_current_stage
stage = get_current_stage()

# ⭐ 加这里
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.asset.gen.omap", True
)

simplify_scene(stage)
hide_ceiling(stage)

# ================= 位置 =================

robot_pos = get_robot_pos()
goal = np.array([TARGET_XY[0], TARGET_XY[1], TARGET_Z])

print("\n=== WORLD ===")
print("robot:", robot_pos)
print("goal:", goal)

# ===== 画球 =====
create_sphere(stage, "/Debug/robot", robot_pos, (0,0,1))   # 蓝
create_sphere(stage, "/Debug/goal", goal, (1,0,0))         # 红

# ================= 规划 =================

free, origin_xy = generate_occupancy(stage)
grid, pts = build_grid(free, origin_xy)
free_keys = set(grid.keys())

def nearest(xy):
    best = None
    dmin = 1e9
    for k, p in pts:
        d = np.linalg.norm(p - xy[:2])
        if d < dmin:
            dmin = d
            best = k
    return best

start_k = nearest(robot_pos)
goal_k = nearest(goal)

print("NavMesh size:", len(free_keys))

path = astar(start_k, goal_k, free_keys)

waypoints = [np.array([grid[k][0], grid[k][1], 0]) for k in path]

# ================= 执行 =================

target = waypoints[0]

while env.simulation_app.is_running():

    action = {move_to_point_cfg.name: [tuple(target)]}
    obs, _, _, _, _ = env.step(action)

    pos = np.array(obs["position"])
    d = np.linalg.norm(pos[:2] - target[:2])

    if d < WAYPOINT_REACH_THRESHOLD:
        if len(waypoints) > 1:
            waypoints.pop(0)
            target = waypoints[0]
            print("➡️ next:", target)

    if np.linalg.norm(pos[:2] - goal[:2]) < REACH_THRESHOLD:
        print("✅ reached")
        break