import json
import shutil
from pathlib import Path
import math

from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "internutopia" / "assets"
OG_ROOT = PROJECT_ROOT.parent / "COHERENT" / "OmniGibson"
OG_DATA_ROOT = OG_ROOT / "omnigibson" / "data"
OG_DATASET_ROOT = OG_DATA_ROOT / "og_dataset"

SCENE_JSON_PATH = OG_DATASET_ROOT / "scenes" / "Merom_1_int" / "json" / "Merom_1_int_best.json"
KEY_PATH = OG_DATA_ROOT / "omnigibson.key"

DECRYPT_CACHE_DIR = Path(__file__).resolve().parent / "logs" / "merom_manual_cache"

ALIENGO_POSITION = [1.2, 7.4, 0.55]
ALIENGO_USD_PATH = ASSET_ROOT / "robots" / "aliengo" / "aliengo_camera.usd"
ALIENGO_PRIM_PATH = "/World/Robots/aliengo"
ALIENGO_SCALE = [1.15, 1.15, 1.15]
ALIENGO_ORIENTATION_XYZW = [0.0, 0.0, 0.70710678, 0.70710678]

SKIP_CATEGORIES = {"ceilings"}


# =========================
# 🔓 资产处理（原样保留）
# =========================
def _get_object_asset_path(category: str, model: str) -> Path:
    usd_dir = OG_DATASET_ROOT / "objects" / category / model / "usd"
    plain_usd = usd_dir / f"{model}.usd"
    encrypted_usd = usd_dir / f"{model}.encrypted.usd"

    if plain_usd.exists():
        return plain_usd
    if encrypted_usd.exists():
        return encrypted_usd

    raise FileNotFoundError(f"Asset not found: {category}/{model}")


def _decrypt_usd_if_needed(asset_path: Path) -> Path:
    if asset_path.suffix == ".usd" and not asset_path.name.endswith(".encrypted.usd"):
        return asset_path

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

    with open(KEY_PATH, "rb") as f:
        key = f.read()

    with open(asset_path, "rb") as f:
        encrypted = f.read()

    decrypted = Fernet(key).decrypt(encrypted)

    with open(out_path, "wb") as f:
        f.write(decrypted)

    return out_path


# =========================
# 📦 场景加载
# =========================
def _load_scene_entries():
    with open(SCENE_JSON_PATH, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    entries = []

    for name, init_info in scene_data["objects_info"]["init_info"].items():
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
                "scale": args.get("scale", [1, 1, 1]),
                "position": state["root_link"]["pos"],
                "orientation": state["root_link"]["ori"],
            }
        )

    return entries

# =========================
# 💡 灯光
# =========================
def _add_basic_lighting(stage):
    from pxr import Gf, UsdLux, UsdGeom

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1500)

    distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
    distant.CreateIntensityAttr(800)

    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Sun"))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 35))


# =========================
# 🎨 transform（修复版）
# =========================
def _apply_transform(stage, prim_path, position, orientation_xyzw, scale):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)

    ops = xform.GetOrderedXformOps()

    translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    scale_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if orient_op is None:
        orient_op = xform.AddOrientOp()
    if scale_op is None:
        scale_op = xform.AddScaleOp()

    translate_op.Set(Gf.Vec3d(*position))

    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_value = Gf.Quatf(float(orientation_xyzw[3]), Gf.Vec3f(*orientation_xyzw[:3]))
    else:
        orient_value = Gf.Quatd(float(orientation_xyzw[3]), Gf.Vec3d(*orientation_xyzw[:3]))

    orient_op.Set(orient_value)
    scale_op.Set(Gf.Vec3d(*scale))


# =========================
# 🤖 加 robot
# =========================
def _spawn_aliengo(stage, add_reference_to_stage):
    from pxr import Gf, UsdGeom

    if not ALIENGO_USD_PATH.exists():
        raise FileNotFoundError(f"Aliengo USD not found: {ALIENGO_USD_PATH}")

    prim_path = ALIENGO_PRIM_PATH
    add_reference_to_stage(str(ALIENGO_USD_PATH), prim_path)

    pos = [2.8, 1.8, 0.55]
    scale = ALIENGO_SCALE

    _apply_transform(
        stage,
        prim_path,
        position=pos,
        orientation_xyzw=ALIENGO_ORIENTATION_XYZW,
        scale=scale,
    )

    marker = UsdGeom.Sphere.Define(stage, "/World/Robots/aliengo_marker")
    marker.GetRadiusAttr().Set(0.12)
    marker.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])
    UsdGeom.XformCommonAPI(marker.GetPrim()).SetTranslate((pos[0], pos[1], pos[2] + 0.9))

    print("✅ Aliengo spawned at:", pos, "scale:", scale)

    return prim_path

# =========================
# 初始位置
# =========================
def _add_big_red_marker(stage, pos):
    from pxr import UsdGeom, Gf, UsdLux

    # 🔴 大红球
    sphere = UsdGeom.Sphere.Define(stage, "/World/DEBUG_RED_MARKER")
    sphere.GetRadiusAttr().Set(0.6)  # 大一点
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1, 0, 0)])

    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(tuple(pos))

    # 💡 给红球打个点光源（保证能看到）
    light = UsdLux.SphereLight.Define(stage, "/World/DEBUG_RED_LIGHT")
    light.CreateIntensityAttr(50000.0)

    xf = UsdGeom.XformCommonAPI(light.GetPrim())
    xf.SetTranslate(tuple(pos))

def main():
    from omni.isaac.kit import SimulationApp

    sim = SimulationApp({"headless": False})

    from omni.isaac.core.utils.stage import create_new_stage, get_current_stage
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
        except ImportError:
            set_camera_view = None

    # =========================
    # 🧱 初始化 stage
    # =========================
    create_new_stage()
    stage = get_current_stage()

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/MeromScene")
    UsdGeom.Xform.Define(stage, "/World/Robots")

    # =========================
    # 💡 加灯光（关键！）
    # =========================
    _add_basic_lighting(stage)

    if set_camera_view is not None:
        set_camera_view(
            eye=[3.0, 9.0, 6.0],
            target=[0.0, 7.0, 0.8],
            camera_prim_path="/OmniverseKit_Persp",
        )

    # =========================
    # 📦 加载场景
    # =========================
    entries = _load_scene_entries()
    print(f"Loading {len(entries)} objects...")

    for entry in entries:
        prim_path = f"/World/MeromScene/{entry['name']}"

        asset_path = _get_object_asset_path(entry["category"], entry["model"])
        usd_path = _decrypt_usd_if_needed(asset_path)

        add_reference_to_stage(str(usd_path), prim_path)

        _apply_transform(
            stage,
            prim_path,
            entry["position"],
            entry["orientation"],
            entry["scale"],
        )

    print("✅ Scene loaded!")
    robot_pos = [2.8, 1.8, 0.55]

    # =========================
    # 🤖 加机器人
    # =========================
    _spawn_aliengo(stage, add_reference_to_stage)
    # _add_big_red_marker(stage, robot_pos)

    # =========================
    # 🚀 简单移动（前进2米）
    # =========================
    import math

    current_pos = list(ALIENGO_POSITION)
    target = [ALIENGO_POSITION[0] + 2.0, ALIENGO_POSITION[1], ALIENGO_POSITION[2]]

    while sim.is_running():
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]

        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 0.02:
            step = 0.02
            current_pos[0] += dx / dist * step
            current_pos[1] += dy / dist * step

            yaw = math.atan2(dy, dx)
            quat = [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]

            _apply_transform(
                stage,
                ALIENGO_PRIM_PATH,
                current_pos,
                quat,
                ALIENGO_SCALE,
            )

        sim.update()


if __name__ == "__main__":
    main()
