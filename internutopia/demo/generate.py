import json
import shutil
from pathlib import Path
from cryptography.fernet import Fernet
from omni.isaac.kit import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OG_ROOT = PROJECT_ROOT.parent / "COHERENT" / "OmniGibson"
OG_DATA_ROOT = OG_ROOT / "omnigibson" / "data"
OG_DATASET_ROOT = OG_DATA_ROOT / "og_dataset"

SCENE_JSON_PATH = OG_DATASET_ROOT / "scenes" / "Merom_1_int" / "json" / "Merom_1_int_best.json"
KEY_PATH = OG_DATA_ROOT / "omnigibson.key"

OUTPUT_USD = Path(__file__).resolve().parent / "merom_scene_baked.usd"
OUTPUT_DIR = OUTPUT_USD.parent
COMPOSED_USD = OUTPUT_DIR / "merom_scene_composed.usda"
LOCAL_ASSET_ROOT = OUTPUT_DIR / "baked_scene_assets"
DECRYPT_CACHE_DIR = OUTPUT_DIR / "logs" / "merom_bake_cache"

SKIP_CATEGORIES = {
    "ceilings",
    "doors",
    "door",
}


simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


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


def _load_scene_entries():
    with open(SCENE_JSON_PATH, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    entries = []

    for name, init_info in scene_data["objects_info"]["init_info"].items():
        args = init_info["args"]
        category = args["category"]

        # ✅ 过滤 ceilings
        if category in SKIP_CATEGORIES:
            continue

        # ✅ 过滤 door（关键！！）
        if "door" in category.lower():
            print(f"� Skipping door: {name} ({category})")
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


def _copy_asset_package(local_usd_path: Path) -> Path:
    relative_dir = local_usd_path.relative_to(DECRYPT_CACHE_DIR).parent
    target_dir = LOCAL_ASSET_ROOT / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for sibling in local_usd_path.parent.iterdir():
        target = target_dir / sibling.name
        if sibling.is_dir():
            shutil.copytree(sibling, target, dirs_exist_ok=True)
        else:
            shutil.copy2(sibling, target)

    return target_dir / local_usd_path.name


def _make_output_relative(asset_path: Path) -> str:
    return str(asset_path.relative_to(OUTPUT_DIR))


def _apply_transform(stage, prim_path, position, orientation_xyzw, scale):
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


def _remove_property_if_exists(prim, name):
    prop = prim.GetProperty(name)
    if prop:
        prim.RemoveProperty(name)


def _strip_physics_from_stage(stage):
    joint_like_types = {
        "FixedJoint",
        "RevoluteJoint",
        "PrismaticJoint",
        "SphericalJoint",
        "DistanceJoint",
        "D6Joint",
    }

    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)

        type_name = prim.GetTypeName()
        if type_name in joint_like_types or prim.HasRelationship("physics:body0") or prim.HasRelationship("physics:body1"):
            prim.SetActive(False)
            continue

        for prop in list(prim.GetProperties()):
            name = prop.GetName()
            if name.startswith("physics:") or name.startswith("physx"):
                prim.RemoveProperty(name)

        for api in (
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.DriveAPI,
            UsdPhysics.FilteredPairsAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.MaterialAPI,
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.RigidBodyAPI,
            PhysxSchema.PhysxArticulationAPI,
            PhysxSchema.PhysxCollisionAPI,
            PhysxSchema.PhysxConvexHullCollisionAPI,
            PhysxSchema.PhysxCookedDataAPI,
            PhysxSchema.PhysxDeformableBodyAPI,
            PhysxSchema.PhysxDeformableSurfaceAPI,
            PhysxSchema.PhysxParticleAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            PhysxSchema.PhysxTriggerAPI,
        ):
            try:
                prim.RemoveAPI(api)
            except Exception:
                pass

    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            prim.SetActive(False)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
DECRYPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

stage = Usd.Stage.CreateNew(str(COMPOSED_USD))

UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/MeromScene")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

entries = _load_scene_entries()
copied_models = {}

print(f"Loading {len(entries)} objects...")

for entry in entries:
    asset_path = _get_object_asset_path(entry["category"], entry["model"])
    usd_path = _decrypt_usd_if_needed(asset_path)

    if usd_path not in copied_models:
        copied_models[usd_path] = _copy_asset_package(usd_path)

    local_usd_path = copied_models[usd_path]
    reference_path = _make_output_relative(local_usd_path)
    prim_path = f"/World/MeromScene/{entry['name']}"
    asset_prim_path = f"{prim_path}/Asset"

    UsdGeom.Xform.Define(stage, prim_path)
    asset_prim = stage.DefinePrim(asset_prim_path)
    asset_prim.GetReferences().AddReference(reference_path)

    _apply_transform(
        stage,
        prim_path,
        entry["position"],
        entry["orientation"],
        entry["scale"],
    )

stage.GetRootLayer().Save()
flat_layer = stage.Flatten()
flat_layer.Export(str(OUTPUT_USD))

baked_stage = Usd.Stage.Open(str(OUTPUT_USD))
_strip_physics_from_stage(baked_stage)
baked_stage.GetRootLayer().Save()

print("✅ DONE")
print("Scene USD:", OUTPUT_USD)
print("Composed USD:", COMPOSED_USD)
print("Assets copied:", len(copied_models))
print("Output asset root:", LOCAL_ASSET_ROOT)

simulation_app.close()
