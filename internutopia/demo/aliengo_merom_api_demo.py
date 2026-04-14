import importlib.util
from pathlib import Path


def _load_og_demo_module():
    project_root = Path(__file__).resolve().parents[3]
    target_file = project_root / "COHERENT" / "OmniGibson" / "Benchmark" / "aliengo_merom_api_demo.py"
    if not target_file.exists():
        raise FileNotFoundError(f"Target demo not found: {target_file}")

    spec = importlib.util.spec_from_file_location("og_aliengo_merom_api_demo", target_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {target_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_og_demo_module()
    module.main()


if __name__ == "__main__":
    main()
