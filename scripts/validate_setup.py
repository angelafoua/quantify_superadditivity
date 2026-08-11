"""Validate the environment is correctly set up for experiments.

Usage:
    python scripts/validate_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(name: str, fn) -> bool:
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("Superadditivity Environment Validation")
    print("=" * 60)
    all_ok = True

    print("\n1. Core dependencies:")
    all_ok &= check("torch", lambda: __import__("torch").__version__)
    all_ok &= check("torchvision", lambda: __import__("torchvision").__version__)
    all_ok &= check("numpy", lambda: __import__("numpy").__version__)
    all_ok &= check("scipy", lambda: __import__("scipy").__version__)
    all_ok &= check("pandas", lambda: __import__("pandas").__version__)
    all_ok &= check("sklearn", lambda: __import__("sklearn").__version__)
    all_ok &= check("networkx", lambda: __import__("networkx").__version__)
    all_ok &= check("h5py", lambda: __import__("h5py").__version__)
    all_ok &= check("hydra", lambda: __import__("hydra").__version__)
    all_ok &= check("statsmodels", lambda: __import__("statsmodels").__version__)
    all_ok &= check("matplotlib", lambda: __import__("matplotlib").__version__)
    all_ok &= check("seaborn", lambda: __import__("seaborn").__version__)

    print("\n2. GPU / Device:")
    import torch
    all_ok &= check("CUDA available", lambda: torch.cuda.is_available())
    if torch.cuda.is_available():
        check("GPU count", lambda: torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            check(f"GPU {i}", lambda i=i: torch.cuda.get_device_name(i))
        check("CUDA version", lambda: torch.version.cuda)

    print("\n3. Package import:")
    all_ok &= check("superadditivity", lambda: __import__("superadditivity").__version__)
    all_ok &= check("superadditivity.datasets", lambda: __import__("superadditivity.datasets") and "OK")
    all_ok &= check("superadditivity.models", lambda: __import__("superadditivity.models") and "OK")
    all_ok &= check("superadditivity.graphs", lambda: __import__("superadditivity.graphs") and "OK")
    all_ok &= check("superadditivity.communication", lambda: __import__("superadditivity.communication") and "OK")
    all_ok &= check("superadditivity.evaluation", lambda: __import__("superadditivity.evaluation") and "OK")
    all_ok &= check("superadditivity.training", lambda: __import__("superadditivity.training") and "OK")
    all_ok &= check("superadditivity.analysis", lambda: __import__("superadditivity.analysis") and "OK")
    all_ok &= check("superadditivity.visualization", lambda: __import__("superadditivity.visualization") and "OK")
    all_ok &= check("superadditivity.logging", lambda: __import__("superadditivity.logging") and "OK")

    print("\n4. Config files:")
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    for subdir in ["data", "graph", "model", "training", "experiment"]:
        yamls = list((config_dir / subdir).glob("*.yaml"))
        all_ok &= check(f"configs/{subdir}", lambda y=yamls: f"{len(y)} files")

    print("\n5. Seeding:")
    from superadditivity.utils.seed import set_all_seeds, derived_seed, PROBE_SEED
    set_all_seeds(42)
    all_ok &= check("PROBE_SEED", lambda: PROBE_SEED)
    all_ok &= check("derived_seed(42, 'weight_init')", lambda: derived_seed(42, kind="weight_init"))
    all_ok &= check("derived_seed(42, 'dirichlet')", lambda: derived_seed(42, kind="dirichlet"))

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review output above")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
