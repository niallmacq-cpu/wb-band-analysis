"""Launch the PyQt6 Western blot analysis app, checking dependencies first."""

import importlib
import sys

REQUIRED = {  # import name -> pip package
    "numpy": "numpy",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "PIL": "pillow",
    "tifffile": "tifffile",
    "PyQt6": "PyQt6",
}
OPTIONAL = {"matplotlib": "matplotlib"}  # profile plots and the profiles PNG


def main() -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing packages. Install them with:\n  pip install " + " ".join(missing))
        sys.exit(1)
    for module, package in OPTIONAL.items():
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"Optional: pip install {package} to see the live profile plots.")
    from wb_gui_qt import main as run

    run()


if __name__ == "__main__":
    main()
