"""image2image - suite of tools to visualise imaging data."""

import sys
import typing as ty
from multiprocessing import freeze_support, set_start_method


def main() -> ty.Any:
    """Main entry point for the i2i CLI."""
    from image2image.cli import main_register

    return main_register()


if __name__ == "__main__":
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    sys.exit(main())
