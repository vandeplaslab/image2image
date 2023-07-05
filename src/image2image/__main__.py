"""image2image - suite of tools to visualise imaging data."""
import sys
from multiprocessing import freeze_support, set_start_method

from image2image.cli import cli


def main():
    """Main entry point for the ionglow CLI."""
    return cli()


if __name__ == "__main__":
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    sys.exit(main())
