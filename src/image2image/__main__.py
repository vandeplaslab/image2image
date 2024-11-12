"""image2image - suite of tools to visualise imaging data."""

import sys
from multiprocessing import freeze_support, set_start_method


def main() -> None:
    """Main entry point for the ionglow CLI."""
    from image2image.cli import cli

    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli.main(windows_expand_args=False)


if __name__ == "__main__":
    from image2image.cli import cli

    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli.main(windows_expand_args=False)

