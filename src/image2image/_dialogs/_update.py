"""Check for latest version on GitHub."""
import qtextra.helpers as hp
from loguru import logger
from qtpy.QtWidgets import QWidget


def check_version(parent: QWidget, add_to_menu: bool = True) -> None:
    """Check for latest version."""
    from koyo.release import is_new_version_available

    from image2image import __version__

    is_new_available, reason = is_new_version_available(current_version=__version__, package="image2image-docs")
    if not is_new_available:
        logger.debug("Failed to check GitHub for latest version.")
        return
    hp.long_toast(parent, "New version available!", reason, 15_000, icon="info")
    logger.debug("Checked for latest version.")

    if hasattr(parent, "menu_help") and add_to_menu:
        hp.make_menu_item(parent, "New version available!", menu=parent.menu_help)
    if hasattr(parent, "update_status_btn"):
        parent.update_status_btn.show()
