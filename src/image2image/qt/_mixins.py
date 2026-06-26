"""Various mixin classes for Qt applications."""

import qtextra.helpers as hp
from loguru import logger
from qtextra.config import THEMES
from qtpy.QtWidgets import QWidget
from superqt.utils import create_worker, ensure_main_thread

from image2image.config import get_app_config
from image2image.qt._dialogs._update import check_version


class ThemeMixin:
    """Mixin for Qt themes."""

    theme_btn: QWidget
    _console: QWidget

    def on_toggle_theme(self) -> None:
        """Toggle theme."""
        THEMES.theme = "dark" if self.theme_btn.dark else "light"
        get_app_config().theme = THEMES.theme

    def on_changed_theme(self) -> None:
        """Update theme of the app."""
        get_app_config().theme = THEMES.theme
        THEMES.set_theme_stylesheet(self)
        # update console
        if self._console:
            self._console._console._update_theme()


class NewVersionMixin:
    """Mixin for Qt version."""

    update_status_btn: QWidget

    def on_check_new_version(self) -> None:
        """Check for the new version."""
        create_worker(
            check_version,
            _connect={
                "returned": self._on_set_new_version,
                "errored": lambda: hp.toast(
                    self, "Failed", "Failed checking for new version", icon="error", position="top_left"
                ),
            },
        )

    @ensure_main_thread()
    def _on_set_new_version(self, res: tuple[bool, str]) -> None:
        """Set the result of version check."""
        is_new_available, reason = res
        if not is_new_available:
            logger.debug("Using the latest version of the app.")
            hp.toast(
                self, "No new version", "You are using the latest version of the app.", icon="info", position="top_left"
            )
            return
        hp.long_toast(self, "New version available!", reason, 15_000, icon="info", position="top_left")
        logger.debug("Checked for latest version.")
        self.update_status_btn.show()
