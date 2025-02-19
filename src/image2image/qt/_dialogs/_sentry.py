"""Sentry widget."""

import os
import sys

from loguru import logger

from image2image import __version__

# setup environment variables
SENTRY_DSN = "https://5473ad03e64c4a2988b9ec143f6181c5@o1133843.ingest.sentry.io/4505464740708352"
os.environ["QTEXTRA_TELEMETRY_SENTRY_DSN"] = SENTRY_DSN
QTEXTRA_TELEMETRY_SHOW_HOSTNAME = "0"
os.environ["QTEXTRA_TELEMETRY_SHOW_HOSTNAME"] = QTEXTRA_TELEMETRY_SHOW_HOSTNAME
QTEXTRA_TELEMETRY_SHOW_LOCALS = "1"
os.environ["QTEXTRA_TELEMETRY_SHOW_LOCALS"] = QTEXTRA_TELEMETRY_SHOW_LOCALS
QTEXTRA_TELEMETRY_DEBUG = "0"
os.environ["QTEXTRA_TELEMETRY_DEBUG"] = QTEXTRA_TELEMETRY_DEBUG
QTEXTRA_TELEMETRY_VERSION = __version__
os.environ["QTEXTRA_TELEMETRY_VERSION"] = QTEXTRA_TELEMETRY_VERSION
QTEXTRA_TELEMETRY_PACKAGE = "image2image"
os.environ["QTEXTRA_TELEMETRY_PACKAGE"] = QTEXTRA_TELEMETRY_PACKAGE
QTEXTRA_TELEMETRY_ORGANIZATION = "illumion"
os.environ["QTEXTRA_TELEMETRY_ORGANIZATION"] = QTEXTRA_TELEMETRY_ORGANIZATION
QTEXTRA_TELEMETRY_PROJECT = "image2image"
os.environ["QTEXTRA_TELEMETRY_PROJECT"] = QTEXTRA_TELEMETRY_PROJECT

# important that this is imported AFTER we setup the environment variables
from qtextra.dialogs.sentry import (  # noqa: E402
    FeedbackDialog,
)
from qtextra.dialogs.sentry import (  # noqa: E402
    ask_opt_in as _ask_opt_in,
)
from qtextra.dialogs.sentry import (  # noqa: E402
    install_error_monitor as _install_error_monitor,
)

IS_PYINSTALLER = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def install_error_monitor() -> None:
    """Initialize the error monitor with sentry.io."""
    from image2image.config import get_app_config

    _install_error_monitor(get_app_config(), pyinstaller=IS_PYINSTALLER)
    logger.debug("Installed sentry error monitor.")


def ask_opt_in(parent):
    """Initialize the error monitor with sentry.io."""
    from image2image.config import get_app_config

    _ask_opt_in(settings=get_app_config(), force=True, parent=parent)


def send_feedback(parent):
    """Send feedback."""
    dlg = FeedbackDialog(parent=parent)
    dlg.exec_()  # type: ignore[attr-defined]
