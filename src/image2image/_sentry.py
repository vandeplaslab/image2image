"""Sentry widget."""
import os
from qtextra.dialogs.sentry import install_error_monitor as _install_error_monitor


# setup environment variables
SENTRY_DSN = "https://5473ad03e64c4a2988b9ec143f6181c5@o1133843.ingest.sentry.io/4505464740708352"
os.environ["QTEXTRA_TELEMETRY_VERSION"] = SENTRY_DSN
QTEXTRA_TELEMETRY_SHOW_HOSTNAME = "0"
os.environ["QTEXTRA_TELEMETRY_SHOW_HOSTNAME"] = QTEXTRA_TELEMETRY_SHOW_HOSTNAME
QTEXTRA_TELEMETRY_SHOW_LOCALS = "1"
os.environ["QTEXTRA_TELEMETRY_SHOW_LOCALS"] = QTEXTRA_TELEMETRY_SHOW_LOCALS
QTEXTRA_TELEMETRY_DEBUG = "0"
os.environ["QTEXTRA_TELEMETRY_DEBUG"] = QTEXTRA_TELEMETRY_DEBUG

def install_error_monitor():
    """Initialize the error monitor with sentry.io."""
    from image2image.config import CONFIG

    _install_error_monitor(CONFIG)