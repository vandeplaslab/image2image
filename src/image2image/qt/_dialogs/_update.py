"""Check for latest version on GitHub."""

from __future__ import annotations

from pathlib import Path

from koyo.release import format_version, get_latest_git, is_new_version_available
from koyo.typing import PathLike

from image2image import __version__


def check_version() -> tuple[bool, str]:
    """Check for latest version."""
    return is_new_version_available(current_version=__version__, package="image2image-docs")


def get_update_info() -> tuple[str, str | None, str, PathLike | None]:
    """Get update info."""
    from image2image.utils.download import get_release_url

    data = get_latest_git(package="image2image-docs")
    is_available, _ = is_new_version_available(__version__, package="image2image-docs", data=data)
    download_url, path_to_file, download_info = None, None, ""
    if is_available:
        download_data = get_release_url()
        if download_data:
            download_url = download_data["download_url"]
            path_to_file = Path.home() / "Downloads" / download_data["filename"]
            download_info = (
                f"Download latest version <b>v{download_data['version']}</b> from <a href='{download_url}'>here</a>."
            )

    return format_version(data), download_url, download_info, path_to_file
