"""Automatically download and unpack new version from Dropbox."""
from __future__ import annotations

import getpass
import typing as ty


def check_if_can_download() -> bool:
    """Check if user can download new version."""
    user = getpass.getuser()
    if user == "lgmigas":
        return True
    elif user.startswith("VMP1"):
        return True
    return False


class DownloadDict(ty.TypedDict):
    """Download dict."""

    filename: str
    version: str
    download_url: str


def get_release_url() -> DownloadDict | None:
    """Get latest release url."""
    from json import loads
    from urllib import request

    if check_if_can_download():
        url = r"https://www.dropbox.com/scl/fi/ohjhe2vi23j5bxre4fhjf/latest.json?rlkey=nfuztwi7p19svcv6zsrlobwcc&dl=1"

        with request.urlopen(url) as response:
            data: str = response.read().decode("utf-8")
        parsed_data: DownloadDict = loads(data)
        return parsed_data
    return None
