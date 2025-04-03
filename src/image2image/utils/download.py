"""Automatically download and unpack new version from Dropbox."""

from __future__ import annotations

import getpass

from koyo.release import DownloadDict, LatestVersion, get_target


def check_if_can_download() -> bool:
    """Check if user can download new version."""
    user = getpass.getuser()
    return bool(user == "lgmigas" or user.startswith("VMP1"))


def get_release_url() -> DownloadDict | None:
    """Get latest release url."""
    from json import loads
    from urllib import request

    if check_if_can_download():
        url = r"https://www.dropbox.com/scl/fi/m73sklf9iftqjl3861wox/latest-multi-target.json?rlkey=r7tw3kgfjsgfh01nozn16wmz4&dl=1"  # noqa

        with request.urlopen(url) as response:
            data: str = response.read().decode("utf-8")
        parsed_data: LatestVersion = loads(data)
        target = get_target()
        if target and target in parsed_data and parsed_data[target]:
            return parsed_data[target]
    return None
