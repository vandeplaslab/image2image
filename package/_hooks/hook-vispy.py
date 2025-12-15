# ------------------------------------------------------------------
# Copyright (c) 2020 PyInstaller Development Team.
#
# This file is distributed under the terms of the GNU General Public
# License (version 2.0 or later).
#
# The full license is available in LICENSE.GPL.txt, distributed with
# this software.
#
# SPDX-License-Identifier: GPL-2.0-or-later
# ------------------------------------------------------------------

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from qtpy import PYQT5, PYQT6, PYSIDE2, PYSIDE6


def filter_vispy(name: str) -> bool:
    """Filter which hidden imports should be included."""
    if "tests" in name or "docs" in name:
        return False
    # Handled separately below
    return "backends" not in name


datas = collect_data_files("vispy")
hiddenimports = collect_submodules("vispy", filter=filter_vispy)

if PYQT5:
    hiddenimports += [
        "vispy.app.backends._pyqt5",
    ]
elif PYQT6:
    hiddenimports += [
        "vispy.app.backends._pyqt6",
    ]
elif PYSIDE2:
    hiddenimports += [
        "vispy.app.backends._pyside2",
    ]
elif PYSIDE6:
    hiddenimports += [
        "vispy.app.backends._pyside6",
    ]
