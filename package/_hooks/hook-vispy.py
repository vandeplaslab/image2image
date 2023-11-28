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

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs
from qtpy import PYQT5, PYSIDE2, PYSIDE6, PYQT6

datas = collect_data_files("vispy")
hiddenimports = collect_submodules("vispy")

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
