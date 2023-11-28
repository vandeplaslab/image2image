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

datas = collect_data_files('itk', include_py_files=True)
hiddenimports = collect_submodules('itk')
binaries = collect_dynamic_libs('itk', search_patterns=['*.dll', '*.dylib', 'lib*.so', "*.pyd"])

# # This hook only works when ITK is pip installed. It
# # does not work when using ITK directly from its build tree.
#
# from PyInstaller.utils.hooks import collect_data_files
#
# hiddenimports = ["new"]
#
# itk_datas = collect_data_files("itk", include_py_files=True, includes=["**/*", "**/*.pyd"])
# datas = [x for x in itk_datas if "__pycache__" not in x[0]]