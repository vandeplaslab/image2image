import os.path
from PyInstaller.utils.hooks import collect_data_files
import debugpy._vendored

hiddenimports = [
    "glasbey",
    "freetype",
    "six",
    "pkg_resources",
]
datas = (
    []
    + collect_data_files("freetype")
    + collect_data_files("glasbey")
    + collect_data_files("ome_types")
    + collect_data_files("distributed")
    + collect_data_files("xmlschema")
    + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")]
)
