import debugpy._vendored
from PyInstaller.utils.hooks import collect_data_files

datas = (
    collect_data_files("napari")
    + collect_data_files("xmlschema")
    + collect_data_files("ome_types")
    + collect_data_files("distributed")
    + collect_data_files("freetype")
    + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")]
)
hiddenimports = ["freetype", "six", "pkg_resources"]
