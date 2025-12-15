"""System utilities."""

import sys
from functools import lru_cache


def get_launch_command() -> str:
    """Get the information how the program was launched.

    Returns
    -------
    str
        The command used to launch the program.
    """
    return " ".join(sys.argv)


@lru_cache(maxsize=2)
def get_system_info(as_html=False) -> None:
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """
    import platform
    import sys

    from napari.utils.info import _sys_name
    from qtextra.helpers import hyper

    sys_version = sys.version.replace("\n", " ")
    text = f"<b>Python</b>: {sys_version}<br>"
    text += f"<b>Platform</b>: {platform.platform()}<br>"
    __sys_name = _sys_name()
    if __sys_name:
        text += f"<b>System</b>: {__sys_name}<br>"

    text += "<br><b>image2image Packages</b>"
    text += "<br>--------------------<br>"
    modules = (
        ("image2image", "image2image"),
        ("image2image_io", "image2image-io"),
        ("image2image_reg", "image2image-reg"),
    )
    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:  # noqa: BLE001
            text += f"<b>{name}</b>: Import failed ({e})<br>"

    text += "<br><b>Packages</b>"
    text += "<br>--------------------<br>"
    modules = (
        ("qtextra", "qtextra"),
        ("qtextraplot", "qtextraplot"),
        ("imzy", "imzy"),
        ("koyo", "koyo"),
    )
    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:  # noqa: BLE001
            text += f"<b>{name}</b>: Import failed ({e})<br>"

    text += "<br><b>Dependencies</b>"
    text += "<br>--------------------<br>"
    modules = (
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("dask", "Dask"),
        ("valis", "valis-wsi"),
        ("itk", "ITK"),
        ("SimpleITK", "SimpleITK"),
        ("qtpy", "QtPy"),
        ("qtawesome", "QtAwesome"),
        ("superqt", "superqt"),
        ("napari", "Napari"),
        ("vispy", "VisPy"),
    )
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:  # noqa: BLE001
            text += f"<b>{name}</b>: Import failed ({e})<br>"

    try:
        from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

        if API_NAME in ["PySide2", "PySide6"]:
            API_VERSION = PYSIDE_VERSION
        elif API_NAME in ["PyQt5", "PyQt6"]:
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = "N/A"

        text += f"<b>Qt</b>: {QtCore.__version__}<br>"
        text += f"<b>{API_NAME}</b>: {API_VERSION}<br>"

    except Exception as e:  # noqa: BLE001
        text += f"<b>Qt</b>: Import failed ({e})<br>"

    text += "<br><b>OpenGL:</b><br>"
    if loaded.get("vispy", False):
        from napari._vispy.utils.gl import get_max_texture_sizes

        sys_info_text = (
            "<br>".join([loaded["vispy"].sys_info().split("\n")[index] for index in [-4, -3]])
            .replace("'", "")
            .replace("<br>", "<br>  - ")
        )
        text += f"  - {sys_info_text}<br>"
        _, max_3d_texture_size = get_max_texture_sizes()
        text += f"  - GL_MAX_3D_TEXTURE_SIZE: {max_3d_texture_size}<br>"
    else:
        text += "  - failed to load vispy"

    text += "<br><b>Screens:</b><br>"
    try:
        from qtpy.QtGui import QGuiApplication

        screen_list = QGuiApplication.screens()
        for i, screen in enumerate(screen_list, start=1):
            text += (
                f"  - screen {i}: resolution {screen.geometry().width()}x{screen.geometry().height()}, "
                f"scale {screen.devicePixelRatio()}<br>"
            )
    except Exception as e:  # noqa BLE001
        text += f"  - failed to load screen information {e}"

    # Add paths
    from image2image.utils._appdirs import USER_CONFIG_DIR, USER_LOG_DIR

    text += "<br><b>Settings path:</b><br>"
    text += f"  - {hyper(USER_CONFIG_DIR) if as_html else USER_LOG_DIR}"
    text += "<br><b>Logs path:</b><br>"
    text += f"  - {hyper(USER_LOG_DIR) if as_html else USER_LOG_DIR}"

    text += "<br><b>Launch command:</b><br>"
    text += f"  - {get_launch_command()}<br>"

    if not as_html:
        text = text.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
    return text


citation_text = "image2image contributors (2023-2025). Van de Plas lab, Delft University of Technology."
title_text = "image2image - registration, visualisation and editing of multiple image types"
