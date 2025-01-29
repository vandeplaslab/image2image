from functools import lru_cache


@lru_cache(maxsize=2)
def get_system_info(as_html=False):
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """
    import platform
    import sys

    import image2image_io

    try:
        import image2image_reg
    except ImportError:
        image2image_reg = None
    try:
        import valis
    except ImportError:
        valis = None
    from napari.utils.info import _sys_name
    from qtextra.helpers import hyper

    import image2image
    from image2image.utils._appdirs import USER_CONFIG_DIR, USER_LOG_DIR

    sys_version = sys.version.replace("\n", " ")
    text = f"<b>Python</b>: {sys_version}<br>"
    text += f"<b>Platform</b>: {platform.platform()}<br>"
    __sys_name = _sys_name()
    if __sys_name:
        text += f"<b>System</b>: {__sys_name}<br>"

    text += f"<br><b>image2image</b>: {image2image.__version__}<br>"
    text += f"<b>image2image-io</b>: {image2image_io.__version__}<br>"
    if image2image_reg:
        text += f"<b>image2image-reg</b>: {image2image_reg.__version__}<br><br>"
    else:
        text += "<b>image2image-reg</b>: not installed<br><br>"
    if valis and hasattr(valis, "__version__"):
        text += f"<b>valis-wsi</b>: {valis.__version__}<br><br>"
    else:
        text += "<b>valis-wsi</b>: not installed<br><br>"

    try:
        from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

        if API_NAME == "PySide2":
            API_VERSION = PYSIDE_VERSION
        elif API_NAME == "PyQt5":
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = "N/A"

        text += f"<b>Qt</b>: {QtCore.__version__}<br>"
        text += f"<b>{API_NAME}</b>: {API_VERSION}<br>"

    except Exception as e:  # noqa: BLE001
        text += f"<b>Qt</b>: Import failed ({e})<br>"

    modules = (
        ("qtpy", "QtPy"),
        ("qtawesome", "QtAwesome"),
        ("qtextra", "qtextra"),
        ("napari", "Napari"),
        ("superqt", "superqt"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("dask", "Dask"),
        ("vispy", "VisPy"),
        ("imzy", "imzy"),
    )
    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:  # noqa: BLE001
            text += f"<b>{name}</b>: Import failed ({e})<br>"

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

    text += "<br><b>Settings path:</b><br>"
    text += f"  - {hyper(USER_CONFIG_DIR) if as_html else USER_LOG_DIR}"

    text += "<br><b>Logs path:</b><br>"
    text += f"  - {hyper(USER_LOG_DIR) if as_html else USER_LOG_DIR}"

    if not as_html:
        text = text.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
    return text


citation_text = "image2image contributors (2023). Van de Plas lab, Delft University of Technology."
title_text = "image2image - registration, visualisation and editing of multiple image types"
