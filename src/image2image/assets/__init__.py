"""Assets."""
from pathlib import Path

from qtextra.assets import update_icon_mapping, update_icons, update_styles
from qtextra.config import THEMES
from qtextra.utils.utilities import IS_MAC, get_module_path

HERE = Path(get_module_path("image2image.assets", "__init__.py")).parent.resolve()

ICON_SVG = str(HERE / "icon.svg")
ICON_ICO = str(HERE / ("icon.icns" if IS_MAC else "icon.ico"))
ICON_PNG = str(HERE / "icon.png")

ICONS_PATH = HERE / "icons"
ICONS_PATH.mkdir(exist_ok=True)
update_icons({x.stem: str(x) for x in ICONS_PATH.iterdir() if x.suffix == ".svg"})


STYLES_PATH = HERE / "stylesheets"
STYLES_PATH.mkdir(exist_ok=True)
update_styles({x.stem: str(x) for x in STYLES_PATH.iterdir() if x.suffix == ".qss"})


update_icon_mapping(
    {
        "add": "ri.add-circle-fill",
        "remove": "ri.indeterminate-circle-line",
        "remove_single": "mdi.close-circle",
        "remove_multiple": "mdi.close-circle-multiple",
        "remove_all": "fa5s.trash-alt",
        "zoom": "mdi.magnify",
        "close": "fa5s.trash-alt",
        "bring_to_top": "fa5s.angle-double-up",
        "github": "fa5b.github",
        "request": "msc.request-changes",
        "web": "mdi.web",
        "bug": "fa5s.bug",
        "info": "fa5s.info-circle",
        "warning": "fa5s.exclamation-triangle",
        "error": "fa5s.times-circle",
        "critical": "fa5s.times-circle",
        "debug": "ph.megaphone",
        "success": "fa5s.check",
        "lock_closed": "fa5s.lock",
        "lock_open": "fa5s.lock-open",
        "telemetry": "mdi.telegram",
        "feedback": "msc.feedback",
        "viewer": "fa5.images",
        "register": "fa5s.layer-group",
    }
)
THEMES.register_themes()
