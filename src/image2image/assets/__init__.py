"""Assets."""
from pathlib import Path

from qtextra.assets import update_icon_mapping, update_icons, update_styles
from qtextra.config.theme import THEMES, Theme
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
        "save": "fa5s.save",
        "screenshot": "mdi.camera-outline",
        "ipython": "mdi.console",
        "ruler": "fa5s.ruler-horizontal",
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
        "sync": "fa5s.sync-alt",
        "crop": "fa5s.crop-alt",
        "export": "mdi6.content-save-check-outline",
    }
)

THEME = {
    "name": "image2image",
    "type": "light",
    "background": "rgb(247, 247, 247)",
    "foreground": "rgb(245, 224, 218)",
    "primary": "rgb(156, 151, 148)",
    "secondary": "rgb(134, 130, 135)",
    "highlight": "rgb(198, 207, 126)",
    "text": "rgb(13, 13, 13)",
    "icon": "rgb(81, 86, 105)",
    "warning": "rgb(255, 105, 60)",
    "error": "rgb(255, 18, 31)",
    "success": "rgb(12, 237, 91)",
    "progress": "rgb(255, 175, 77)",
    "current": "rgb(12, 237, 91)",
    "syntax_style": "default",
    "console": "rgb(255, 255, 255)",
    "canvas": "rgb(255, 255, 255)",
    "standout": "rgb(255, 252, 0)",
    "font_size": "14px",
    "header_size": "18px",
}

theme = Theme(**THEME)
THEMES.add_theme(theme.name, theme, register=True)
THEMES.register_themes()
# THEMES.theme = "image2image"
THEMES[THEMES.theme].font_size = "9pt"
