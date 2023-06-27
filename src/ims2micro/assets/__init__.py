"""Assets."""
from pathlib import Path

from qtextra.assets import update_icon_mapping, update_icons, update_styles
from qtextra.config import THEMES
from qtextra.utils.utilities import IS_MAC, get_module_path

HERE = Path(get_module_path("ims2micro.assets", "__init__.py")).parent.resolve()

ICON_SVG = str(HERE / "icon.svg")
ICON_ICO = str(HERE / ("icon.icns" if IS_MAC else "icon.ico"))

icon_path = HERE / "icons"
icon_path.mkdir(exist_ok=True)
update_icons({x.stem: str(x) for x in icon_path.iterdir() if x.suffix == ".svg"})


styles_path = HERE / "stylesheets"
styles_path.mkdir(exist_ok=True)
update_styles({x.stem: str(x) for x in styles_path.iterdir() if x.suffix == ".qss"})


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
    }
)
THEMES.register_themes()
