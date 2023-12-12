"""Enums."""
from enum import auto

from image2image_io.enums import ViewType
from napari.utils.misc import StringEnum

ALLOWED_IMPORT_REGISTER_FORMATS = (
    "Any transformation (*.i2r.json *.i2r.toml *.toml);; "
    "Transformation (*.i2r.json);; "
    "Transformation (*.i2r.toml);;"
    "imsmicrolink transformation (*.toml);;"
)
ALLOWED_EXPORT_REGISTER_FORMATS = (
    "Any transformation (*.json *.toml);; "
    "Transformation (*.i2r.json);; "
    "Transformation (*.i2r.toml);;"
    "MATLAB fusion format (*.xml);;"
)
ALLOWED_VIEWER_FORMATS = (
    "Any projects (*.i2v.json *.i2v.toml);; " "JSON Project (*.i2v.json);; " "TOML Project (*.i2v.toml);;"
)
ALLOWED_CROP_FORMATS = (
    "Any projects (*.i2c.json *.i2c.toml);; " "JSON Project (*.i2c.json);; " "TOML Project (*.i2c.toml);;"
)
ALLOWED_WSIPREP_FORMATS = (
    "Any projects (*.i2wsiprep.json *.i2wsiprep.toml);; "
    "JSON Project (*.i2wsiprep.json);; "
    "TOML Project (*.i2wsiprep.toml);;"
)
ALLOWED_IMAGE_FORMATS = (
    "Any imaging (*.d *.tsf *.tdf *.imzML  *.ibd *.data *.metadata.h5 peaks_*.h5 *.npy *.czi *.ome.tiff *.tiff *.scn"
    " *.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi *.jpg *.jpeg *.png);; "
    "Bruker (*.d *.tsf *.tdf);; "
    "imzML (*.imzML *.ibd);; "
    "ionglow (*.data *.metadata.h5 peaks_*.h5);;"
    "Numpy (*.npy);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
)
ALLOWED_IMAGE_FORMATS_WITH_GEOJSON = (
    "Any imaging (*.d *.tsf *.tdf *.imzML  *.ibd *.data *.metadata.h5 peaks_*.h5 *.npy *.czi *.ome.tiff *.tiff *.scn"
    "*.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi *.jpg *.jpeg *.png *.geojson *.json);; "
    "Bruker (*.d *.tsf *.tdf);; "
    "imzML (*.imzML *.ibd);; "
    "ionglow (*.data *.metadata.h5 peaks_*.h5);;"
    "Numpy (*.npy);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
    "GeoJSON (*.geojson *.json);;"
)
ALLOWED_IMAGE_FORMATS_CZI_ONLY = "Any Zeiss CZI (*.czi);;"
ALLOWED_IMAGE_FORMATS_TIFF_ONLY = (
    "Any OME-TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
)


class ImageTransformation(StringEnum):
    """Image transformation."""

    EUCLIDEAN = auto()
    SIMILARITY = auto()
    PROJECTIVE = auto()
    AFFINE = auto()


TRANSFORMATION_TRANSLATIONS = {
    ImageTransformation.EUCLIDEAN: "Euclidean",
    ImageTransformation.SIMILARITY: "Similarity",
    ImageTransformation.PROJECTIVE: "Projective",
    ImageTransformation.AFFINE: "Affine",
}


class ViewerOrientation(StringEnum):
    """Viewer orientation."""

    VERTICAL = auto()
    HORIZONTAL = auto()


ORIENTATION_TRANSLATIONS = {
    ViewerOrientation.VERTICAL: "Vertical",
    ViewerOrientation.HORIZONTAL: "Horizontal",
}


VIEW_TYPE_TRANSLATIONS = {
    ViewType.RANDOM: "Random",
    ViewType.OVERLAY: "Overlay",
}
