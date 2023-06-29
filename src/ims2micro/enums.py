"""Enums."""
from enum import auto

from napari.utils.misc import StringEnum

ALLOWED_EXPORT_FORMATS = (
    "Any transformation (*.json *.toml);; "
    "Transformation (*.json);; "
    "Transformation (*.toml);;"
    "MATLAB fusion format (*.xml);;"
)
ALLOWED_FORMATS = (
    "Any imaging (*.tsf *.tdf *.imzML *.metadata.h5 peaks_*.h5 *.npy *.czi *.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi"
    " *.jpg *.jpeg *.png);; "
    "Bruker (*.tsf *.tdf);; "
    "imzML (*.imzML);; "
    "ionglow (*.metadata.h5, peaks_*.h5);;"
    "Numpy (*.npy);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
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


class ViewType(StringEnum):
    """View type."""

    RANDOM = auto()
    OVERLAY = auto()


VIEW_TYPE_TRANSLATIONS = {
    ViewType.RANDOM: "Random",
    ViewType.OVERLAY: "Overlay",
}
