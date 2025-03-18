"""Constant values."""

UINT8_TIP = (
    "Convert to uint8 to reduce file size with minimal data loss. This will result in change of the"
    " dynamic range of the image to between 0-255."
)
UINT8_WARNING = (
    "While this option reduces the amount of space an image takes on your disk, it can lead to data loss<br>"
    " and should be used with caution. If you are exporting RGB images (e.g. H&E or PAS), then this is not<br>"
    " a concert since that data is already in the 0-255 range. However, if you are exporting images with a<br>"
    " dynamic range greater than 0-255, then you should be careful (e.g. AF, IHC, CODEX)."
)
