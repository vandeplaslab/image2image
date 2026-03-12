#!/usr/bin/env bash
set -euo pipefail

INPUT_SVG="${1:-}"
APP_NAME="${2:-}"

if [[ -z "$INPUT_SVG" ]]; then
    echo "Usage: $0 input.svg [output_name]"
    exit 1
fi

if [[ ! -f "$INPUT_SVG" ]]; then
    echo "Error: file not found: $INPUT_SVG"
    exit 1
fi

if [[ -z "$APP_NAME" ]]; then
    APP_NAME="$(basename "$INPUT_SVG" .svg)"
fi

OUT_DIR="output"
ICONSET_DIR="${OUT_DIR}/${APP_NAME}.iconset"
mkdir -p "$OUT_DIR"
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

if command -v inkscape >/dev/null 2>&1; then
    RENDERER="inkscape"
elif command -v rsvg-convert >/dev/null 2>&1; then
    RENDERER="rsvg"
else
    echo "Error: need either 'inkscape' or 'rsvg-convert' installed."
    exit 1
fi

if ! command -v magick >/dev/null 2>&1; then
    echo "Error: ImageMagick 'magick' is required."
    exit 1
fi

HAS_ICONUTIL=0
if command -v iconutil >/dev/null 2>&1; then
    HAS_ICONUTIL=1
fi

render_svg() {
    local size="$1"
    local output="$2"

    if [[ "$RENDERER" == "inkscape" ]]; then
        inkscape "$INPUT_SVG" \
            --export-type=png \
            --export-filename="$output" \
            -w "$size" -h "$size" \
            >/dev/null 2>&1
    else
        rsvg-convert -w "$size" -h "$size" "$INPUT_SVG" -o "$output"
    fi
}

echo "Using renderer: $RENDERER"

MAIN_PNG="${OUT_DIR}/${APP_NAME}.png"
render_svg 1024 "$MAIN_PNG"
echo "Created: $MAIN_PNG"

ICO_SIZES=(16 24 32 48 64 128 256)
ICO_FILES=()

for size in "${ICO_SIZES[@]}"; do
    file="${OUT_DIR}/${APP_NAME}-${size}.png"
    render_svg "$size" "$file"
    ICO_FILES+=("$file")
done

ICO_FILE="${OUT_DIR}/${APP_NAME}.ico"
magick "${ICO_FILES[@]}" "$ICO_FILE"
echo "Created: $ICO_FILE"

ICNS_FILES=(
    "icon_16x16.png:16"
    "icon_16x16@2x.png:32"
    "icon_32x32.png:32"
    "icon_32x32@2x.png:64"
    "icon_128x128.png:128"
    "icon_128x128@2x.png:256"
    "icon_256x256.png:256"
    "icon_256x256@2x.png:512"
    "icon_512x512.png:512"
    "icon_512x512@2x.png:1024"
)

for item in "${ICNS_FILES[@]}"; do
    filename="${item%%:*}"
    size="${item##*:}"
    render_svg "$size" "${ICONSET_DIR}/${filename}"
done

if [[ "$HAS_ICONUTIL" -eq 1 ]]; then
    ICNS_FILE="${OUT_DIR}/${APP_NAME}.icns"
    iconutil -c icns "$ICONSET_DIR" -o "$ICNS_FILE"
    echo "Created: $ICNS_FILE"
else
    echo "Skipping .icns creation: 'iconutil' not found."
fi

echo "Done."