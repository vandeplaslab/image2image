#!/bin/sh
# Get icon location
icon_path=$(realpath $PWD/../../src/image2image/assets/icon.icns)
echo "Icon path: " $icon_path

# Create a folder (named dmg) to prepare our DMG in (if it doesn't already exist).
mkdir -p dist/dmg
# Empty the dmg folder.
rm -r dist/dmg/*
# Copy the app bundle to the dmg folder.
cp -r "dist/image2image.app" dist/dmg
# If the DMG already exists, delete it.
test -f "dist/image2image.dmg" && rm "dist/image2image.dmg"
create-dmg \
  --volname "image2image" \
  --volicon $icon_path \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --icon "image2image.app" 175 120 \
  --hide-extension "image2image.app" \
  --app-drop-link 425 120 \
  "dist/image2image.dmg" \
  "dist/dmg/"