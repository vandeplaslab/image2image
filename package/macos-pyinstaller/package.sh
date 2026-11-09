#!/bin/zsh
# Create a versioned DMG without moving the original app bundle.
set -euo pipefail

if [[ "${1:-}" == -h ]]; then
  echo 'Usage: package.sh (set MACOS_SIGN_IDENTITY to sign the app before packaging)'
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1" >&2
  exit 1
fi
script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
python="$script_dir/venv_package/bin/python"
app_bundle="$script_dir/dist/image2image.app"
if [[ ! -d "$app_bundle" || ! -x "$python" ]]; then
  echo 'Build the application and packaging environment with build.sh first.' >&2
  exit 1
fi
version="$("$python" -c 'import image2image; print(image2image.__version__)')"
output="$script_dir/dist/image2image-v${version}-macosx_arm64.dmg"
work_dir="$(mktemp -d "$script_dir/dist/.dmg-XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
mkdir "$work_dir/staging"
ditto "$app_bundle" "$work_dir/staging/image2image.app"
ln -s /Applications "$work_dir/staging/Applications"

if [[ -n "${MACOS_SIGN_IDENTITY:-}" ]]; then
  codesign --deep --force --options=runtime \
    --entitlements "$script_dir/entitlements.plist" \
    --sign "$MACOS_SIGN_IDENTITY" --timestamp "$work_dir/staging/image2image.app"
  codesign --verify --deep --strict "$work_dir/staging/image2image.app"
else
  echo 'MACOS_SIGN_IDENTITY is unset; creating an unsigned local DMG.'
fi
hdiutil create -volname image2image -srcfolder "$work_dir/staging" \
  -format UDZO "$work_dir/tmp.dmg"
hdiutil convert "$work_dir/tmp.dmg" -format ULFO -ov -o "$output"
echo "Created DMG: $output"
