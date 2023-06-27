# Create icon.ico and icon.icns files
# Requires imagemagick to be installed
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $script_dir

convert -background transparent $script_dir/icon.svg -define icon:auto-resize=16,24,32,48,64,72,96,128,256 $script_dir/icon.ico
convert -background transparent $script_dir/icon.svg -define icon:auto-resize=16,24,32,48,64,72,96,128,256 $script_dir/icon.icns
cp icon.ico ../src/ims2micro/assets/icon.ico
cp icon.icns ../src/ims2micro/assets/icon.icns
cp icon.svg ../src/ims2micro/assets/icon.svg