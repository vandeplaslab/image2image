# Create icon.ico and icon.icns files
# Requires imagemagick to be installed
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $script_dir

convert -background transparent -resize 2000x2000 $script_dir/icon.svg $script_dir/icon.png
echo "Generated png icon"
convert -background transparent $script_dir/icon.svg -define icon:auto-resize=16,24,32,48,64,72,96,128,256 $script_dir/icon.ico
echo "Generated ico icon"
sh $script_dir/svg_to_icns.sh $script_dir/icon.svg $script_dir/icon.icns
echo "Generated icns icon"
mv icon.ico ../src/ims2micro/assets/icon.ico
mv icon.icns ../src/ims2micro/assets/icon.icns
mv icon.png ../src/ims2micro/assets/icon.png
cp icon.svg ../src/ims2micro/assets/icon.svg