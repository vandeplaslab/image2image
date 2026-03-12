# Create icon.ico and icon.icns files
# Requires imagemagick to be installed
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $script_dir

echo "Converting icon.svg to png, ico, and icns formats..."
sh $script_dir/convert_svg.sh $script_dir/icon.svg "icon"
# Move images to the appropriate location
mv output/icon.png ../src/image2image/assets/icon.png
mv output/icon.ico ../src/image2image/assets/icon.ico
mv output/icon.icns ../src/image2image/assets/icon.icns
cp $script_dir/icon.svg ../src/image2image/assets/icon.svg
# cleanup
rm -r output