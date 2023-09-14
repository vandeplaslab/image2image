#!/bin/zsh
# get github directory
script_path=$(realpath $0)
github_dir=$(realpath $script_path/../../../)
echo "GitHub directory: " $github_dir

echo "Re-installing image2image..."
cd $(realpath $github_dir/image2image) || exit 1
pip install -e .
echo "Reinstalled image2image"

echo "Re-installing qtextra..."
cd $(realpath $github_dir/qtextra) || exit 1
pip install -e .
echo "Reinstalled qtextra"

echo "Re-installing qtreload..."
cd $(realpath $github_dir/qtreload) || exit 1
pip install -e .
echo "Reinstalled qtreload"

echo "Re-installing napari-plot..."
cd $(realpath $github_dir/napari-plot) || exit 1
pip install -e .
echo "Reinstalled napari-plot"

echo "Re-installing koyo..."
cd $(realpath $github_dir/koyo) || exit 1
pip install -e .
echo "Reinstalled koyo"

echo "Re-installing napari"
pip install -U napari==0.4.17
echo "Reinstalled napari"

# go back to where we started
cd $(realpath $github_dir/image2image) || exit 1