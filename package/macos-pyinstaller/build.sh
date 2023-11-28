#!/bin/zsh
# This script is used to build the macOS pyinstaller package.
# See various issues:
# https://github.com/RandomByte/tacl-gui/commit/dbf1771051f2f1de90c1558ce073153b0d7d4741
# https://github.com/pyinstaller/pyinstaller/issues/4629
# https://gist.github.com/txoof/0636835d3cc65245c6288b2374799c43


update=false
update_i2i=false
debug=false
no_docs=true
run=false
help=false

while getopts uidnrh opt; do
  case $opt in
    u) update=true;;
    i) update_i2i=$OPTARG;;
    d) debug=true;;
    n) no_docs=true;;
    r) run=true;;
    h) help=true;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1;;
  esac
done

echo "Building macOS pyinstaller package..."
echo "update: $update"
echo "update_i2i: $update_i2i"
echo "debug: $debug"
echo "no_docs: $no_docs"
echo "run: $run"
echo "help: $help"


shift "$(( OPTIND - 1 ))"

if $help
then
  echo "Usage: ./build.sh [-update] [-update_i2i] [-debug] [-no_docs] [-run] [-help]"
  echo "  -update: update the i2i package before building"
  echo "  -update_i2i: update the i2i package to a specific commit before building"
  echo "  -debug: build the package in debug mode"
  echo "  -no_docs: do not build the documentation"
  echo "  -run: run the package after building"
  echo "  -help: show this help message"
  exit 0
fi

# activate environment
if [[ $(uname -m) != "arm64" ]]
then
  echo "----------------------------------------------------------"
  echo "Running on Rosetta. Please deactivate rosetta beforehand"
  echo "Run: arch -arm64 /bin/zsh"
  echo "----------------------------------------------------------"
  exit 1
fi



start_dir=$PWD
echo "Current directory: " $start_dir
github_dir=$(realpath $start_dir/../../../)
echo "GitHub directory: " $github_dir
source_path=$(realpath $start_dir/../../venv_package/bin/activate)
echo "Source path: " $source_path

# activate appropriate environment
source $source_path

# extract version information
python_ver=$(python -V) 2>&1
echo "Python version "$python_ver
echo "Python path: " $(which python)

if $update
then
    # Re-install qtextra
    echo "Re-installing qtextra..."
    cd $(realpath $github_dir/qtextra) || exit 1
    pip install .
    echo "Reinstalled qtextra"

    # Re-install koyo
    echo "Re-installing koyo..."
    new_dir=$(realpath $github_dir/koyo)
    cd $new_dir || exit 1
    pip install .
    echo "Reinstalled koyo"

    # Re-install image2image
    echo "Re-installing image2image..."
    new_dir=$(realpath $github_dir/image2image)
    cd $new_dir || exit 1
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image"

    # Re-install image2image
    echo "Re-installing image2image-reader..."
    new_dir=$(realpath $github_dir/image2image-reader)
    cd $new_dir || exit 1
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image-reader"

    # Re-install napari (latest)
    echo "Re-installing napari..."
    pip install -U napari==0.4.18
    echo "Reinstalled napari"

    # Re-install PySide6
    echo "Re-installing PyQt6..."
    pip install -U PyQt6
    echo "Reinstalled PyQt6"

    # Re-install pyinstaller
    echo "Re-installing pyinstaller..."
    pip install -U pyinstaller
    echo "Reinstalled pyinstaller"
fi

if $update_i2i
then
    # Re-install image2image
    echo "Re-installing image2image-reader..."
    new_dir=$(realpath $github_dir/image2image-reader)
    cd $new_dir || exit 1
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image-reader"

    # Re-install image2image
    echo "Re-installing image2image..."
    new_dir=$(realpath $github_dir/image2image)
    cd $new_dir || exit 1
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image"
fi

# Get path
filename="image2image_split.spec"


# Build bundle
echo "Building bundle... debug=$debug; filename=$filename"
if $debug
then
    pyinstaller --windowed --noconfirm --clean --codesign-identity vandeplaslab --debug=all $filename
else
    pyinstaller --noconfirm --clean $filename
fi