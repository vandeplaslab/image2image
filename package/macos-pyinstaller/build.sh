#!/bin/zsh
# This script is used to build the macOS pyinstaller package.
# See various issues:
# https://github.com/RandomByte/tacl-gui/commit/dbf1771051f2f1de90c1558ce073153b0d7d4741
# https://github.com/pyinstaller/pyinstaller/issues/4629
# https://gist.github.com/txoof/0636835d3cc65245c6288b2374799c43


update=false
update_app=false
just_app=false
just_reader=false
no_docs=true
execute=false
help=false

while getopts uajrneh opt; do
  case $opt in
    u) update=true;;
    a) update_app=true;;
    j) just_app=true;;
    r) just_reader=true;;
    n) no_docs=true;;
    e) execute=true;;
    h) help=true;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1;;
  esac
done

echo "Building macOS pyinstaller package..."
echo "update: $update"
echo "update_app: $update_app"
echo "just_app: $just_app"
echo "no_docs: $no_docs"
echo "execute: $execute"
echo "help: $help"


shift "$(( OPTIND - 1 ))"

if $help
then
  echo "Usage: ./build.sh [-update] [-update_app] [-no_docs] [-execute] [-help]"
  echo "  -u: update the i2i package before building"
  echo "  -a: update the i2i package to a specific commit before building"
  echo "  -j: update the image2image package only"
  echo "  -r: update the image2image-io package only"
  echo "  -n: do not build the documentation"
  echo "  -e: execute the package after building"
  echo "  -h: show this help message"
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

declare -a local_install=()
declare -a to_install=()

if $update
then
    local_install+=("qtextra")
    local_install+=("koyo")
    local_install+=("image2image")
    local_install+=("image2image-io")

    to_install+=("napari==0.4.18")
    to_install+=("PyQt6==6.5.3")
    to_install+=("pyinstaller")
fi

if $update_app
then
    local_install+=("image2image-io")
    local_install+=("image2image")
fi

if $just_reader
then
    local_install+=("image2image-io")
fi

if $just_app
then
    local_install+=("image2image")
fi

# iterate over the list
for pkg in "${local_install[@]}"
do
    echo "Installing package: " $pkg
    cd $(realpath $github_dir/$pkg) || exit 1
    pip install -U .
    echo "Installed package: " $pkg
    cd $start_dir
done

# iterate over the list
for pkg in "${to_install[@]}"
do
    echo "Installing package: " $pkg
    pip install -U $pkg
    echo "Installed package: " $pkg
done

# Get path
filename="image2image_split.spec"


# Build bundle
echo "Building bundle... filename=$filename"
pyinstaller --noconfirm --clean $filename