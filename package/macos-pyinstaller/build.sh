#!/bin/zsh
# This script is used to build the macOS pyinstaller package.
# See various issues:
# https://github.com/RandomByte/tacl-gui/commit/dbf1771051f2f1de90c1558ce073153b0d7d4741
# https://github.com/pyinstaller/pyinstaller/issues/4629
# https://gist.github.com/txoof/0636835d3cc65245c6288b2374799c43


update=false
update_app=false
update_deps=false
update_just_reader=false
update_just_register=false
update_just_app=false
update_pip=false
no_docs=true
help=false
package=false
debug=false

while getopts uadrwjpngzh opt; do
  case $opt in
    u) update=true;;
    a) update_app=true;;
    d) update_deps=true;;
    r) update_just_reader=true;;
    w) update_just_register=true;;
    j) update_just_app=true;;
    p) update_pip=true;;
    n) no_docs=true;;
    g) debug=true;;
    z) package=true;;
    h) help=true;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1;;
  esac
done

shift "$(( OPTIND - 1 ))"

if $help
then
  echo "Usage: ./build.sh [-update] [-update_app] [-update_deps] [-update_just_reader] [-update_just_register] [-update_just_app] [-update_pip] [-no_docs] [-uv] [-package] [-help]"
  echo "  -u / update ($update): update the i2i package before building"
  echo "  -a / update_app ($update_app): update the i2i and i2i-io packages before building"
  echo "  -d / update_deps ($update_deps): update dependencies before building"
  echo "  -r / update_just_reader ($update_just_reader): update the i2i-io package to a specific commit before building"
  echo "  -w / update_just_register ($update_just_register): update the i2i-register package to a specific commit before building"
  echo "  -j / update_just_app ($update_just_app): update the i2i package to a specific commit before building"
  echo "  -i / update_pip ($update_pip): update the pip packages before building"
  echo "  -n / no_docs ($no_docs): do not build the documentation"
  echo "  -g / debug ($debug): enable debug mode"
  echo "  -z / package ($package): package the application"
  echo "  -h / help: show this help message"
  exit 0
fi

# --- helpers ---

abspath() {
  # portable absolute path resolver (no dependency on realpath)
  /usr/bin/python3 -c 'import os,sys; print(os.path.abspath(os.path.realpath(sys.argv[1])))' "$1"
}

pkg_name_only() {
  # strip version/extras markers for uninstall; e.g. "napari==0.6.6" -> "napari"
  # also handles "<", ">", "=", "!", "~"
  local s="$1"
  echo "${s%%[<>=!~]*}"
}

# activate environment
if [[ $(uname -m) != "arm64" ]]
then
  echo "----------------------------------------------------------"
  echo "Running on Rosetta. Please deactivate rosetta beforehand"
  echo "Run: arch -arm64 /bin/zsh"
  echo "----------------------------------------------------------"
  exit 1
fi

if $debug
then
  echo "Debug mode enabled"
  export PYINSTALLER_DEBUG="all"
else
  export PYINSTALLER_DEBUG="imports"
fi


start_dir=$PWD
echo "Current directory: " $start_dir
app_dir=$(realpath $start_dir/../../)
echo "GitHub directory: " $app_dir
github_dir=$(realpath $start_dir/../../../)
echo "GitHub directory: " $github_dir
source_path=$(realpath $start_dir/venv_package/bin/activate)
echo "Source path: " $source_path

# activate appropriate environment
source $source_path

# extract version information
python_ver=$(python -V) 2>&1
echo "Python version "$python_ver
echo "Python path: " $(which python)

declare -a local_install=()
declare -a pip_install=()
declare -a before_install=()
declare -a after_install=()

always_install+=("pip")
always_install+=("uv")

if $update
then
    update_app=true
    update_deps=true
    update_pip=true
fi

if $update_app
then
  update_just_app=true
  update_just_reader=true
  update_just_register=true
fi

update_qt=false
if $update_deps
then
  update_qt=true
fi

# inform user what's happening
echo "Building macOS pyinstaller package..."
echo "update(-u): $update"
echo "update_app(-a): $update_app"
echo "update_deps(-d): $update_deps"
echo "update_just_reader(-r): $update_just_reader"
echo "update_just_app(-j): $update_just_app"
echo "update_just_register(-w): $update_just_register"
echo "update_pip(-i): $update_pip"
echo "update_qtextra: $update_qt"
echo "no_docs(-n): $no_docs"
echo "package(-z): $package"
echo "help(-h): $help"

if $update_just_app
then
    local_install+=("image2image")
fi

if $update_just_register
then
    local_install+=("image2image-reg")
fi

# actually install the packages
if $update_just_reader
then
    local_install+=("image2image-io")
fi


# always install latest version of koyo
local_install+=("koyo")

if $update_pip
then
    pip_install+=("napari==0.6.6")
    pip_install+=("pydantic>=2")
    pip_install+=("PyQt6>=6.9.1")

    after_install+=("tifffile<2025.5.10")
    after_install+=("pandas>2,<3")
    after_install+=("zarr>2,<3")
    after_install+=("numpy>2")
    after_install+=("numba>0.60")
fi

# always install latest version of pyinstaller to ensure we have the latest fixes for Apple Silicon
pip_install+=("pyinstaller")

# before installs
if (( ${#before_install[@]} > 0 )); then
  for spec in "${before_install[@]}"; do
    name="$(pkg_name_only "$spec")"
    echo "Installing pip package: $spec (uninstall name: $name)"
    uv pip uninstall "$name" || true
    uv pip install -U "$spec"
    echo "Installed pip package: $spec"
  done
fi

# pip installs
for spec in "${pip_install[@]}"; do
  name="$(pkg_name_only "$spec")"
  echo "Installing pip package: $spec (uninstall name: $name)"
  uv pip uninstall "$name" || true
  uv pip install -U "$spec"
  echo "Installed pip package: $spec"
done

# local editable installs
if (( ${#local_install[@]} > 0 )); then
  for pkg in "${local_install[@]}"; do
    echo "Installing local package: $pkg"
    cd "$(abspath "$github_dir/$pkg")" || exit 1
    uv pip install -U "$pkg @ ." --force-reinstall
    echo "Installed local package: $pkg"
    cd "$start_dir"
  done
fi

# iterate over the list
for pkg in "${always_install[@]}"
do
    echo "Installing package: " $pkg
    uv pip uninstall $pkg
    uv pip install -U $pkg
    echo "Installed package: " $pkg
done

# install qtextra
if $update_qt
then
    echo "Installing qtextra..."
    cd $(realpath $github_dir/qtextra) || exit 1
    uv pip uninstall qtextra
    uv pip install -U ".[console,sentry]"
    echo "Installed qtextra."
    cd $start_dir

    echo "Installing qtextraplot..."
    cd $(realpath $github_dir/qtextraplot) || exit 1
    uv pip uninstall qtextraplot
    uv pip install -U ".[2d]"
    echo "Installed qtextraplot."
    cd $start_dir
fi

# after pip installs
if (( ${#after_install[@]} > 0 )); then
  for spec in "${after_install[@]}"; do
    name="$(pkg_name_only "$spec")"
    echo "Installing pip package: $spec (uninstall name: $name)"
    uv pip uninstall "$name" || true
    uv pip install -U "$spec"
    echo "Installed pip package: $spec"
  done
fi

# Get path
filename="image2image.spec"

echo "### Printing versions ###"
print_script_path=$(realpath $app_dir/scripts/print_versions.py)
python $print_script_path
echo "### End of versions ###"

# Build bundle
echo "Building bundle... filename=$filename"
pyinstaller --noconfirm --clean $filename

if $package
then
  echo "Packaging application..."
  sh ./package.sh
  echo "Packaging complete."
fi
