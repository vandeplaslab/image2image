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
uv=false
package=false
debug=false

while getopts uadrwjpnvgzh opt; do
  case $opt in
    u) update=true;;
    a) update_app=true;;
    d) update_deps=true;;
    r) update_just_reader=true;;
    w) update_just_register=true;;
    j) update_just_app=true;;
    p) update_pip=true;;
    n) no_docs=true;;
    v) uv=true;;
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
  echo "  -u / update: update the i2i package before building"
  echo "  -a / update_app: update the i2i and i2i-io packages before building"
  echo "  -d / update_deps: update dependencies before building"
  echo "  -r / update_just_reader: update the i2i-io package to a specific commit before building"
  echo "  -w / update_just_register: update the i2i-register package to a specific commit before building"
  echo "  -j / update_just_app: update the i2i package to a specific commit before building"
  echo "  -i / update_pip: update the pip packages before building"
  echo "  -n / no_docs: do not build the documentation"
  echo "  -v / uv: use uv for updates"
  echo "  -g / debug: enable debug mode"
  echo "  -p / package: package the application"
  echo "  -h / help: show this help message"
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

if $debug
then
  echo "Debug mode enabled"
  export PYINSTALLER_DEBUG="all"
else
  export PYINSTALLER_DEBUG="imports"
fi


start_dir=$PWD
echo "Current directory: " $start_dir
github_dir=$(realpath $start_dir/../../../)
echo "GitHub directory: " $github_dir
if $uv
then
#  source_path=$(realpath $start_dir/../../venv_package_uv/bin/activate)
  source_path=$(realpath $start_dir/../../venv_package_uv/bin/activate)
else
  source_path=$(realpath $start_dir/../../venv_package/bin/activate)
fi
echo "Source path: " $source_path

# activate appropriate environment
source $source_path

# extract version information
python_ver=$(python -V) 2>&1
echo "Python version "$python_ver
echo "Python path: " $(which python)

declare -a always_install=()
declare -a pip_install=()
declare -a local_install=()

always_install+=("pip")
if $uv
then
  always_install+=("uv")
fi

if $update
then
    update_app=true
    update_deps=true
    update_pip=true
fi

if $update_app
then
  update_just_reader=true
  update_just_app=true
  update_just_register=true
fi

# inform user what's happening
echo "Building macOS pyinstaller package..."
echo "update: $update"
echo "update_app: $update_app"
echo "update_deps: $update_deps"
echo "update_just_reader: $update_just_reader"
echo "update_just_app: $update_just_app"
echo "update_just_register: $update_just_register"
echo "update_pip: $update_pip"
echo "no_docs: $no_docs"
echo "uv: $uv"
echo "package: $package"
echo "help: $help"


# actually install the packages
if $update_just_app
then
    local_install+=("image2image")
fi

if $update_just_reader
then
    local_install+=("image2image-io")
fi

if $update_just_register
then
    local_install+=("image2image-reg")
fi

qtextra=false
if $update_deps
then
  qtextra=true
fi

# always install latest version of koyo
local_install+=("koyo")

if $update_pip
then
    pip_install+=("napari==0.5.6")
    pip_install+=("pydantic>=2")
    pip_install+=("pandas<2")
    pip_install+=("numpy<2")
    pip_install+=("PyQt6==6.5.3")
    pip_install+=("pyinstaller")
fi

# iterate over the list
for pkg in "${always_install[@]}"
do
    echo "Installing package: " $pkg
    if $uv
    then
      uv pip uninstall $pkg
      uv pip install -U $pkg
    else
      pip install -U $pkg
    fi
    echo "Installed package: " $pkg
done

# install qtextra
if $qtextra
then
    echo "Installing qtextra..."
    cd $(realpath $github_dir/qtextra) || exit 1
    if $uv
    then
      uv pip uninstall qtextra
      uv pip install -U ".[console,sentry]"
    else
      pip install -U ".[console,sentry]"
    fi
    echo "Installed qtextra."
    cd $start_dir

    echo "Installing qtextraplot..."
    cd $(realpath $github_dir/qtextraplot) || exit 1
    if $uv
    then
      uv pip uninstall qtextraplot
      uv pip install -U ".[2d]"
    else
      pip install -U ".[2d]"
    fi
    echo "Installed qtextraplot."
    cd $start_dir
fi

# iterate over the list
for pkg in "${local_install[@]}"
do
    echo "Installing package: " $pkg
    cd $(realpath $github_dir/$pkg) || exit 1
    if $uv
    then
      uv pip uninstall $pkg
      uv pip install -U "$pkg @ ." --refresh
    else
      pip install -U .
    fi
    echo "Installed package: " $pkg
    cd $start_dir
done

# iterate over the list
for pkg in "${pip_install[@]}"
do
    echo "Installing package: " $pkg
    if $uv
    then
      uv pip uninstall $pkg
      uv pip install -U $pkg
    else
      pip install -U $pkg
    fi
    echo "Installed package: " $pkg
done

# Get path
filename="image2image.spec"

echo "### Printing versions ###"
print_script_path=$(realpath $github_dir/scripts/print_versions.py)
python $print_script_path
echo "### End of versions ###"

# Build bundle
echo "Building bundle... filename=$filename"
#if $debug
#then
pyinstaller --noconfirm --clean $filename
#else
#  pyinstaller --noconfirm --clean --noconsole $filename
#fi

if $package
then
  echo "Packaging application..."
  if $uv
  then
    sh ./package.sh -v
  else
    sh ./package.sh
  fi
  echo "Packaging complete."
fi