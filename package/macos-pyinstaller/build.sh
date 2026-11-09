#!/bin/zsh
# Build the macOS PyInstaller bundle using its dedicated uv environment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
repo_dir="$(cd -- "$script_dir/../.." && pwd -P)"
python="$script_dir/venv_package/bin/python"
install_args=()
no_build=false
package=false
run=false
debug=false

while getopts ':uadrwjpiqngzslh' opt; do
  case "$opt" in
    u) install_args+=(--update);;
    a) install_args+=(--update-app);;
    d) install_args+=(--update-deps);;
    r) install_args+=(--update-just-reader);;
    w) install_args+=(--update-just-register);;
    j) install_args+=(--update-just-app);;
    p|i) install_args+=(--update-pip);;
    q) install_args+=(--update-qt);;
    n) ;; # Legacy no-docs flag: documentation is not bundled.
    g) debug=true;;
    z) package=true;;
    s) no_build=true;;
    l) run=true;;
    h)
      cat <<'HELP'
Usage: build.sh [-uadqirwj] [-g] [-z] [-s] [-l]
  -u  Update all local and registry dependencies
  -a  Update image2image, image2image-io, and image2image-reg
  -d  Update local ecosystem dependencies and native backends
  -q  Update qtextra and qtextraplot
  -i  Update registry dependencies (-p is a legacy alias)
  -r  Update only image2image-io
  -w  Update only image2image-reg
  -j  Update only image2image
  -g  Enable PyInstaller debug output
  -z  Package a DMG (MACOS_SIGN_IDENTITY enables signing)
  -s  Set up/update the environment without building
  -n  Accepted for compatibility; documentation is not bundled
  -h  Show help
  -l  Launch the app after building

Creates venv_package with uv and Python 3.12 when missing.
Set PACKAGING_PYTHON to override the Python version for a new environment.
HELP
      exit 0;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1;;
  esac
done
shift "$((OPTIND - 1))"
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1" >&2
  exit 1
fi

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'macOS packaging requires native arm64. Run: arch -arm64 /bin/zsh' >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo 'uv is required. Install it before running this script.' >&2
  exit 1
fi
if [[ ! -d "$script_dir/venv_package" ]]; then
  uv venv --python "${PACKAGING_PYTHON:-3.12}" "$script_dir/venv_package"
  install_args+=(--bootstrap)
fi
if [[ ! -x "$python" ]]; then
  echo "Invalid environment at $script_dir/venv_package. Rename it and rerun this script." >&2
  exit 1
fi

export QT_API=pyqt6
export PYINSTALLER_DEBUG=0
$debug && export PYINSTALLER_DEBUG=all
"$python" "$repo_dir/package/install_dependencies.py" "${install_args[@]}"
$no_build && exit 0

cd "$script_dir"
echo "Python: $("$python" -V 2>&1) ($python)"
"$python" "$repo_dir/scripts/print_versions.py"
"$python" -m PyInstaller --noconfirm --clean "$script_dir/image2image.spec"
if $package; then
  /bin/zsh "$script_dir/package.sh"
fi
if $run; then
  open "$script_dir/dist/image2image.app"
fi
