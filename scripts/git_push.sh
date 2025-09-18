#!/bin/zsh
# Usage: ./git_pull.sh

set -e

script_path=$(realpath "$0")
start_dir=$(pwd)
github_dir=$(realpath "$script_path/../../../")

packages=(
    koyo imzy
    image2image image2image-io image2image-reg
    qtextra qtextraplot
)

for package_name in "${packages[@]}"; do
    echo "Pushing: $package_name"
    to_dir="$github_dir/$package_name"
    if [ -d "$to_dir/.git" ]; then
        cd "$to_dir"
        git commit -am "Auto-commit"
        git push
        echo "Pushed: $package_name"
    else
        echo "Directory $to_dir does not exist or is not a git repository."
    fi
done

cd "$start_dir"