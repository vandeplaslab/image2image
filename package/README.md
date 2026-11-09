# Application packaging

image2image uses PyInstaller for local macOS arm64 and Windows x64 builds. Each
platform directory owns its `venv_package`, `build`, and `dist` directories.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

## First build

Run these commands from any directory, adjusting the repository path as needed:

```sh
/path/to/image2image/package/macos-pyinstaller/build.sh -s
/path/to/image2image/package/macos-pyinstaller/build.sh -z
```

```powershell
& C:\path\to\image2image\package\windows-pyinstaller\build.ps1 -no_build
& C:\path\to\image2image\package\windows-pyinstaller\build.ps1 -zip -installer
```

The first invocation creates a uv environment with Python 3.12 and installs all
packaging dependencies. An interrupted installation with missing distributions
is bootstrapped again on the next invocation. Later builds reuse installed
versions and validate native imports and the Qt binding before bundling.
No activation is required, and an active development environment is not used.
Set `PACKAGING_PYTHON` before setup to select a different Python version (>=3.12).
The app's own minimum Python version is unchanged.

Existing Python 3.11 packaging environments must be renamed or removed manually
before the build script can create their replacement. The former Windows
repository-root `venv_package` is no longer used. `scripts/create_win_env.ps1`
forwards to the new Windows setup workflow.

## Local dependencies

Keep these checkouts alongside image2image:

- image2image-io and image2image-reg
- koyo, qtextra, qtextraplot, ims-utils, and yoki5
- imzy and timsMac (distribution name `pytsf`)

All local packages are installed non-editably. The shared installer resolves
these sources together, using their project metadata for registry requirements.
Local source overrides prevent registry wheels from replacing sibling builds and
allow the development ims-utils checkout below imzy's declared version minimum.
Any update needs all listed checkouts to resolve the dependency graph. Registry
updates follow the current checkout metadata and can update dependencies required
by the selected projects.

Install Rust and Cargo before setup or local dependency updates. Windows also
requires Visual Studio C++ build tools and the Windows SDK. Updating imzy/pytsf
bypasses cached wheels and uses `IMZY_BUILD_RUST=1`. Every build requires imports
of `imzy._imzml_rust` and `pytsf` to succeed. Use a local dependency update to
rebuild a missing native backend. Optional annotation backends are not installed.

## Updating and building

| Operation | macOS | Windows |
| --- | --- | --- |
| Update everything | `-u` | `-update` |
| Update all three image2image projects | `-a` | `-update_app` |
| Update local ecosystem dependencies, including Qt and native backends | `-d` | `-update_deps` |
| Update registry dependencies and packaging tools | `-i` (`-p` alias) | `-update_pip` |
| Update Qt helpers | `-q` | `-update_qt` |
| Update only image2image | `-j` | `-update_just_app` |
| Update only image2image-io | `-r` | `-update_just_reader` |
| Update only image2image-reg | `-w` | `-update_just_register` |
| Set up/update without building | `-s` | `-no_build` |
| Debug output | `-g` | `-debug` |
| Package | `-z` (DMG) | `-zip`, `-installer` |
| Launch after building | `-l` | `-run` |
| Help | `-h` | `-help` |

The legacy macOS `-n` option is accepted; documentation is not bundled.
Windows `-activate` only creates/activates the environment, without installation.
Dot-source that command if activation should persist in the current shell.

The shared installer can also be called directly with the packaging interpreter:

```sh
package/macos-pyinstaller/venv_package/bin/python package/install_dependencies.py --update-app
package/macos-pyinstaller/venv_package/bin/python package/install_dependencies.py --update-deps --dry-run
```

```powershell
& package\windows-pyinstaller\venv_package\Scripts\python.exe package\install_dependencies.py --update-qt
```

There is no bundler argument. Hyphenated flags, underscore aliases, and the old
single-dash Python installer flags are accepted. `--bootstrap` installs everything.
`--dry-run` prints the requested commands and local overrides without creating
files, installing packages, or importing native modules. It can run outside the
packaging environment; it does not perform dependency resolution.

## Outputs and signing

Outputs live in each platform's `dist` directory:

- macOS: `image2image.app` and `image2image-v<VERSION>-macosx_arm64.dmg`
- Windows: `image2image/image2image.exe`, `image2image-v<VERSION>-win_amd64.zip`,
  and `image2image-<VERSION>-win_amd64-setup.exe`

`package.sh`, `zip.ps1`, and `installer.ps1` can package an existing build without
rebuilding it. The DMG copies the bundle into temporary staging and preserves the
original `.app`. Set `MACOS_SIGN_IDENTITY` to sign and verify the staged app with
your Developer ID identity; without it, the DMG is for unsigned local use.
Notarization is not performed.

Windows installers require Inno Setup 6. `installer.ps1` searches PATH and the
standard installation directories; pass `-InnoSetupCompiler` to override it.
The installer retains its existing AppId, per-user installation, shortcuts, and
file context menus. Windows code signing is not included.

## Frozen reader support

The shared PyInstaller hooks include imzy's Rust extension, pytsf modules and
metadata, and bundled shared libraries. Bruker vendor libraries are excluded on
macOS, which uses pytsf; Waters vendor libraries are included only on Windows.
The specs explicitly exclude pyspa. The universal spec remains available for the
separate release CI and Linux workflows; those workflows are not migrated here.

Packaging unit tests do not need GUI plugins:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_package_installer.py tests/test_package_hooks.py tests/test_package_scripts.py
```
