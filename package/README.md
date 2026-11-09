# Application packaging

The macOS and Windows PyInstaller build scripts validate `imzy._imzml_rust` and `pytsf` before building.
Use `-d` on macOS or `-update_deps` on Windows to rebuild these backends from sibling checkouts. The full update flags
(`-u` or `-update`) also rebuild them. All existing app-only and reader-only update flags continue to work; they validate
the backends already installed in the packaging environment.

The shared installer can also run directly from the active packaging environment:

```shell
python package/install_dependencies.py --update-deps
python package/install_dependencies.py --update-deps --dry-run
```

The local repositories are `../imzy`, `../timsMac` (distribution name `pytsf`), and `../ims-utils`.
The local ims-utils checkout has a development version below imzy's declared minimum. A temporary resolver override
explicitly selects that source checkout. The packages are installed together after all other dependencies, bypassing
cached wheels and setting `IMZY_BUILD_RUST=1`.
Install Rust and Cargo first; Windows also requires Visual Studio C++ build tools and the Windows SDK.
Import checks reject missing imzy native extensions.

The shared PyInstaller hooks include imzy's Rust extension and pytsf modules, metadata, and any bundled shared libraries.
imzy vendor libraries are collected as binaries in their package directories, including
versioned `.so` files. Windows receives DLLs, Linux receives shared objects, and macOS excludes the unsupported Bruker
vendor libraries and uses pytsf. Waters vendor libraries are included only on Windows. The platform-specific specs load
all shared hooks; the universal spec explicitly imports imzy to trigger its hook.
All specs explicitly exclude pyspa from the distribution, even when it is installed in the build environment.
