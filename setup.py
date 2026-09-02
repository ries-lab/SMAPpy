import subprocess
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind11_build_ext
from setuptools import setup

# Optimization is asked for per compiler, not per platform: MSVC does not know
# -O3 and would fail on it, and the flag has to be chosen when the compiler is
# known, which is inside build_ext rather than here.
OPTIMIZE = {"msvc": ["/O2"], "unix": ["-O3"], "mingw32": ["-O3"]}

extra_compile_args = []
extra_link_args = []


class build_ext(pybind11_build_ext):
    """Add the compiler's own optimization flag once the compiler is known."""

    def build_extensions(self):
        flags = OPTIMIZE.get(self.compiler.compiler_type, [])
        for extension in self.extensions:
            extension.extra_compile_args = list(extension.extra_compile_args) + flags
        super().build_extensions()


def _macos_libcxx_workaround():
    """Work around a Command Line Tools install with incomplete libc++ headers.

    Some macOS setups have an almost empty ``CommandLineTools/usr/include/c++/v1``
    while the SDK's copy is complete, so any ``#include <cstddef>`` fails.  When
    that is the case, point the compiler at the SDK's headers instead.

    The real fix is to repair the toolchain, e.g. by pointing xcode-select at a
    full Xcode (``sudo xcode-select -s /Applications/Xcode.app/Contents/Developer``)
    or reinstalling the Command Line Tools.
    """
    if sys.platform != "darwin":
        return []
    if _compiles([]):
        return []  # the toolchain is fine, which is the normal case

    try:
        developer = subprocess.check_output(["xcode-select", "-p"], text=True).strip()
    except Exception:
        developer = ""

    candidates = [Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")]
    if developer:
        candidates.insert(0, Path(developer) / "SDKs/MacOSX.sdk")
    for sdk in candidates:
        headers = sdk / "usr/include/c++/v1"
        flags = ["-nostdinc++", "-isystem", str(headers)]
        if (headers / "cstddef").exists() and _compiles(flags):
            print(f"note: incomplete libc++ in the active toolchain; "
                  f"using the headers from {headers}")
            return flags

    print("warning: no working C++ standard library found; the build will "
          "probably fail. Try 'sudo xcode-select --install', or point "
          "xcode-select at a full Xcode and run 'sudo xcodebuild -license accept'.")
    return []


def _compiles(flags):
    """Whether a trivial C++ program compiles with these extra flags."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.cpp"
        source.write_text("#include <cstddef>\n#include <thread>\nint main(){}\n")
        try:
            return subprocess.run(
                ["c++", *flags, str(source), "-o", str(Path(tmp) / "probe")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
        except Exception:
            return False


extra_compile_args += _macos_libcxx_workaround()

HEADERS = sorted(str(h) for h in Path("csrc").glob("*.hpp"))

ext_modules = [
    Pybind11Extension(
        "smappy._fit3d",
        ["csrc/fit.cpp"],
        depends=HEADERS,          # a changed header rebuilds the module
        include_dirs=["csrc"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Pybind11Extension(
        "smappy._drift",
        ["csrc/drift.cpp"],
        depends=HEADERS,
        include_dirs=["csrc"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Pybind11Extension(
        "smappy._group",
        ["csrc/group.cpp"],
        depends=HEADERS,
        include_dirs=["csrc"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Pybind11Extension(
        "smappy._render",
        ["csrc/render.cpp"],
        depends=HEADERS,
        include_dirs=["csrc"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
