import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

import mlx
import nanobind

r = Path(__file__).parent
m = Path(mlx.__path__[0])
n = Path(nanobind.__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=r.parent / "src/rec_mlx")
a = parser.parse_args()
out = a.output / ("_grammar_native" + sysconfig.get_config_var("EXT_SUFFIX"))
link_flags = (
    ["-undefined", "dynamic_lookup", "-Wl,-rpath,@loader_path/../mlx/lib"]
    if sys.platform == "darwin"
    else ["-Wl,-rpath,$ORIGIN/../mlx/lib"]
)
subprocess.run(
    [
        "clang++",
        "-std=c++20",
        "-DNB_DOMAIN=mlx",
        "-O3",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        *link_flags,
        "-I" + str(m / "include"),
        "-I" + str(n / "include"),
        "-I" + str(n / "ext/robin_map/include"),
        "-I" + sysconfig.get_path("include"),
        str(r / "grammar_native.cpp"),
        str(n / "src/nb_combined.cpp"),
        "-L" + str(m / "lib"),
        "-lmlx",
        "-o",
        str(out),
    ],
    check=True,
)
print(out)
