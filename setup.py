import re
from collections.abc import Sequence
from pathlib import Path

from setuptools import setup

REQUIREMENTS_DIR = Path("requirements")
EXTRA_REQUIREMENTS = [
    p.name[:-4]  # file name without extention
    for p in REQUIREMENTS_DIR.glob("*.txt")
    if p.name != "base.txt"
]
print("OK")


def read_requirements_file(filename: Path) -> Sequence[str]:
    res = []
    with open(filename) as f:
        for line in f:
            # strip out comment and empty lines
            line = re.sub(
                r"\s*#.s$", "", line.strip()
            )  # \r mean raw string i.e '\n' will be treated as '\' and 'n'
            if line:
                res.append(line)

    return res


read_requirements_file(REQUIREMENTS_DIR / "base.txt")

setup(
    name="efficient_transformer",
    version="0.0.1",
    author="triet",
    packages=["efficient_transformer"],
    install_requires=read_requirements_file(REQUIREMENTS_DIR / "base.txt"),
    extras_require={
        extra: read_requirements_file(REQUIREMENTS_DIR / f"{extra}.txt")
        for extra in EXTRA_REQUIREMENTS
    },
)
