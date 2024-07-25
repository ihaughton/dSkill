import codecs
import os

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


core_requirements = [
    "matplotlib",
    "mojo @ git+https://git@github.com/stepjam/mojo.git#dev",
    "pillow",
    "scipy",
    "opencv-python",
    "opencv-contrib-python",
    "transforms3d",
    "pyquaternion",
    "numpy",
    "gymnasium",
    "imageio",

]

setuptools.setup(
    name="dSkill",
    author="Iain Haughton",
    author_email="iain.haughton@gmail.com",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": ["pre-commit", "pytest"],
        "real": [
            "minimalmodbus",
            "pyRobotiqGripper @ git+https://git@github.com/ihaughton/pyRobotiqGripper.git",
        ],
    },
)
