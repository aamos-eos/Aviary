from setuptools import setup, find_packages
import re
import os

this_directory = os.path.abspath(os.path.dirname(__file__))

# Read version from atlas package
with open(os.path.join(this_directory, "models", "atlas", "atlas", "__init__.py")) as f:
    __version__ = re.findall(r"""__version__ = ["']+([0-9\.]*)["']+""", f.read())[0]

# Read README for long description
readme_path = os.path.join(this_directory, "models", "atlas", "readme.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = ""

# Read requirements from requirements.txt
def parse_requirements(filepath):
    """Parse requirements.txt, ignoring comments and empty lines."""
    requirements = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

install_requires = parse_requirements(os.path.join(this_directory, "requirements.txt"))

setup(
    name="atlas",
    version=__version__,
    description="Open aircraft conceptual design tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="aircraft design optimization multidisciplinary multi-disciplinary analysis",
    author="Alexander Amos",
    author_email="",
    license="MIT License",
    package_dir={"": "models/atlas"},
    packages=find_packages(where="models/atlas", include=["atlas*"]),
    install_requires=install_requires,
    extras_require={
        "testing": ["pytest", "pytest-cov", "coverage", "openaerostruct", "parameterized"],
        "docs": ["sphinx_mdolab_theme", "openaerostruct<=2.7.1"],
        "plot": ["matplotlib"],
    },
)
