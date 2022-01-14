# -*- coding: utf-8 -*-
from setuptools import setup

import versioneer

package_dir = {"": "src"}

packages = [
    "pyprocessta",
    "pyprocessta.causalimpact",
    "pyprocessta.eda",
    "pyprocessta.model",
    "pyprocessta.preprocess",
    "pyprocessta.utils",
]

package_data = {"": ["*"]}

extras = {
    "causalimpact": ["tfcausalimpact"],
    "testing": ["pytest", "pytest-cov"],
    "docs": [
        "sphinx",
        "sphinx-book-theme",
        "sphinx-autodoc-typehints",
        "sphinx-copybutton",
    ],
    "dev": ["versioneer"],
    "pre-commit": ["pylint", "pre-commit"],
}
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


required = []
dependency_links = []

# Do not add to required lines pointing to Git repositories
EGG_MARK = "#egg="
for line in requirements:
    if (
        line.startswith("-e git:")
        or line.startswith("-e git+")
        or line.startswith("git:")
        or line.startswith("git+")
    ):
        line = line.lstrip("-e ")  # in case that is using "-e"
        if EGG_MARK in line:
            package_name = line[line.find(EGG_MARK) + len(EGG_MARK) :]
            repository = line[: line.find(EGG_MARK)]
            required.append("%s @ %s" % (package_name, repository))
            dependency_links.append(line)
        else:
            print("Dependency to a git repository should have the format:")
            print("git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name")
    else:
        required.append(line)

setup_kwargs = {
    "name": "pyprocessta",
    "version": versioneer.get_version(),
    "cmdclass": versioneer.get_cmdclass(),
    "description": "Python library for the analysis of time series data from chemical (engineering) processes",
    "long_description": None,
    "author": "Kevin M. Jablonka",
    "author_email": "kevin.jablonka@epfl.ch",
    "maintainer": "Kevin M. Jablonka",
    "maintainer_email": "kevin.jablonka@epfl.ch",
    "url": "https://github.com/kjappelbaum/pyprocessta",
    "package_dir": package_dir,
    "packages": packages,
    "package_data": package_data,
    "extras_require": extras,
    "install_requires": required,
    "dependency_links": dependency_links,
    "python_requires": ">=3.8,<3.9.0",
}


setup(**setup_kwargs)
