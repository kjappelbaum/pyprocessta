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

setup_kwargs = {
    "name": "pyprocessta",
    "version": "0.1.0",
    "description": "Python library for the analysis of time series data from chemical (engineering) processes",
    "long_description": None,
    "author": "Kevin M. Jablonka",
    "author_email": "kevin.jablonka@epfl.ch",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "package_dir": package_dir,
    "packages": packages,
    "package_data": package_data,
    "extras_require": extras,
    "install_requires": requirements,
    "python_requires": ">=3.8,<3.9.0",
}


setup(**setup_kwargs)
