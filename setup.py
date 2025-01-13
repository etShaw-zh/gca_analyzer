"""Setup script for the GCA Analyzer package.

This script handles the setup and installation of the GCA Analyzer package,
including dependency management and package metadata.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import os

from setuptools import find_packages, setup


# Read version number
with open(
    os.path.join('gca_analyzer', '__version__.py'),
    'r',
    encoding='utf-8'
) as f:
    exec(f.read())


setup(
    name="gca_analyzer",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'jieba>=0.42.1',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'networkx>=2.6.0',
        'plotly>=5.3.0',
        'loguru>=0.7.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'sentence-transformers>=2.2.0',
    ],
    author="Jianjun Xiao",
    author_email="et_shaw@126.com",
    description=(
        "A package for Group Conversation Analysis with improved text "
        "processing and visualization"
    ),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/etShaw-zh/gca_analyzer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires='>=3.12',
    entry_points={
        'console_scripts': [
            'gca_analyzer = gca_analyzer.__main__:main',
        ],
    },
    include_package_data=True,
    package_data={
        'gca_analyzer': ['data/*.txt'],
    },
)
