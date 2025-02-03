Welcome to GCA Analyzer
========================

GCA Analyzer is a Python package for analyzing group conversation dynamics using NLP techniques and quantitative metrics. It provides comprehensive tools for understanding participation patterns, interaction dynamics, content novelty, and communication density in group conversations.

.. .. image:: _static/gca_results.jpg
..    :alt: GCA Analysis Results
..    :align: center
.. image:: https://badge.fury.io/py/gca-analyzer.svg
   :target: https://pypi.org/project/gca-analyzer
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/gca-analyzer
   :target: https://img.shields.io/pypi/pyversions/gca-analyzer
   :alt: Python Support

.. image:: https://img.shields.io/github/license/etShaw-zh/gca_analyzer
   :target: https://github.com/etShaw-zh/gca_analyzer/blob/master/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/last-commit/etShaw-zh/gca_analyzer
   :target: https://github.com/etShaw-zh/gca_analyzer/commits/master
   :alt: Last Commit

.. image:: https://github.com/etShaw-zh/gca_analyzer/actions/workflows/python-test.yml/badge.svg
   :target: https://github.com/etShaw-zh/gca_analyzer/actions/workflows/python-test.yml
   :alt: Tests

.. image:: https://codecov.io/gh/etShaw-zh/gca_analyzer/branch/main/graph/badge.svg?token=GLAVYYCD9L
   :target: https://codecov.io/gh/etShaw-zh/gca_analyzer
   :alt: Coverage Status

.. image:: https://app.codacy.com/project/badge/Grade/581d2fea968f4b0ab821c8b3d94eaac0
   :target: https://app.codacy.com/gh/etShaw-zh/gca_analyzer/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
   :alt: Codacy Badge

.. image:: https://readthedocs.org/projects/gca-analyzer/badge/?version=latest
   :target: https://gca-analyzer.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://static.pepy.tech/badge/gca-analyzer
   :target: https://pepy.tech/projects/gca-analyzer
   :alt: PyPI Downloads

.. image:: https://static.pepy.tech/badge/gca-analyzer/month
   :target: https://pepy.tech/projects/gca-analyzer
   :alt: PyPI Downloads (monthly)

.. image:: https://zenodo.org/badge/915395583.svg
   :target: https://doi.org/10.5281/zenodo.14647250
   :alt: DOI

Key Features
-------------

* **Multi-language Support**: Built-in support for multiple languages through LLM models
* **Comprehensive Metrics**: Analyzes group interactions through multiple dimensions
* **Automated Analysis**: Finds optimal analysis windows and generates detailed statistics
* **Flexible Configuration**: Customizable parameters for different analysis needs
* **Easy Integration**: Command-line interface and Python API support

Documentation Contents
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   advanced_usage
   mathematics
   faq

.. toctree::
   :maxdepth: 2
   :caption: Development

   api_reference
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Project Info

   authors
   contact
   license

Citation
--------

If you use GCA Analyzer in your research, please cite:

.. code-block:: text

   @software{gca_analyzer,
     title = {GCA Analyzer: A Tool for Group Conversation Analysis},
     author = {Jianjun Xiao},
     year = {2025},
     url = {https://github.com/etShaw-zh/gca_analyzer}
   }

References
------------

This package is based on the following papers [Dowell2019]_ [Wang2025]_:

.. [Dowell2019] Dowell, N. M. M., Nixon, T. M., & Graesser, A. C. (2019). Group communication analysis: A computational linguistics approach for detecting sociocognitive roles in multiparty interactions. *Behavior Research Methods, 51* (3), 1007–1041. https://doi.org/10.3758/s13428-018-1102-z

.. [Wang2025] Wang, C., & Xiao, J. (2025). A role recognition model based on students' social-behavioural–cognitive-emotional features during collaborative learning. *Interactive Learning Environments, 0* (0), 1–20. https://doi.org/10.1080/10494820.2024.2442706
