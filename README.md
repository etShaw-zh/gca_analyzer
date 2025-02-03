English | [简体中文](README_zh.md)
# GCA Analyzer

[![PyPI version](https://badge.fury.io/py/gca-analyzer.svg)](https://pypi.org/project/gca-analyzer)
[![support-version](https://img.shields.io/pypi/pyversions/gca-analyzer)](https://img.shields.io/pypi/pyversions/gca-analyzer)
[![license](https://img.shields.io/github/license/etShaw-zh/gca_analyzer)](https://github.com/etShaw-zh/gca_analyzer/blob/master/LICENSE)
[![commit](https://img.shields.io/github/last-commit/etShaw-zh/gca_analyzer)](https://github.com/etShaw-zh/gca_analyzer/commits/master)
[![flake8](https://github.com/etShaw-zh/gca_analyzer/workflows/lint/badge.svg)](https://github.com/etShaw-zh/gca_analyzer/actions?query=workflow%3ALint)
![Tests](https://github.com/etShaw-zh/gca_analyzer/actions/workflows/python-test.yml/badge.svg)
[![Coverage Status](https://codecov.io/gh/etShaw-zh/gca_analyzer/branch/main/graph/badge.svg?token=GLAVYYCD9L)](https://codecov.io/gh/etShaw-zh/gca_analyzer)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/581d2fea968f4b0ab821c8b3d94eaac0)](https://app.codacy.com/gh/etShaw-zh/gca_analyzer/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Documentation Status](https://readthedocs.org/projects/gca-analyzer/badge/?version=latest)](https://gca-analyzer.readthedocs.io/en/latest/?badge=latest)
[![PyPI Downloads](https://static.pepy.tech/badge/gca-analyzer)](https://pepy.tech/projects/gca-analyzer)
[![PyPI Downloads](https://static.pepy.tech/badge/gca-analyzer/month)](https://pepy.tech/projects/gca-analyzer)
[![DOI](https://zenodo.org/badge/915395583.svg)](https://doi.org/10.5281/zenodo.14647250)

## Introduction

GCA Analyzer is a Python package for analyzing group conversation dynamics using NLP techniques and quantitative metrics. It provides comprehensive tools for understanding **participation patterns**, **interaction dynamics**, **content novelty**, and **communication density** in group conversations.

## Features

- **Multi-language Support**: Built-in support for Chinese and other languages through LLM models
- **Comprehensive Metrics**: Analyzes group interactions through multiple dimensions
- **Automated Analysis**: Finds optimal analysis windows and generates detailed statistics
- **Flexible Configuration**: Customizable parameters for different analysis needs
- **Easy Integration**: Command-line interface and Python API support

> [!tip]  
> 👁 Watch this repo so that you can be notified whenever there are fixes & updates.  
> 📰 2025-01-22 GCA Analyzer v0.4.4 (beta) has been released!  
> 📰 2025-01-14 GCA Analyzer v0.4.3 (beta) has been released!

> [!note]  
> 📝 If you have any questions or suggestions, please [open an issue](https://github.com/etShaw-zh/gca_analyzer/issues) or contact [etShaw-zh](https://github.com/etShaw-zh).  
> 📝 You can find more information about GCA Analyzer in our [documentation](https://gca-analyzer.readthedocs.io/en/latest/).  

## Quick Start

### Installation

```bash
# Install from PyPI
pip install gca-analyzer

# For development
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### Basic Usage

1. Prepare your conversation data in CSV format with required columns:
```
conversation_id,person_id,time,text
1A,student1,0:08,Hello teacher!
1A,teacher,0:10,Hello everyone!
```

2. Run analysis:
```bash
python -m gca_analyzer --data your_data.csv
```

3. Descriptive statistics for GCA measures:

   The analyzer generates comprehensive statistics for the following measures:

   ![Descriptive Statistics](/docs/_static/gca_results.jpg)

   - **Participation**
      - Measures relative contribution frequency
      - Negative values indicate below-average participation
      - Positive values indicate above-average participation

   - **Responsivity**
      - Measures how well participants respond to others
      - Higher values indicate better response behavior

   - **Internal Cohesion**
      - Measures consistency in individual contributions
      - Higher values indicate more coherent messaging

   - **Social Impact**
      - Measures influence on group discussion
      - Higher values indicate stronger impact on others

   - **Newness**
      - Measures introduction of new content
      - Higher values indicate more novel contributions

   - **Communication Density**
      - Measures information content per message
      - Higher values indicate more information-rich messages

   Results are saved as CSV files in the specified output directory.

4. Visualizations for GCA measures:

   The analyzer provides interactive and informative visualizations for the following measures:

   ![GCA Analysis Results](/docs/_static/vizs.png)

   - **Radar Plots**: Compare measures across participants
   - **Distribution Plots**: Visualize measure distributions

   Results are saved as interactive HTML files in the specified output directory.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gca_analyzer,
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  author = {Xiao, Jianjun},
  year = {2025},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
