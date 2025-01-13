English | [简体中文](README_zh.md)
# GCA Analyzer

<div align="left">
    <a href="https://pypi.org/project/gca-analyzer">
        <img src="https://badge.fury.io/py/gca-analyzer.svg" alt="PyPI version">
    </a>
    <img src="https://github.com/etShaw-zh/gca_analyzer/actions/workflows/python-test.yml/badge.svg" alt="Tests">
    <a href="https://gca-analyzer.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/gca-analyzer/badge/?version=latest" alt="Documentation Status">
    </a>
    <a href="https://codecov.io/gh/etShaw-zh/gca_analyzer">
        <img src="https://codecov.io/gh/etShaw-zh/gca_analyzer/branch/main/graph/badge.svg?token=GLAVYYCD9L" alt="Coverage Status">
    </a>
</div>

## Introduction

GCA Analyzer is a Python package for analyzing group conversation dynamics using NLP techniques and quantitative metrics.

## Features

- **Multi-language Support**: Built-in support for Chinese and other languages through LLM models
- **Comprehensive Metrics**: Analyzes group interactions through multiple dimensions
- **Automated Analysis**: Finds optimal analysis windows and generates detailed statistics
- **Flexible Configuration**: Customizable parameters for different analysis needs
- **Easy Integration**: Command-line interface and Python API support

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

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gca_analyzer,
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  author = {Xiao, Jianjun},
  year = {2025},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
