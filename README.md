# GCA Analyzer

A Python package for analyzing group conversation dynamics using NLP techniques and quantitative metrics.

English | [中文](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

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
pip install gca_analyzer

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
