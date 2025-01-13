# GCA Analyzer

A Python package for analyzing group conversation dynamics using advanced NLP techniques and quantitative metrics.

English | [中文](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

## Features

- **Multi-language Support**: Built-in support for Chinese and other languages through advanced LLM models
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

## Detailed Usage

### Command Line Options

```bash
python -m gca_analyzer --data <path_to_data.csv> [options]
```

#### Required Arguments
- `--data`: Path to the conversation data CSV file

#### Optional Arguments
- `--output`: Output directory for results (default: `gca_results`)
- `--best-window-indices`: Window size optimization threshold (default: 0.3)
  - Range: 0.0-1.0
  - Lower values result in smaller windows
- `--console-level`: Logging level (default: INFO)
  - Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `--model-name`: LLM model for text processing
  - Default: `iic/nlp_gte_sentence-embedding_chinese-base`
- `--model-mirror`: Model download mirror
  - Default: `https://modelscope.cn/models`

### Input Data Format

Required CSV columns:
- `conversation_id`: Unique identifier for each conversation
- `person_id`: Participant identifier
- `time`: Message date (format: YYYY-MM-DD HH:MM:SS or HH:MM:SS or MM:SS)
- `text`: Message content

### Output Metrics

The analyzer generates comprehensive statistics for the following measures:

1. **Participation**
   - Measures relative contribution frequency
   - Negative values indicate below-average participation
   - Positive values indicate above-average participation

2. **Responsivity**
   - Measures how well participants respond to others
   - Higher values indicate better response behavior

3. **Internal Cohesion**
   - Measures consistency in individual contributions
   - Higher values indicate more coherent messaging

4. **Social Impact**
   - Measures influence on group discussion
   - Higher values indicate stronger impact on others

5. **Newness**
   - Measures introduction of new content
   - Higher values indicate more novel contributions

6. **Communication Density**
   - Measures information content per message
   - Higher values indicate more information-rich messages

Results are saved as CSV files in the specified output directory.

## FAQ

1. **Q: Why are some participation values negative?**
   A: Participation values are normalized around the mean. Negative values indicate below-average participation, while positive values indicate above-average participation.

2. **Q: What's the optimal window size?**
   A: The analyzer automatically finds the optimal window size based on the `best-window-indices` parameter. Lower values (e.g., 0.03) result in smaller windows, which may be more suitable for shorter conversations.

3. **Q: How to handle different languages?**
   A: The analyzer uses LLM models for text processing and supports multiple languages by default. For Chinese text, it uses the Chinese base model.

## Contributing

We welcome contributions to GCA Analyzer! Here's how you can help:

### Ways to Contribute
- Report bugs and feature requests through [GitHub Issues](https://github.com/etShaw-zh/gca_analyzer/issues)
- Submit pull requests for bug fixes and features
- Improve documentation
- Share your use cases and feedback

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gca_analyzer.git
   cd gca_analyzer
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature-or-fix-name
   ```
5. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
6. Push and create a pull request

### Pull Request Guidelines
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass
- Keep pull requests focused on a single change

## License

Apache 2.0

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gca_analyzer2025,
  author = {Xiao, Jianjun},
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
