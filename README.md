# GCA Analyzer

A Python package for analyzing group conversation and interaction patterns with advanced text processing and visualization capabilities.

[中文文档](README_zh.md)

## Overview

GCA Analyzer is a comprehensive tool for analyzing group conversations, with particular emphasis on Chinese text processing. It implements a series of mathematical metrics to quantify various aspects of group interactions, including participation patterns, response dynamics, and content analysis.

## Core Components

### 1. Text Processing & Analysis (`llm_processor.py`)
- **Advanced Language Model Integration**
  - Multilingual support with transformer-based models
  - Default model: paraphrase-multilingual-MiniLM-L12-v2
  - Support for custom models and mirror sources
- **Text Processing Features**
  - Semantic similarity computation
  - Document vectorization
  - Embedding generation
  - Multilingual capability (50+ languages)

### 2. Interaction Analysis (`analyzer.py`)
- **Participation Metrics**
  - Individual contribution count and rate
  - Participation standard deviation
  - Normalized participation rate
  - Temporal participation patterns

- **Interaction Dynamics**
  - Cross-cohesion analysis with sliding windows
  - Internal cohesion measurement
  - Response pattern analysis
  - Social impact evaluation

- **Content Analysis**
  - Message novelty calculation
  - Communication density metrics
  - Semantic similarity analysis
  - Temporal content evolution

### 3. Visualization Tools (`visualizer.py`)
- **Interactive Plots**
  - Participation heatmaps
  - Interaction network graphs
  - Metric radar charts
  - Temporal evolution plots
- **Customization Options**
  - Configurable color schemes
  - Adjustable plot dimensions
  - Multiple visualization styles
  - Export capabilities

### 4. Logging System (`logger.py`)
- Comprehensive logging with different levels
- Colored console output
- File logging support
- Configurable log rotation

## Installation

```bash
pip install gca_analyzer
```

For development installation:
```bash
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

## Quick Start

```python
from gca_analyzer import GCAAnalyzer, GCAVisualizer
import pandas as pd

# Initialize analyzer with default settings
analyzer = GCAAnalyzer()

# Load your conversation data
data = pd.read_csv('conversation_data.csv')

# Analyze conversations
results = analyzer.analyze_conversation('1A', data)

# Create visualizations
visualizer = GCAVisualizer()
heatmap = visualizer.plot_participation_heatmap(data, conversation_id='1A')
network = visualizer.plot_interaction_network(results, conversation_id='1A')
metrics = visualizer.plot_metrics_radar(results)
```

## Input Data Format

Required CSV columns:
- `conversation_id`: Conversation identifier
- `person_id`: Participant identifier
- `time`: Timestamp (HH:MM:SS or MM:SS)
- `text`: Message content

Optional columns:
- `coding`: Cognitive coding
- Additional metadata columns are preserved

Example:
```csv
conversation_id,person_id,time,text,coding
1A,teacher,0:06,Hello everyone!,greeting
1A,student1,0:08,Hello teacher!,response
```

## Advanced Usage

### Custom Model Configuration
```python
analyzer = GCAAnalyzer(
    model_name="your-custom-model",
    mirror_url="https://your-model-mirror.com"
)
```

### Visualization Customization
```python
visualizer = GCAVisualizer()
visualizer.plot_participation_heatmap(
    data,
    title="Custom Title",
    figsize=(12, 8)
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use GCA Analyzer in your research, please cite:

```bibtex
@software{gca_analyzer2025,
  author = {Jianjun Xiao},
  title = {GCA Analyzer: A Python Package for Group Conversation Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
