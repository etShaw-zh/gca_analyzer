# GCA Analyzer

A Python package for analyzing group conversation and interaction patterns with advanced text processing and visualization capabilities.

[中文文档](README_zh.md)

## Features

### 1. Text Processing & Analysis
- Chinese word segmentation (using jieba)
- Stop words filtering
- URL and emoji handling
- Special character normalization
- TF-IDF vectorization

### 2. Interaction Metrics
- **Participation**
  - Measures each participant's contribution proportion
  - Considers speaking frequency and content length

- **Internal Cohesion**
  - Analyzes topic consistency of participant utterances
  - Based on text similarity calculations

- **Overall Responsivity**
  - Evaluates response speed to others' utterances
  - Considers time intervals and content relevance

- **Social Impact**
  - Measures the discussion level triggered by utterances
  - Based on subsequent response quantity and quality

- **Newness**
  - Assesses ability to introduce new topics
  - Uses text similarity and topic modeling

- **Communication Density**
  - Analyzes effective information per unit time
  - Considers speaking frequency and content richness

### 3. Visualization Tools
- Participation heatmaps
- Interaction networks
- Metric radar charts
- Temporal evolution plots

## Installation

```bash
# Using pip
pip install gca_analyzer

# Or install from source
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

## Quick Start

```python
from gca_analyzer import GCAAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = GCAAnalyzer()

# Load data
data = pd.read_csv('your_data.csv')

# Analyze video conversation
results = analyzer.analyze_video('video_id', data)

# View results
print(results)
```

## Data Format

Input data should be a CSV file with these required columns:
- `video_id`: Conversation/video identifier
- `person_id`: Participant identifier
- `time`: Timestamp (format: HH:MM:SS or MM:SS)
- `text`: Text content
- `coding`: Cognitive coding (optional)

Example format:
```csv
video_id,person_id,time,text,coding
1A,teacher,0:06,Hello everyone!,
1A,student1,0:08,Hello teacher!,
...
```

## Advanced Usage

### Custom Text Processing

```python
from gca_analyzer.text_processor import TextProcessor

# Create text processor
processor = TextProcessor()

# Add custom stop words
processor.add_stop_words(['word1', 'word2', 'word3'])

# Process text
processed_text = processor.chinese_word_cut("your text content")
```

### Metric Calculation Customization

```python
# Custom window size analysis
results = analyzer.analyze_video(
    video_id='1A',
    data=your_data,
    window_size=30,  # 30-second analysis window
    min_response_time=5  # 5-second minimum response time
)
```

### Visualization Customization

```python
from gca_analyzer.visualizer import GCAVisualizer

# Create visualizer
viz = GCAVisualizer()

# Plot interaction network
viz.plot_interaction_network(
    results,
    threshold=0.3,  # Set connection threshold
    node_size_factor=100,  # Adjust node size
    edge_width_factor=2  # Adjust edge width
)

# Plot temporal evolution
viz.plot_temporal_evolution(
    results,
    metrics=['participation', 'newness'],
    window_size='5min'  # 5-minute sliding window
)
```

## Contributing

Pull requests are welcome! Before submitting, please ensure:

1. Code follows PEP 8 style guide
2. Appropriate unit tests are added
3. Documentation is updated
4. All tests pass

## Issue Reporting

If you find any issues or have suggestions for improvements, please submit an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
