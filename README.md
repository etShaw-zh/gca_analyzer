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
  - Measures each participant's contribution count and average rate
  - Calculates participation standard deviation and normalized rate
  - Normalized relative to equal participation (1/k)

- **Cross-Cohesion**
  - Analyzes temporal interactions between participants
  - Uses sliding window analysis with optimal window size
  - Based on message cosine similarities and participation patterns

- **Internal Cohesion**
  - Measures self-interaction patterns
  - Derived from cross-cohesion matrix diagonal

- **Overall Responsivity**
  - Evaluates average response patterns to others
  - Computed from cross-participant interactions
  - Normalized by number of other participants (k-1)

- **Social Impact**
  - Measures how others respond to participant's messages
  - Based on incoming cross-cohesion values
  - Normalized by number of other participants (k-1)

- **Newness**
  - Calculates orthogonal projection to previous messages
  - Uses QR decomposition for numerical stability
  - Normalized by participant's total contributions

- **Communication Density**
  - Measures vector norm per word length ratio
  - Averaged over all participant's messages
  - Normalized by total participation count

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

### Metric Calculation Details

```python
from gca_analyzer import GCAAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = GCAAnalyzer()

# Load and preprocess data
data = pd.read_csv('your_data.csv')
current_data, person_list, seq_list, k, n, M = analyzer.participant_pre('video_id', data)

# Get optimal window size
w = analyzer.get_best_window_num(
    seq_list=seq_list,
    M=M,
    best_window_indices=0.3,  # Target participation threshold
    min_num=2,  # Minimum window size
    max_num=10  # Maximum window size
)

# Calculate cross-cohesion matrix
vector, dataset = analyzer.text_processor.doc2vector(current_data.text_clean)
cosine_similarity_matrix = pd.DataFrame(...)  # Calculate similarities
Ksi_lag = analyzer.get_Ksi_lag(w, person_list, k, seq_list, M, cosine_similarity_matrix)

# Get all metrics
results = analyzer.analyze_video('video_id', data)
```

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
