# GCA Analyzer (群组对话分析器)

一个用于分析群组对话和互动模式的Python包，具有高级文本处理和可视化功能。

[English Documentation](README.md)

## 概述

GCA Analyzer 是一个全面的群组对话分析工具，特别注重中文文本处理。它实现了一系列数学指标来量化群组互动的各个方面，包括参与模式、响应动态和内容分析。

## 核心组件

### 1. 文本处理与分析 (`llm_processor.py`)
- **高级语言模型集成**
  - 基于 transformer 的多语言支持
  - 默认模型：paraphrase-multilingual-MiniLM-L12-v2
  - 支持自定义模型和镜像源
- **文本处理功能**
  - 语义相似度计算
  - 文档向量化
  - 嵌入向量生成
  - 多语言支持（50+种语言）

### 2. 互动分析 (`analyzer.py`)
- **参与度指标**
  - 个人贡献计数和比率
  - 参与标准差
  - 归一化参与率
  - 时序参与模式

- **互动动态**
  - 基于滑动窗口的交叉内聚分析
  - 内部内聚度测量
  - 响应模式分析
  - 社会影响力评估

- **内容分析**
  - 消息新颖度计算
  - 通信密度指标
  - 语义相似度分析
  - 时序内容演化

### 3. 可视化工具 (`visualizer.py`)
- **交互式图表**
  - 参与度热力图
  - 互动网络图
  - 指标雷达图
  - 时序演化图
- **自定义选项**
  - 可配置的配色方案
  - 可调整的图表尺寸
  - 多种可视化风格
  - 导出功能

### 4. 日志系统 (`logger.py`)
- 多级别日志记录
- 彩色控制台输出
- 文件日志支持
- 可配置的日志轮转

## 安装方法

```bash
pip install gca_analyzer
```

开发安装：
```bash
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

## 快速开始

```python
from gca_analyzer import GCAAnalyzer, GCAVisualizer
import pandas as pd

# 初始化分析器
analyzer = GCAAnalyzer()

# 加载对话数据
data = pd.read_csv('conversation_data.csv')

# 分析对话
results = analyzer.analyze_conversation('1A', data)

# 创建可视化
visualizer = GCAVisualizer()
heatmap = visualizer.plot_participation_heatmap(data, conversation_id='1A')
network = visualizer.plot_interaction_network(results, conversation_id='1A')
metrics = visualizer.plot_metrics_radar(results)
```

## 数据格式要求

必需的CSV列：
- `conversation_id`: 对话标识符
- `person_id`: 参与者标识符
- `time`: 时间戳（HH:MM:SS或MM:SS格式）
- `text`: 消息内容

可选列：
- `coding`: 认知编码
- 其他元数据列会被保留

示例：
```csv
conversation_id,person_id,time,text,coding
1A,教师,0:06,同学们好！,greeting
1A,学生1,0:08,老师好！,response
```

## 高级用法

### 自定义模型配置
```python
analyzer = GCAAnalyzer(
    model_name="your-custom-model",
    mirror_url="https://your-model-mirror.com"
)
```

### 可视化自定义
```python
visualizer = GCAVisualizer()
visualizer.plot_participation_heatmap(
    data,
    title="自定义标题",
    figsize=(12, 8)
)
```

## 贡献

欢迎提交贡献！请随时提交 Pull Request。对于重大更改，请先创建 issue 讨论您想要更改的内容。

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 引用

如果您在研究中使用了 GCA Analyzer，请引用：

```bibtex
@software{gca_analyzer2025,
  author = {Jianjun Xiao},
  title = {GCA Analyzer: A Python Package for Group Conversation Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
