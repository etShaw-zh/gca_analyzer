# GCA Analyzer (群组对话分析器)

一个用于分析群组对话和互动模式的Python包，具有高级文本处理和可视化功能。

[English Documentation](README.md)

## 主要功能

### 1. 文本预处理与分析
- 中文分词（使用jieba分词）
- 停用词过滤
- URL和表情符号处理
- 特殊字符标准化
- TF-IDF向量化

### 2. 互动分析指标
- **参与度 (Participation)**
  - 衡量每个参与者在对话中的贡献比例
  - 考虑发言频率和内容长度

- **内部凝聚力 (Internal Cohesion)**
  - 分析参与者发言的主题连贯性
  - 基于文本相似度计算

- **整体响应性 (Overall Responsivity)**
  - 评估参与者对他人发言的回应速度
  - 考虑时间间隔和内容相关性

- **社会影响力 (Social Impact)**
  - 衡量参与者发言引发的讨论程度
  - 基于后续回应数量和质量

- **新颖性 (Newness)**
  - 评估参与者引入新话题的能力
  - 使用文本相似度和主题建模

- **通信密度 (Communication Density)**
  - 分析单位时间内的有效信息量
  - 考虑发言频率和内容丰富度

### 3. 可视化工具
- 参与度热力图
- 互动网络图
- 指标雷达图
- 时间序列演化图

## 安装方法

```bash
# 使用pip安装
pip install gca_analyzer

# 或者从源代码安装
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

## 快速开始

```python
from gca_analyzer import GCAAnalyzer
import pandas as pd

# 初始化分析器
analyzer = GCAAnalyzer()

# 读取数据
data = pd.read_csv('your_data.csv')

# 分析视频对话
results = analyzer.analyze_video('video_id', data)

# 查看结果
print(results)
```

## 数据格式要求

输入数据应为CSV文件，包含以下必要列：
- `video_id`: 对话/视频标识符
- `person_id`: 参与者标识符
- `time`: 时间点（格式：HH:MM:SS或MM:SS）
- `text`: 文本内容
- `编码`: 认知编码（可选）

示例数据格式：
```csv
video_id,person_id,time,text,编码
1A,教师,0:06,同学们好！,
1A,学生1,0:08,老师好！,
...
```

## 高级用法

### 自定义文本处理

```python
from gca_analyzer.text_processor import TextProcessor

# 创建文本处理器
processor = TextProcessor()

# 添加自定义停用词
processor.add_stop_words(['词1', '词2', '词3'])

# 处理文本
processed_text = processor.chinese_word_cut("你的文本内容")
```

### 指标计算定制

```python
# 自定义窗口大小的分析
results = analyzer.analyze_video(
    video_id='1A',
    data=your_data,
    window_size=30,  # 30秒的分析窗口
    min_response_time=5  # 5秒的最小响应时间
)
```

### 可视化定制

```python
from gca_analyzer.visualizer import GCAVisualizer

# 创建可视化器
viz = GCAVisualizer()

# 绘制互动网络
viz.plot_interaction_network(
    results,
    threshold=0.3,  # 设置连接阈值
    node_size_factor=100,  # 调整节点大小
    edge_width_factor=2  # 调整边的宽度
)

# 绘制时间序列图
viz.plot_temporal_evolution(
    results,
    metrics=['participation', 'newness'],
    window_size='5min'  # 5分钟的滑动窗口
)
```

## 贡献指南

欢迎提交Pull Request来改进代码！在提交之前，请确保：

1. 代码符合PEP 8规范
2. 添加了适当的单元测试
3. 更新了相关文档
4. 所有测试都能通过

## 问题反馈

如果你发现任何问题或有改进建议，请在GitHub上提交Issue。

## 开源协议

本项目采用MIT协议开源 - 详见 [LICENSE](LICENSE) 文件
