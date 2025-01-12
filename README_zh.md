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
  - 计算每个参与者的发言次数和平均参与率
  - 计算参与标准差和归一化参与率
  - 相对于平均参与度(1/k)进行归一化

- **交叉凝聚力 (Cross-Cohesion)**
  - 分析参与者之间的时序互动模式
  - 使用最优窗口大小的滑动窗口分析
  - 基于消息余弦相似度和参与模式

- **内部凝聚力 (Internal Cohesion)**
  - 测量参与者的自我互动模式
  - 从交叉凝聚力矩阵的对角线元素导出

- **整体响应性 (Overall Responsivity)**
  - 评估对其他参与者的平均响应模式
  - 基于跨参与者互动计算
  - 按其他参与者数量(k-1)归一化

- **社会影响力 (Social Impact)**
  - 衡量其他人对参与者消息的响应程度
  - 基于传入的交叉凝聚力值
  - 按其他参与者数量(k-1)归一化

- **新颖性 (Newness)**
  - 计算与历史消息的正交投影
  - 使用QR分解确保数值稳定性
  - 按参与者的总贡献次数归一化

- **通信密度 (Communication Density)**
  - 计算向量范数与词长比率
  - 对参与者所有消息取平均
  - 按总参与次数归一化

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

### 指标计算详解

```python
from gca_analyzer import GCAAnalyzer
import pandas as pd

# 初始化分析器
analyzer = GCAAnalyzer()

# 加载并预处理数据
data = pd.read_csv('your_data.csv')
current_data, person_list, seq_list, k, n, M = analyzer.participant_pre('video_id', data)

# 获取最优窗口大小
w = analyzer.get_best_window_num(
    seq_list=seq_list,
    M=M,
    best_window_indices=0.3,  # 目标参与阈值
    min_num=2,  # 最小窗口大小
    max_num=10  # 最大窗口大小
)

# 计算交叉凝聚力矩阵
vector, dataset = analyzer.text_processor.doc2vector(current_data.text_clean)
cosine_similarity_matrix = pd.DataFrame(...)  # 计算相似度矩阵
Ksi_lag = analyzer.get_Ksi_lag(w, person_list, k, seq_list, M, cosine_similarity_matrix)

# 获取所有指标
results = analyzer.analyze_video('video_id', data)
```

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
