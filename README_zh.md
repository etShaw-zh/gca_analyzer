# GCA Analyzer

一个基于LLM和定量指标分析群组对话动态的Python包。

[English](README.md) | 中文 | [日本語](README_ja.md) | [한국어](README_ko.md)

## 功能特点

- **多语言支持**: 通过LLM模型内置支持多种语言
- **全面的指标**: 通过多个维度分析群组互动
- **自动化分析**: 自动寻找最佳分析窗口并生成详细统计
- **灵活配置**: 可根据不同分析需求自定义参数
- **易于集成**: 支持命令行界面和Python API

## 快速开始

### 安装

```bash
# 从PyPI安装
pip install gca_analyzer

# 开发安装
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### 基本用法

1. 准备CSV格式的对话数据（包含必需列）:
```
conversation_id,person_id,time,text
1A,student1,0:08,老师好！
1A,teacher,0:10,同学们好！
```

2. 运行分析:
```bash
python -m gca_analyzer --data your_data.csv
```

3. GCA 指标的描述性统计:

![描述性统计](/doc/imgs/gca_results.jpg)

## 详细用法

### 命令行选项

```bash
python -m gca_analyzer --data <path_to_data.csv> [options]
```

#### 必需参数
- `--data`: 对话数据CSV文件路径

#### 可选参数
- `--output`: 结果输出目录（默认：`gca_results`）
- `--best-window-indices`: 窗口大小优化阈值（默认：0.3）
  - 范围：0.0-1.0
  - 较低的值会产生较小的窗口
- `--console-level`: 日志级别（默认：INFO）
  - 选项：DEBUG, INFO, WARNING, ERROR, CRITICAL
- `--model-name`: 文本处理用的LLM模型
  - 默认：`iic/nlp_gte_sentence-embedding_chinese-base`
- `--model-mirror`: 模型下载镜像
  - 默认：`https://modelscope.cn/models`

### 输入数据格式

必需的CSV列：
- `conversation_id`: 对话唯一标识符
- `person_id`: 参与者标识符
- `time`: 消息时间（格式：YYYY-MM-DD HH:MM:SS 或 HH:MM:SS 或 MM:SS）
- `text`: 消息内容

### 输出指标

分析器会生成以下指标的综合统计：

1. **参与度**
   - 衡量相对贡献频率
   - 负值表示低于平均参与度
   - 正值表示高于平均参与度

2. **响应性**
   - 衡量参与者对他人的响应程度
   - 较高的值表示更好的响应行为

3. **内部凝聚力**
   - 衡量个人贡献的一致性
   - 较高的值表示消息更连贯

4. **社会影响力**
   - 衡量对群组讨论的影响
   - 较高的值表示对他人有更强影响力

5. **新颖性**
   - 衡量新内容的引入
   - 较高的值表示贡献更具创新性

6. **通信密度**
   - 衡量每条消息的信息含量
   - 较高的值表示消息信息更丰富

结果将以CSV文件形式保存在指定的输出目录中。

## 常见问题

1. **问：为什么某些参与度值为负数？**
   答：参与度值基于群体规模修正，表示相对于完全平等参与的偏离程度。当参与者的贡献低于平等参与量时会得到负值，高于平等参与量时会得到正值，如果所有参与者贡献相等则为0。这种度量方式让我们可以直观地看出每个参与者相对于平等参与的表现。

2. **问：最佳窗口大小是多少？**
   答：分析器会根据`best-window-indices`参数自动寻找最佳窗口大小。较低的值（如0.03）会产生较小的窗口，可能更适合稀疏的对话。

3. **问：如何处理不同语言？**
   答：分析器使用LLM模型进行文本处理，默认支持多种语言。对于中文文本，使用中文基础模型。

## 参与贡献

我们欢迎对GCA Analyzer的贡献！以下是参与方式：

### 贡献方式
- 通过[GitHub Issues](https://github.com/etShaw-zh/gca_analyzer/issues)报告问题和功能需求
- 提交Pull Request修复bug或添加新功能
- 改进文档
- 分享您的使用案例和反馈

### 开发环境设置
1. Fork项目仓库
2. 克隆您的Fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gca_analyzer.git
   cd gca_analyzer
   ```
3. 安装开发依赖:
   ```bash
   pip install -e ".[dev]"
   ```
4. 创建新分支:
   ```bash
   git checkout -b feature-or-fix-name
   ```
5. 修改并提交:
   ```bash
   git add .
   git commit -m "修改描述"
   ```
6. 推送并创建Pull Request

### Pull Request指南
- 遵循现有的代码风格
- 为新功能添加测试
- 及时更新文档
- 确保所有测试通过
- 保持每个Pull Request专注于单一更改

## 许可证

Apache 2.0

## 引用

如果您在研究中使用了此工具，请引用：

```bibtex
@software{gca_analyzer2025,
  author = {Xiao, Jianjun},
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
