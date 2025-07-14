[English](README.md) | 简体中文
# GCA Analyzer

[![PyPI version](https://badge.fury.io/py/gca-analyzer.svg)](https://pypi.org/project/gca-analyzer)
[![support-version](https://img.shields.io/pypi/pyversions/gca-analyzer)](https://img.shields.io/pypi/pyversions/gca-analyzer)
[![license](https://img.shields.io/github/license/etShaw-zh/gca_analyzer)](https://github.com/etShaw-zh/gca_analyzer/blob/master/LICENSE)
[![commit](https://img.shields.io/github/last-commit/etShaw-zh/gca_analyzer)](https://github.com/etShaw-zh/gca_analyzer/commits/master)
[![flake8](https://github.com/etShaw-zh/gca_analyzer/workflows/lint/badge.svg)](https://github.com/etShaw-zh/gca_analyzer/actions?query=workflow%3ALint)
![Tests](https://github.com/etShaw-zh/gca_analyzer/actions/workflows/python-test.yml/badge.svg)
[![Coverage Status](https://codecov.io/gh/etShaw-zh/gca_analyzer/branch/main/graph/badge.svg?token=GLAVYYCD9L)](https://codecov.io/gh/etShaw-zh/gca_analyzer)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/581d2fea968f4b0ab821c8b3d94eaac0)](https://app.codacy.com/gh/etShaw-zh/gca_analyzer/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Documentation Status](https://readthedocs.org/projects/gca-analyzer/badge/?version=latest)](https://gca-analyzer.readthedocs.io/en/latest/?badge=latest)
[![PyPI Downloads](https://static.pepy.tech/badge/gca-analyzer)](https://pepy.tech/projects/gca-analyzer)
[![PyPI Downloads](https://static.pepy.tech/badge/gca-analyzer/month)](https://pepy.tech/projects/gca-analyzer)
[![DOI](https://zenodo.org/badge/915395583.svg)](https://doi.org/10.5281/zenodo.14647250)

## 介绍

GCA Analyzer 是一个使用 NLP 技术和定量指标分析群体对话动态的 Python 包。它提供全面的工具来理解参与者**参与模式**、**互动动态**、**内容新颖性**和**沟通密度**。

## 特性

- **多语言支持**：通过 LLM 模型内置支持中文和其他语言
- **全面的指标**：通过多个维度分析群组互动
- **自动化分析**：自动寻找最优分析窗口并生成详细统计
- **灵活配置**：可根据不同分析需求自定义参数
- **易于集成**：支持命令行界面和 Python API

## 快速开始

### 安装

```bash
# 从 PyPI 安装
pip install gca-analyzer

# 开发安装
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### 基本使用

1. 准备 CSV 格式的对话数据，包含以下必需列：
```
conversation_id,person_id,time,text
1A,student1,0:08,老师好！
1A,teacher,0:10,同学们好！
```

2. 运行分析：

   **交互模式（推荐新手使用）：**
   ```bash
   python -m gca_analyzer --interactive
   # 或
   python -m gca_analyzer -i
   ```

   **命令行模式：**
   ```bash
   python -m gca_analyzer --data your_data.csv
   ```

   **高级选项：**
   ```bash
   python -m gca_analyzer --data your_data.csv --output results/ --model-name your-model --console-level INFO
   ```

3. GCA Analyzer 会生成以下指标的描述性统计数据：

   ![描述性统计](/docs/_static/gca_results.jpg)

   - **参与度**
      - 衡量相对贡献频率
      - 负值表示低于平均水平的参与
      - 正值表示高于平均水平的参与

   - **响应性**
      - 衡量参与者对他人的响应程度
      - 较高的值表示更好的响应行为

   - **内部凝聚力**
      - 衡量个人贡献的一致性
      - 较高的值表示更连贯的消息传递

   - **社会影响力**
      - 衡量对群组讨论的影响力
      - 较高的值表示对他人有更强的影响力

   - **新颖性**
      - 衡量新内容的引入程度
      - 较高的值表示更具创新性的贡献

   - **沟通密度**
      - 衡量每条消息的信息含量
      - 较高的值表示信息更丰富的消息

   结果将以 CSV 文件形式保存在指定的输出目录中。

4. GCA Analyzer 为以下度量提供交互式和信息丰富的可视化：

   ![GCA分析结果](/docs/_static/vizs.png)

   - **雷达图**：对比参与者之间的各项度量
   - **分布图**：展示度量的分布情况

   结果以交互式 HTML 文件的形式保存在指定的输出目录中。

## 引用

如果您在研究中使用了此工具，请引用：

```bibtex
@software{gca_analyzer,
  title = {GCA Analyzer: Group Communication Analysis Tool},
  author = {Xiao, Jianjun},
  year = {2025},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
```
