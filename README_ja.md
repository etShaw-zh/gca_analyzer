# GCA Analyzer

NLP技術と定量的指標を用いたグループ会話分析のためのPythonパッケージ。

[English](README.md) | [中文](README_zh.md) | 日本語 | [한국어](README_ko.md)

## 特徴

- **多言語サポート**：LLMモデルによる中国語その他の言語のビルトインサポート
- **包括的な指標**：複数の次元でグループ相互作用を分析
- **自動分析**：最適な分析ウィンドウを自動検出し、詳細な統計を生成
- **柔軟な設定**：様々な分析ニーズに対応するカスタマイズ可能なパラメータ
- **簡単な統合**：コマンドラインインターフェースとPython APIをサポート

## クイックスタート

### インストール

```bash
# PyPIからインストール
pip install gca_analyzer

# 開発用インストール
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### 基本的な使用方法

1. 必要な列を含むCSV形式の会話データを準備：
```
conversation_id,person_id,time,text
1A,student1,0:08,先生、おはようございます！
1A,teacher,0:10,皆さん、おはようございます！
```

2. 分析を実行：
```bash
python -m gca_analyzer --data your_data.csv
```

3. GCA指標の記述統計:

分析ツールは以下の指標について包括的な統計を生成します：

![記述統計](/docs/_static/gca_results.jpg)

- **参加度**
   - 相対的な貢献頻度を測定
   - 負の値は平均以下の参加を示す
   - 正の値は平均以上の参加を示す

- **応答性**
   - 参加者の他者への応答度を測定
   - 高い値はより良い応答行動を示す

- **内部凝集性**
   - 個人の貢献の一貫性を測定
   - 高い値はより一貫性のあるメッセージングを示す

- **社会的影響力**
   - グループ討論への影響力を測定
   - 高い値は他者への強い影響力を示す

- **新規性**
   - 新しい内容の導入を測定
   - 高い値はより革新的な貢献を示す

- **コミュニケーション密度**
   - メッセージあたりの情報量を測定
   - 高い値は情報量の多いメッセージを示す

結果は指定された出力ディレクトリにCSVファイルとして保存されます。

## 引用

研究でこのツールを使用する場合は、以下を引用してください：

```bibtex
@software{gca_analyzer,
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  author = {Xiao, Jianjun},
  year = {2025},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
