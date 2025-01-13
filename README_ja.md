# GCA Analyzer

高度なNLP技術と定量的指標を用いてグループ会話のダイナミクスを分析するPythonパッケージです。

[English](README.md) | [中文](README_zh.md) | 日本語 | [韓国語](README_ko.md)

## 特徴

- **多言語サポート**: 高度なLLMモデルによる日本語を含む多言語対応
- **包括的な指標**: 複数の次元でグループ相互作用を分析
- **自動分析**: 最適な分析ウィンドウを検出し、詳細な統計を生成
- **柔軟な設定**: 異なる分析ニーズに対応するカスタマイズ可能なパラメータ
- **簡単な統合**: コマンドラインインターフェースとPython APIのサポート

## クイックスタート

### インストール

```bash
# PyPIからインストール
pip install gca_analyzer

# 開発用
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### 基本的な使用方法

1. 会話データをCSV形式で準備します（必須列を含む）:
```
conversation_id,person_id,time,text
1A,student1,0:08,先生、こんにちは！
1A,teacher,0:10,皆さん、こんにちは！
```

2. 分析を実行:
```bash
python -m gca_analyzer --data your_data.csv
```

## 詳細な使用方法

### コマンドラインオプション

```bash
python -m gca_analyzer --data <path_to_data.csv> [options]
```

#### 必須引数
- `--data`: 会話データCSVファイルへのパス

#### オプション引数
- `--output`: 結果の出力ディレクトリ（デフォルト: `gca_results`）
- `--best-window-indices`: ウィンドウサイズ最適化のしきい値（デフォルト: 0.3）
  - 範囲: 0.0-1.0
  - 低い値はより小さなウィンドウになります
- `--console-level`: ログレベル（デフォルト: INFO）
  - オプション: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `--model-name`: テキスト処理用のLLMモデル
  - デフォルト: `iic/nlp_gte_sentence-embedding_chinese-base`
- `--model-mirror`: モデルダウンロードミラー
  - デフォルト: `https://modelscope.cn/models`

### 入力データ形式

必須のCSV列:
- `conversation_id`: 各会話の一意の識別子
- `person_id`: 参加者の識別子
- `time`: メッセージの日時（形式: YYYY-MM-DD HH:MM:SS または HH:MM:SS または MM:SS）
- `text`: メッセージ内容

### 出力指標

分析ツールは以下の指標について包括的な統計を生成します：

1. **参加度**
   - 相対的な貢献頻度を測定
   - 負の値は平均以下の参加を示す
   - 正の値は平均以上の参加を示す

2. **応答性**
   - 参加者の他者への応答の程度を測定
   - 高い値はより良い応答行動を示す

3. **内部凝集性**
   - 個人の貢献の一貫性を測定
   - 高い値はより一貫性のあるメッセージを示す

4. **社会的影響力**
   - グループディスカッションへの影響力を測定
   - 高い値は他者への強い影響力を示す

5. **新規性**
   - 新しい内容の導入を測定
   - 高い値はより斬新な貢献を示す

6. **コミュニケーション密度**
   - メッセージあたりの情報量を測定
   - 高い値は情報量の多いメッセージを示す

結果は指定された出力ディレクトリにCSVファイルとして保存されます。

## よくある質問

1. **Q: 参加度の値が負になるのはなぜですか？**
   A: 参加度の値は平均を中心に正規化されています。負の値は平均以下の参加を、正の値は平均以上の参加を示します。

2. **Q: 最適なウィンドウサイズはどれくらいですか？**
   A: 分析ツールは`best-window-indices`パラメータに基づいて最適なウィンドウサイズを自動的に見つけます。より低い値（例：0.03）はより小さなウィンドウになり、短い会話に適している場合があります。

3. **Q: 異なる言語はどのように処理されますか？**
   A: 分析ツールはテキスト処理にLLMモデルを使用し、デフォルトで複数の言語をサポートしています。

## コントリビューション

GCA Analyzerへの貢献を歓迎します！以下は参加方法です：

### 貢献方法
- [GitHub Issues](https://github.com/etShaw-zh/gca_analyzer/issues)でバグ報告や機能リクエストを提出
- バグ修正や機能追加のPull Requestを提出
- ドキュメントの改善
- ユースケースやフィードバックの共有

### 開発環境のセットアップ
1. リポジトリをフォーク
2. フォークをクローン:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gca_analyzer.git
   cd gca_analyzer
   ```
3. 開発依存関係をインストール:
   ```bash
   pip install -e ".[dev]"
   ```
4. 変更用のブランチを作成:
   ```bash
   git checkout -b feature-or-fix-name
   ```
5. 変更をコミット:
   ```bash
   git add .
   git commit -m "変更の説明"
   ```
6. プッシュしてPull Requestを作成

### Pull Requestのガイドライン
- 既存のコードスタイルに従う
- 新機能にはテストを追加
- 必要に応じてドキュメントを更新
- すべてのテストが通過することを確認
- Pull Requestは1つの変更に焦点を当てる

## ライセンス

Apache 2.0

## 引用

研究でこのツールを使用する場合は、以下を引用してください：

```bibtex
@software{gca_analyzer2025,
  author = {Xiao, Jianjun},
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
