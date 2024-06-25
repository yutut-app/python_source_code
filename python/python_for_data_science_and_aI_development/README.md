# NFL Helmet Assignment - Getting Started Guide

## 概要
このプロジェクトは、NFLヘルメット割り当て課題の解決を支援するためのガイドです。データの読み込み、特徴量の追加、スコアリング、可視化、ベースライン提出の作成を行います。

## セットアップ

1. 仮想環境を作成してアクティブにします：
   ```bash
   python -m venv myenv
   .\venv\Scripts\activate
   ```

2. 必要なライブラリをインストールします：
   ```bash
   pip install -r requirements.txt
   ```

## スクリプトの説明

### main.py
メインスクリプトであり、以下の処理を行います：
1. データの読み込み
2. 提出ファイルのスコアリング
3. インパクトと非インパクトのスコアリング
4. 可視化の例
5. ベースライン提出ファイルの作成

### data_loader.py
データのロードと特徴量の追加を行います：
- `load_data`: 各種データをCSVファイルから読み込みます。
- `add_track_features`: ビデオデータと同期するための特徴量を追加します。

### scorer.py
提出ファイルのスコアリングを行います：
- `NFLAssignmentScorer`: 提出ファイルを評価するクラス。提出ファイルと正解ラベルを結合し、IoUを計算してスコアを算出します。
- `check_submission`: 提出ファイルが制約を満たしているか確認します。

### visualization.py
データの可視化を行います：
- `video_with_baseline_boxes`: ビデオにベースラインモデルのボックスと正解ボックスを注釈します。
- `create_football_field`: プレイの表示用にフットボールフィールドをプロットします。
- `add_plotly_field`: Plotlyフィールドを追加します。

### baseline_submission.py
ベースライン提出ファイルを作成します：
- `random_label_submission`: 各フレームの最も自信のある22個のヘルメットボックスに基づいてランダムに割り当てられたヘルメットを持つベースライン提出ファイルを作成します。

## 出力

- スコアリング結果がコンソールに表示されます。
- 処理されたビデオが `outputs/labeled_output.mp4` に保存されます。
- ベースライン提出が `outputs/submission.csv` に保存されます。
