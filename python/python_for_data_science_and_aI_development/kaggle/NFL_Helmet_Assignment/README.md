# 概要
このプロジェクトは、NFLのヘルメット割り当てコンペティションのためのコードベースです。ビデオ映像とNFLのNext Gen Stats（NGS）トラッキングデータを活用して、各ヘルメットを正しい選手に割り当てることが目標です。

## ファイル構成

1. `main.py`
   - データの読み込み、処理、スコアリング、可視化を行う
   - ベースラインの提出ファイルを生成する

2. `data_loader.py`
   - NFLのコンペデータを読み込み、前処理するモジュール

3. `scorer.py`
   - 提NFLヘルメットの課題提出採点用モジュール

4. `visualization.py`
   - NFLのヘルメット検出とトラッキングデータを可視化するモジュール

5. `baseline_submission.py`
   - NFL Helmet Assignment Competitionのbaseline submissionを作成するためのモジュール

## セットアップと使用方法

1. Kaggleからデータをダウンロードします：
   - [NFL Health & Safety - Helmet Assignment](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/data) ページにアクセスします。
   - 'Download All' ボタンをクリックしてデータをダウンロードします。
   - ダウンロードしたZIPファイルを解凍し、解凍されたフォルダの名前を 'inputs' に変更します。
   - 'inputs' フォルダをこのプロジェクトのルートディレクトリに移動します。

2. 仮想環境を作成し、アクティベートします：
   ```
   python -m venv myenv
   source myenv/bin/activate  # Linuxまたは macOS の場合
   myenv\Scripts\activate  # Windows の場合
   ```

3. 提供されている `requirements.txt` を使用して、必要なライブラリをインストールします：
   ```
   pip install -r requirements.txt
   ```

4. メインスクリプトを実行します：
   ```
   python main.py
   ```

5. 処理結果は `outputs` ディレクトリに保存されます。

注意：
- `outputs` ディレクトリは既に存在し、処理結果の保存に使用されます。
- `requirements.txt` ファイルには、プロジェクトに必要な全ての依存関係が記載されています。

## ディレクトリ構造

```
project_root/
│
├── inputs.zip          # 圧縮された入力データファイル
├── inputs/             # 解凍後の入力データディレクトリ
├── outputs/            # 処理結果の出力先
├── myenv/              # 仮想環境
├── requirements.txt    # プロジェクトの依存関係リスト
├── main.py
├── data_loader.py
├── scorer.py
├── visualization.py
├── baseline_submission.py
└── README.md
```
