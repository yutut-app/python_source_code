# NFL ヘルメット割り当てコンペティション

このプロジェクトは、NFLのヘルメット割り当てコンペティションのコードベースです。
ビデオ映像とNFLのNext Gen Stats（NGS）トラッキングデータを活用して、各ヘルメットを正しい選手に割り当てることが目的です。

## ファイル構成

1. `main.py`
   - データの読み込み、処理、スコアリング、可視化を行う
   - ベースラインの提出ファイルを生成する

2. `data_loader.py`
   - データの読み込みと前処理を行う
   - `load_data()` 関数: 全ての必要なデータファイルを読み込む
   - `add_track_features()` 関数: トラッキングデータに追加の特徴量を付与する

3. `scorer.py`
   - 提出ファイルのスコアリングを行うクラスと関数を含む
   - `NFLAssignmentScorer` クラス: 提出ファイルを評価し、スコアを計算する
   - `check_submission()` 関数: 提出ファイルが要件を満たしているかチェックする

4. `visualization.py`
   - データの可視化に関する関数を含む
   - `video_with_baseline_boxes()`: ビデオにベースラインのボックスと正解ラベルを描画する
   - `create_football_field()`: フットボールフィールドの図を作成する
   - `add_plotly_field()`: Plotlyの図にフットボールフィールドを追加する

5. `baseline_submission.py`
   - ベースラインの提出ファイルを生成する関数を含む
   - `create_random_label_submission()`: ランダムにラベルを割り当てた提出ファイルを作成する

## セットアップと使用方法

このプロジェクトでは、必要なデータファイル、出力ディレクトリ、および依存関係リストが既に用意されています。

1. 仮想環境を作成し、アクティベートします：
   ```
   python -m venv myenv
   source myenv/bin/activate  # Linuxまたは macOS の場合
   myenv\Scripts\activate  # Windows の場合
   ```

2. 提供されている `requirements.txt` を使用して、必要なライブラリをインストールします：
   ```
   pip install -r requirements.txt
   ```

3. メインスクリプトを実行します：
   ```
   python main.py
   ```

4. 処理結果は `outputs` ディレクトリに保存されます。

注意：
- 入力データは既に `inputs` ディレクトリに配置されています。
- `outputs` ディレクトリは既に存在し、処理結果の保存に使用されます。
- `requirements.txt` ファイルには、プロジェクトに必要な全ての依存関係が記載されています。

## ディレクトリ構造

```
project_root/
│
├── inputs/             # 入力データファイル
├── outputs/            # 処理結果の出力先
├── myenv/              # 仮想環境（gitignoreに追加することを推奨）
├── requirements.txt    # プロジェクトの依存関係リスト
├── main.py
├── data_loader.py
├── scorer.py
├── visualization.py
├── baseline_submission.py
└── README.md
```

## 注意事項

- このコードは、Jupyter NotebookやJupyterLab環境での実行も可能です。
- 大量のデータを扱うため、十分なメモリを持つ環境で実行してください。
- 可視化の部分では、実行に時間がかかる可能性があります。

## ライセンス

このプロジェクトは [MITライセンス](https://opensource.org/licenses/MIT) のもとで公開されています。
```

このREADME.mdは、プロジェクトの概要、ファイル構成、セットアップと使用方法、ディレクトリ構造、注意事項、およびライセンス情報を含んでいます。既存のディレクトリ構造とファイルを前提とした説明になっており、プロジェクトの全体像を把握しやすくなっています。
