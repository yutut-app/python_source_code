import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class NFLAssignmentScorer:
    def __init__(
        self,
        labels_df: pd.DataFrame = None,
        labels_csv="train_labels.csv",
        check_constraints=True,
        weight_col="isDefinitiveImpact",
        impact_weight=1000,
        iou_threshold=0.35,
        remove_sideline=True,
    ):
        """
        2021 Kaggle Competition用の提出ファイルを評価するヘルパークラス。
        
        Args:
            labels_df (pd.DataFrame, optional):
                正解ラベルのデータフレーム。
            labels_csv (str, optional):
                正解ラベルのCSVファイル。
            check_constraints (bool, optional):
                提出ファイルが競技の制約を満たしているかどうかを確認するかどうか。デフォルトはTrue。
            weight_col (str, optional):
                スコアリングに使用する重みを適用するためのラベルデータフレーム内の列。
            impact_weight (int, optional):
                インパクトに適用されるスコアリングメトリックの重み。デフォルトは1000。
            iou_threshold (float, optional):
                正解ボックスとラベルを正しくペアにするために必要な最小IoU。デフォルトは0.35。
            remove_sideline (bool, optional):
                スコアリング前にサイドラインプレイヤーをラベルデータフレームから削除するかどうか。デフォルトはTrue。
        """
        if labels_df is None:
            # CSVからラベルを読み込む
            if labels_csv is None:
                raise Exception("labels_df or labels_csv must be provided")
            else:
                self.labels_df = pd.read_csv(labels_csv)
        else:
            self.labels_df = labels_df.copy()
        if remove_sideline:
            self.labels_df = (
                self.labels_df.query("isSidelinePlayer == False")
                .reset_index(drop=True)
                .copy()
            )
        self.impact_weight = impact_weight
        self.check_constraints = check_constraints
        self.weight_col = weight_col
        self.iou_threshold = iou_threshold

    def check_submission(self, sub):
        """
        提出ファイルがすべての要件を満たしているかどうかを確認する。

        1. フレームごとに22個以下のボックスであること。
        2. ビデオ/フレームごとにラベル予測が1つのみであること。
        3. フレームごとに重複したボックスがないこと。

        Args:
            sub : 提出データフレーム。

        Returns:
            True -> テストをパス
            False -> テストに失敗
        """
        # フレームごとに最大22個のボックス
        max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
        if max_box_per_frame > 22:
            print("Has more than 22 boxes in a single frame")
            return False
        # フレームごとに1つのラベルのみ
        has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
        if has_duplicate_labels:
            print("Has duplicate labels")
            return False
        # 重複ボックスのチェック
        has_duplicate_boxes = (
            sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
        )
        if has_duplicate_boxes:
            print("Has duplicate boxes")
            return False
        return True

    def add_xy(self, df):
        """
        IoUを計算するために必要な`x1`、`x2`、`y1`、`y2`列を追加。

        ピクセルの計算では、0,0は左上隅なので、ボックスの方向は右と下（高さ）で定義される
        """
        df["x1"] = df["left"]
        df["x2"] = df["left"] + df["width"]
        df["y1"] = df["top"]
        df["y2"] = df["top"] + df["height"]
        return df

    def merge_sub_labels(self, sub, labels, weight_col="isDefinitiveImpact"):
        """
        提出ファイルとラベルを外部結合。
        提出ボックスごとに一致するラベルを格納する`sub_label`データフレームを作成。
        正解値には`_gt`サフィックスを、提出値には`_sub`サフィックスを付ける。
        """
        sub = sub.copy()
        labels = labels.copy()

        sub = self.add_xy(sub)
        labels = self.add_xy(labels)

        base_columns = [
            "label",
            "video_frame",
            "x1",
            "x2",
            "y1",
            "y2",
            "left",
            "width",
            "top",
            "height",
        ]

        sub_labels = sub[base_columns].merge(
            labels[base_columns + [weight_col]],
            on=["video_frame"],
            how="right",
            suffixes=("_sub", "_gt"),
        )
        return sub_labels

    def get_iou_df(self, df):
        """
        提出（sub）のバウンディングボックスと
        正解ボックス（gt）のIoUを計算する関数。
        """
        df = df.copy()
        # 1. intersの座標を取得
        df["ixmin"] = df[["x1_sub", "x1_gt"]].max(axis=1)
        df["ixmax"] = df[["x2_sub", "x2_gt"]].min(axis=1)
        df["iymin"] = df[["y1_sub", "y1_gt"]].max(axis=1)
        df["iymax"] = df[["y2_sub", "y2_gt"]].min(axis=1)

        df["iw"] = np.maximum(df["ixmax"] - df["ixmin"] + 1, 0.0)
        df["ih"] = np.maximum(df["iymax"] - df["iymin"] + 1, 0.0)
        
        # 2. intersの面積を計算
        df["inters"] = df["iw"] * df["ih"]

        # 3. unionの面積を計算
        df["uni"] = (
            (df["x2_sub"] - df["x1_sub"] + 1) * (df["y2_sub"] - df["y1_sub"] + 1)
            + (df["x2_gt"] - df["x1_gt"] + 1) * (df["y2_gt"] - df["y1_gt"] + 1)
            - df["inters"]
        )
        # print(uni)
        
        # 4. pred_boxとgt_boxの重なりを計算
        df["iou"] = df["inters"] / df["uni"]

        return df.drop(
            ["ixmin", "ixmax", "iymin", "iymax", "iw", "ih", "inters", "uni"], axis=1
        )

    def filter_to_top_label_match(self, sub_labels):
        """
        正解ボックスが提出ファイル内で
        最も高いIoUを持つボックスにのみリンクされるようにする。
        """
        return (
            sub_labels.sort_values("iou", ascending=False)
            .groupby(["video_frame", "label_gt"])
            .first()
            .reset_index()
        )

    def add_isCorrect_col(self, sub_labels):
        """
        正解ラベルと提出ラベルが一致しているかどうかを示す
        True/False列を追加。
        """
        sub_labels["isCorrect"] = (
            sub_labels["label_gt"] == sub_labels["label_sub"]
        ) & (sub_labels["iou"] >= self.iou_threshold)
        return sub_labels

    def calculate_metric_weighted(
        self, sub_labels, weight_col="isDefinitiveImpact", weight=1000
    ):
        """
        スコアメトリックを計算。
        """
        sub_labels["weight"] = sub_labels.apply(
            lambda x: weight if x[weight_col] else 1, axis=1
        )
        y_pred = sub_labels["isCorrect"].values
        y_true = np.ones_like(y_pred)
        weight = sub_labels["weight"]
        return accuracy_score(y_true, y_pred, sample_weight=weight)

    def score(self, sub, labels_df=None, drop_extra_cols=True):
        """
        提出ファイルを評価し、スコアを計算する。
        """
        if labels_df is None:
            labels_df = self.labels_df.copy()

        if self.check_constraints:
            if not self.check_submission(sub):
                return -999
        sub_labels = self.merge_sub_labels(sub, labels_df, self.weight_col)
        sub_labels = self.get_iou_df(sub_labels).copy()
        sub_labels = self.filter_to_top_label_match(sub_labels).copy()
        sub_labels = self.add_isCorrect_col(sub_labels)
        score = self.calculate_metric_weighted(
            sub_labels, self.weight_col, self.impact_weight
        )
        # Keep `sub_labels for review`
        if drop_extra_cols:
            drop_cols = [
                "x1_sub",
                "x2_sub",
                "y1_sub",
                "y2_sub",
                "x1_gt",
                "x2_gt",
                "y1_gt",
                "y2_gt",
            ]
            sub_labels = sub_labels.drop(drop_cols, axis=1)
        self.sub_labels = sub_labels
        return score

def check_submission(sub):
    """
    提出ファイルが制約を満たしているか確認する。
    """
    # Maximum of 22 boxes per frame.
    max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
    if max_box_per_frame > 22:
        print("Has more than 22 boxes in a single frame")
        return False
    # Only one label allowed per frame.
    has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
    if has_duplicate_labels:
        print("Has duplicate labels")
        return False
    # Check for unique boxes
    has_duplicate_boxes = (
        sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
    )
    if has_duplicate_boxes:
        print("Has duplicate boxes")
        return False
    return True