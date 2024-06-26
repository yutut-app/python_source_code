"""NFLヘルメットの課題提出スコアリングモジュール"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class NFLAssignmentScorer:
    """NFLのヘルメット課題提出をスコアリングするクラス"""

    def __init__(
        self,
        labels_df: pd.DataFrame = None,
        labels_csv: str = "train_labels.csv",
        check_constraints: bool = True,
        weight_column: str = "isDefinitiveImpact",
        impact_weight: int = 1000,
        iou_threshold: float = 0.35,
        remove_sideline: bool = True,
    ):
        """
        NFLAssignmentScorerを初期化
        """
        self.labels_df = self._initialize_labels(labels_df, labels_csv, remove_sideline)
        self.impact_weight = impact_weight
        self.check_constraints = check_constraints
        self.weight_column = weight_column
        self.iou_threshold = iou_threshold

    def _initialize_labels(self, labels_df, labels_csv, remove_sideline):
        """Labels DataFrameを初期化し、前処理を行う"""
        if labels_df is None:
            if labels_csv is None:
                raise ValueError("Either labels_df or labels_csv must be provided")
            labels_df = pd.read_csv(labels_csv)
        else:
            labels_df = labels_df.copy()

        if remove_sideline:
            labels_df = labels_df.query("isSidelinePlayer == False").reset_index(drop=True)

        return labels_df

    def check_submission(self, submission):
        """
        投稿が全ての要件を満たしているかチェックする
        """
        max_box_per_frame = submission.groupby(["video_frame"])["label"].count().max()
        if max_box_per_frame > 22:
            print("Has more than 22 boxes in a single frame")
            return False

        has_duplicate_labels = submission[["video_frame", "label"]].duplicated().any()
        if has_duplicate_labels:
            print("Has duplicate labels")
            return False

        has_duplicate_boxes = (
            submission[["video_frame", "left", "width", "top", "height"]].duplicated().any()
        )
        if has_duplicate_boxes:
            print("Has duplicate boxes")
            return False

        return True

    def add_xy(self, df):
        """
        IoUを計算するためにx1, x2, y1, y2列を追加
        """
        df["x1"] = df["left"]
        df["x2"] = df["left"] + df["width"]
        df["y1"] = df["top"]
        df["y2"] = df["top"] + df["height"]
        return df

    def merge_sub_labels(self, submission, labels, weight_column="isDefinitiveImpact"):
        """
        submissionとground truthのラベルをマージす
        """
        submission = self.add_xy(submission.copy())
        labels = self.add_xy(labels.copy())

        base_columns = [
            "label", "video_frame", "x1", "x2", "y1", "y2",
            "left", "width", "top", "height",
        ]

        return submission[base_columns].merge(
            labels[base_columns + [weight_column]],
            on=["video_frame"],
            how="right",
            suffixes=("_sub", "_gt"),
        )

    def get_iou_df(self, df):
        """
        ground truth boxに対するsubmission boxのIoUを計算
        """
        df = df.copy()

        df["ixmin"] = df[["x1_sub", "x1_gt"]].max(axis=1)
        df["ixmax"] = df[["x2_sub", "x2_gt"]].min(axis=1)
        df["iymin"] = df[["y1_sub", "y1_gt"]].max(axis=1)
        df["iymax"] = df[["y2_sub", "y2_gt"]].min(axis=1)

        df["iw"] = np.maximum(df["ixmax"] - df["ixmin"] + 1, 0.0)
        df["ih"] = np.maximum(df["iymax"] - df["iymin"] + 1, 0.0)

        df["inters"] = df["iw"] * df["ih"]
        df["uni"] = (
            (df["x2_sub"] - df["x1_sub"] + 1) * (df["y2_sub"] - df["y1_sub"] + 1)
            + (df["x2_gt"] - df["x1_gt"] + 1) * (df["y2_gt"] - df["y1_gt"] + 1)
            - df["inters"]
        )
        df["iou"] = df["inters"] / df["uni"]

        return df.drop(
            ["ixmin", "ixmax", "iymin", "iymax", "iw", "ih", "inters", "uni"], axis=1
        )

    def filter_to_top_label_match(self, sub_labels):
        """
        ground truth boxが最も高いIoUを持つboxにのみマッチされるようにする
        """
        return (
            sub_labels.sort_values("iou", ascending=False)
            .groupby(["video_frame", "label_gt"])
            .first()
            .reset_index()
        )

    def add_is_correct_column(self, sub_labels):
        """
        ground truthとsubmissionラベルが一致するかどうかを示すカラムを追加
        """
        sub_labels["isCorrect"] = (
            (sub_labels["label_gt"] == sub_labels["label_sub"])
            & (sub_labels["iou"] >= self.iou_threshold)
        )
        return sub_labels

    def calculate_weighted_metric(
        self, sub_labels, weight_column="isDefinitiveImpact", weight=1000
    ):
        """
        weighted accuracy score metricを計算
        """
        sub_labels["weight"] = sub_labels.apply(
            lambda x: weight if x[weight_column] else 1, axis=1
        )
        y_pred = sub_labels["isCorrect"].values
        y_true = np.ones_like(y_pred)
        sample_weight = sub_labels["weight"]
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    def score(self, submission, labels_df=None, drop_extra_columns=True):
        """
        submission fileをground truth labelsに対してスコアリングする
        """
        if labels_df is None:
            labels_df = self.labels_df.copy()

        if self.check_constraints and not self.check_submission(submission):
            return -999

        sub_labels = self.merge_sub_labels(submission, labels_df, self.weight_column)
        sub_labels = self.get_iou_df(sub_labels).copy()
        sub_labels = self.filter_to_top_label_match(sub_labels).copy()
        sub_labels = self.add_is_correct_column(sub_labels)
        score = self.calculate_weighted_metric(
            sub_labels, self.weight_column, self.impact_weight
        )

        if drop_extra_columns:
            drop_columns = [
                "x1_sub", "x2_sub", "y1_sub", "y2_sub",
                "x1_gt", "x2_gt", "y1_gt", "y2_gt",
            ]
            sub_labels = sub_labels.drop(drop_columns, axis=1)

        self.sub_labels = sub_labels
        return score

def check_submission(submission):
    """
    submission が全ての要件を満たしているかチェックする
    """
    max_box_per_frame = submission.groupby(["video_frame"])["label"].count().max()
    if max_box_per_frame > 22:
        print("Has more than 22 boxes in a single frame")
        return False

    has_duplicate_labels = submission[["video_frame", "label"]].duplicated().any()
    if has_duplicate_labels:
        print("Has duplicate labels")
        return False

    has_duplicate_boxes = (
        submission[["video_frame", "left", "width", "top", "height"]].duplicated().any()
    )
    if has_duplicate_boxes:
        print("Has duplicate boxes")
        return False

    return True