"""
  # NFL Helmet Assignment Competition

<p align="center">
  <img src="https://storage.googleapis.com/kaggle-media/competitions/NFL%20player%20safety%20analytics/assingment_example.gif.gif" alt="animated" />
</p>

Welcome to the 2021 NFL Health & Safety, Helmet Assignment competition! The competition is part of a collaborative effort between National Football League (NFL) and Amazon Web Services (AWS) to assist in the development of the best sports injury surveillance and mitigation program in the world.

If you participated in the 2020 NFL Impact Detection competition this dataset may look familiar to you. While last year's competition was focused on detecting helmet impacts from video footage - this competition challenges competitors to correctly assign each helmet to its associated player. To solve this problem, Kagglers will need to leverage both video footage and NFL's Next Gen Stats (NGS) tracking data.
"""
import pandas as pd
import matplotlib.pylab as plt
import cv2
from IPython.display import Video, display
import plotly.express as px
import plotly.graph_objects as go

# Read in data files
BASE_DIR = 'datasets'

# Labels and sample submission
labels = pd.read_csv(f'{BASE_DIR}/train_labels.csv')
ss = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')

# Player tracking data
tr_tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
te_tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')

# Baseline helmet detection labels
tr_helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
te_helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')

# Extra image labels
img_labels = pd.read_csv(f'{BASE_DIR}/image_labels.csv')


"""
# What is the goal of this competition?

Simply put, we are trying to assign the correct player "label" on helmets in NFL game footage. Labels consist of a value H (for home team) and V (for visiting team) followed by the player's jersey number. Player labels are provided within the NGS tracking data for each play. A perfect submission would correctly identify the helmet box for every helmet in every frame of video- and assign that helmet the correct player label.
"""
"""
# How does scoring work?

The scoring used for this competition is `Weighted Accuracy`, where detected helmets involved in an impact are weighted 1000x more than non-impact helmets. This scoring is designed specifically because ultimately the NFL will be using this algorithm to assign helmet impacts to players.

To reduce gamification of this metric additional restrictions are placed on submissions. They are:
1. Submission boxes must have at least a 0.35 [Intersection over Union (IoU)](https://en.wikipedia.org/wiki/Jaccard_index) with the ground truth helmet box.
2. Each ground truth helmet box will only be paired with one helmet box per frame in the submitted solution.  For each ground truth box, the submitted box with the highest IoU will be considered for scoring.
3. No more than 22 helmet predictions per video frame (the maximum number of players participating on field at a time). In some situations, sideline players can be seen in the video footage. Sideline players' helmets are not scored in the grading algorithm and can be ignored. Sideline players will have the helmet labels "H00" and "V00". Sideline players should not be included in the submission to avoid exceeding the 22-box and unique label constraints.
4. A player's helmet label must only be predicted once per video frame, i.e. no duplicated labels per frame.
5. All submitted helmet boxes must be unique per video frame, i.e. no identical (left, right, height and width) boxes per frame.


The `check_submission` function can be used to check if your submission meets the above requirements.
"""
def check_submission(sub):
    """
    Checks that the submission meets all the requirements.

    1. No more than 22 Boxes per frame.
    2. Only one label prediction per video/frame
    3. No duplicate boxes per frame.

    Returns:
        True -> Passed the tests
        False -> Failed the test
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

# The sample submission meets these requirements.
check_submission(ss)


"""
The provided `NFLImpactScorer` class can be used to assist in scoring your local predictions. Note that this is not the exact code used in the kaggle scoring system, but results should be nearly identical.
"""
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


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
        Helper class for grading submissions in the
        2021 Kaggle Competition for helmet assignment.
        Version 1.0
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide

        Use:
        ```
        scorer = NFLAssignmentScorer(labels)
        scorer.score(submission_df)

        or

        scorer = NFLAssignmentScorer(labels_csv='labels.csv')
        scorer.score(submission_df)
        ```

        Args:
            labels_df (pd.DataFrame, optional):
                Dataframe containing theground truth label boxes.
            labels_csv (str, optional): CSV of the ground truth label.
            check_constraints (bool, optional): Tell the scorer if it
                should check the submission file to meet the competition
                constraints. Defaults to True.
            weight_col (str, optional):
                Column in the labels DataFrame used to applying the scoring
                weight.
            impact_weight (int, optional):
                The weight applied to impacts in the scoring metrics.
                Defaults to 1000.
            iou_threshold (float, optional):
                The minimum IoU allowed to correctly pair a ground truth box
                with a label. Defaults to 0.35.
            remove_sideline (bool, optional):
                Remove slideline players from the labels DataFrame
                before scoring.
        """
        if labels_df is None:
            # Read label from CSV
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
        Checks that the submission meets all the requirements.

        1. No more than 22 Boxes per frame.
        2. Only one label prediction per video/frame
        3. No duplicate boxes per frame.

        Args:
            sub : submission dataframe.

        Returns:
            True -> Passed the tests
            False -> Failed the test
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

    def add_xy(self, df):
        """
        Adds `x1`, `x2`, `y1`, and `y2` columns necessary for computing IoU.

        Note - for pixel math, 0,0 is the top-left corner so box orientation
        defined as right and down (height)
        """

        df["x1"] = df["left"]
        df["x2"] = df["left"] + df["width"]
        df["y1"] = df["top"]
        df["y2"] = df["top"] + df["height"]
        return df

    def merge_sub_labels(self, sub, labels, weight_col="isDefinitiveImpact"):
        """
        Perform an outer join between submission and label.
        Creates a `sub_label` dataframe which stores the matched label for each submission box.
        Ground truth values are given the `_gt` suffix, submission values are given `_sub` suffix.
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
        This function computes the IOU of submission (sub)
        bounding boxes against the ground truth boxes (gt).
        """
        df = df.copy()

        # 1. get the coordinate of inters
        df["ixmin"] = df[["x1_sub", "x1_gt"]].max(axis=1)
        df["ixmax"] = df[["x2_sub", "x2_gt"]].min(axis=1)
        df["iymin"] = df[["y1_sub", "y1_gt"]].max(axis=1)
        df["iymax"] = df[["y2_sub", "y2_gt"]].min(axis=1)

        df["iw"] = np.maximum(df["ixmax"] - df["ixmin"] + 1, 0.0)
        df["ih"] = np.maximum(df["iymax"] - df["iymin"] + 1, 0.0)

        # 2. calculate the area of inters
        df["inters"] = df["iw"] * df["ih"]

        # 3. calculate the area of union
        df["uni"] = (
            (df["x2_sub"] - df["x1_sub"] + 1) * (df["y2_sub"] - df["y1_sub"] + 1)
            + (df["x2_gt"] - df["x1_gt"] + 1) * (df["y2_gt"] - df["y1_gt"] + 1)
            - df["inters"]
        )
        # print(uni)
        # 4. calculate the overlaps between pred_box and gt_box
        df["iou"] = df["inters"] / df["uni"]

        return df.drop(
            ["ixmin", "ixmax", "iymin", "iymax", "iw", "ih", "inters", "uni"], axis=1
        )

    def filter_to_top_label_match(self, sub_labels):
        """
        Ensures ground truth boxes are only linked to the box
        in the submission file with the highest IoU.
        """
        return (
            sub_labels.sort_values("iou", ascending=False)
            .groupby(["video_frame", "label_gt"])
            .first()
            .reset_index()
        )

    def add_isCorrect_col(self, sub_labels):
        """
        Adds True/False column if the ground truth label
        and submission label are identical
        """
        sub_labels["isCorrect"] = (
            sub_labels["label_gt"] == sub_labels["label_sub"]
        ) & (sub_labels["iou"] >= self.iou_threshold)
        return sub_labels

    def calculate_metric_weighted(
        self, sub_labels, weight_col="isDefinitiveImpact", weight=1000
    ):
        """
        Calculates weighted accuracy score metric.
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
        Scores the submission file against the labels.

        Returns the evaluation metric score for the helmet
        assignment kaggle competition.

        If `check_constraints` is set to True, will return -999 if the
            submission fails one of the submission constraints.
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

SUB_COLUMNS = ss.columns # Expected submission columns
scorer = NFLAssignmentScorer(labels)

# Score the sample submission
ss_score = scorer.score(ss)
print(f'Sample submission scores: {ss_score:0.4f}')

# Score a submission with only impacts
perfect_impacts = labels.query('isDefinitiveImpact == True and isSidelinePlayer == False')
imp_score = scorer.score(perfect_impacts)
print(f'A submission with perfect predictions only for impacts scores: {imp_score:0.4f}')

# Score a submission with only non-impacts
perfect_nonimpacts = labels.query('isDefinitiveImpact == False and isSidelinePlayer == False')
nonimp_score = scorer.score(perfect_nonimpacts)
print(f'A submission with perfect predictions only for non-impacts scores: {nonimp_score:0.4f}')

# Score a perfect submission
perfect_train = labels.query('isSidelinePlayer == False')[SUB_COLUMNS].copy()
perfect_score = scorer.score(perfect_train)
print(f'A perfrect training submission scores: {perfect_score:0.4f}')


'''
After scoring, the `sub_labels` dataframe within the `NFLAssignmentScorer` object can be used to evaluate results including the `iou` between predictions and ground truth boxes and `isCorrect` for correct labels. Ground truth fields have the suffix `_gt` while submission fields have the suffix `_sub`.
'''
print(scorer.sub_labels.head())


'''
# What data are provided?

For the full data details, please review the [data description page](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/data).

*Note: This is a code competition. When you submit, your model will be rerun on a set of 15 unseen plays located in a holdout test set. The publicly provided test videos are simply a set of mock plays (copied from the training set) which are not used in scoring.*

- `/train/` and `/test/` folders contain the video mp4 files to be labeled.
- `train_labels.csv` - This file is only available for the training dataset and provides the ground truth for the 120 training videos.
- `train_player_tracking.csv` and `test_player_tracking.csv` contain the tracking data for all players on the field during the play.
- `train_baseline_helmets.csv` and `test_baseline_helmets.csv` contain imperfect baseline predictions for helmet boxes. The model used to create these files was trained only on the additional images found in the images folder. If you so choose, you may train your own helmet detection model and ignore these files.

Extra Images:
- The `/images/` folder and `image_labels.csv` contains helmet boxes for random frames in videos. These images were used to train the model that produced the `*_baseline_helmets.csv` files. You may choose to use these to train your own helmet detection model.
'''
'''
## Video and Baseline Boxes
As noted above, the provided baseline boxes are imperfect but allow you to quickly tackle the problem without having to address the helmet detection. The below video shows an example of these baseline predictions alongside the true helmet labels.
'''
import os
import cv2
import subprocess
from IPython.display import Video, display
import pandas as pd
import cv2
import pandas as pd
import os

def video_with_baseline_boxes(video_path: str, baseline_boxes: pd.DataFrame, gt_labels: pd.DataFrame, output_path: str, verbose=True) -> str:
    """
    Annotates a video with both the baseline model boxes and ground truth boxes.
    Baseline model prediction confidence is also displayed.
    """
    VIDEO_CODEC = "mp4v"  # Codec changed to 'mp4v'
    HELMET_COLOR = (0, 0, 0)  # Black
    BASELINE_COLOR = (255, 255, 255)  # White
    IMPACT_COLOR = (0, 0, 255)  # Red
    video_name = os.path.basename(video_path).replace(".mp4", "")
    if verbose:
        print(f"Running for {video_name}")
    baseline_boxes = baseline_boxes.copy()
    gt_labels = gt_labels.copy()

    baseline_boxes["video"] = (
        baseline_boxes["video_frame"].str.split("_").str[:3].str.join("_")
    )
    gt_labels["video"] = gt_labels["video_frame"].str.split("_").str[:3].str.join("_")
    baseline_boxes["frame"] = (
        baseline_boxes["video_frame"].str.split("_").str[-1].astype("int")
    )
    gt_labels["frame"] = gt_labels["video_frame"].str.split("_").str[-1].astype("int")

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_output_path = "tmp_" + output_path
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    )
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1

        # Let's add a frame index to the video so we can track where we are
        img_name = f"{video_name}_frame{frame}"
        cv2.putText(
            img,
            img_name,
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            HELMET_COLOR,
            thickness=2,
        )

        # Now, add the boxes
        boxes = baseline_boxes.query("video == @video_name and frame == @frame")
        for box in boxes.itertuples(index=False):
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                BASELINE_COLOR,
                thickness=1,
            )
            cv2.putText(
                img,
                f"{box.conf:0.2}",
                (box.left, max(0, box.top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BASELINE_COLOR,
                thickness=1,
            )

        boxes = gt_labels.query("video == @video_name and frame == @frame")
        for box in boxes.itertuples(index=False):
            # Filter for definitive head impacts and turn labels red
            if box.isDefinitiveImpact == True:
                color, thickness = IMPACT_COLOR, 3
            else:
                color, thickness = HELMET_COLOR, 1
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                color,
                thickness=thickness,
            )
            cv2.putText(
                img,
                box.label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
            )

        output_video.write(img)
    output_video.release()
    
    return tmp_output_path

example_video = 'datasets/train/57584_000336_Sideline.mp4'
output_path = 'labeled_output.mp4'
output_video = video_with_baseline_boxes(example_video, tr_helmets, labels, output_path)

frac = 0.65  # scaling factor for display
display(Video(data=output_video, embed=True, height=int(720*frac), width=int(1280*frac)))


'''
## NGS Tracking Data
The use of the NGS tracking data will be important for correctly labeling videos. Some things to note are:
- NGS data is sampled at a rate of 10Hz, while videos are sampled at roughly 59.94Hz.
- NGS data and videos can be approximately synced by linking the NGS data where `event == "ball_snap"` to the 10th frame of the video (approximately syncronized to the ball snap in the video).
- The NGS data and the orientation of the video cameras are not consistent. Your solution must account for matching the orientation of the video angle relative to the NGS data.

The provided `add_track_features` function may be helpful when attempting to synchronize the NGS data with the video footage.
'''
def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).dt.total_seconds()
    # Estimated video frame
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks



tr_tracking = add_track_features(tr_tracking)
te_tracking = add_track_features(te_tracking)

import matplotlib.patches as patches
import matplotlib.pylab as plt

def create_football_field(
    linenumbers=True,
    endzones=True,
    highlight_line=False,
    highlight_line_number=50,
    highlighted_name="Line of Scrimmage",
    fifty_is_los=False,
    figsize=(12, 6.33),
    field_color="lightgreen",
    ez_color='forestgreen',
    ax=None,
):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor="r",
        facecolor=field_color,
        zorder=0,
    )

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='black')

    if fifty_is_los:
        ax.plot([60, 60], [0, 53.3], color="gold")
        ax.text(62, 50, "<- Player Yardline at Snap", color="gold")
    # Endzones
    if endzones:
        ez1 = patches.Rectangle(
            (0, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor="black",
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ez2 = patches.Rectangle(
            (110, 0),
            120,
            53.3,
            linewidth=0.1,
            edgecolor="black",
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    ax.axis("off")
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(
                x,
                5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color="black",
            )
            ax.text(
                x - 0.95,
                53.3 - 5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color="black",
                rotation=180,
            )
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color="black")
        ax.plot([x, x], [53.0, 52.5], color="black")
        ax.plot([x, x], [22.91, 23.57], color="black")
        ax.plot([x, x], [29.73, 30.39], color="black")

    if highlight_line:
        hl = highlight_line_number + 10
        ax.plot([hl, hl], [0, 53.3], color="yellow")
        ax.text(hl + 2, 50, "<- {}".format(highlighted_name), color="yellow")

    border = patches.Rectangle(
        (-5, -5),
        120 + 10,
        53.3 + 10,
        linewidth=0.1,
        edgecolor="orange",
        facecolor="white",
        alpha=0,
        zorder=0,
    )
    ax.add_patch(border)
    ax.set_xlim((0, 120))
    ax.set_ylim((0, 53.3))
    return ax


'''
Below is a plot of the NGS data for an example play at the moment the ball is snapped. NGS also includes the speed (`s`), acceleration (`a`), orientation (`o`) and direction (`dir`) for each player. More details can be found in the [data description page](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/data).
'''
game_play = "57584_000336"
example_tracks = tr_tracking.query("game_play == @game_play and isSnap == True")
ax = create_football_field()
for team, d in example_tracks.groupby("team"):
    ax.scatter(d["x"], d["y"], label=team, s=65, lw=1, edgecolors="black", zorder=5)
ax.legend().remove()
ax.set_title(f"Tracking data for {game_play}: at snap", fontsize=15)
plt.show()

cap = cv2.VideoCapture(
    "datasets/train/57584_000336_Endzone.mp4"
)
cap.get(10)
_, ez_snap_img = cap.read()


'''
For context, below are images from the sideline and endzone view of the above play at the moment of snap.
'''
cap = cv2.VideoCapture(
    "datasets/train/57584_000336_Sideline.mp4"
)
cap.get(10)
_, sl_snap_img = cap.read()

fig, axs = plt.subplots(2, 1, figsize=(15, 15))

axs[0].imshow(cv2.cvtColor(ez_snap_img, cv2.COLOR_BGR2RGB))
axs[0].axis("off")
axs[0].set_title(f"57584_000336_Endzone.mp4 at snap", fontsize=20)
axs[1].imshow(cv2.cvtColor(sl_snap_img, cv2.COLOR_BGR2RGB))
axs[1].set_title(f"57584_000336_Sideline.mp4 at snap", fontsize=20)
axs[1].axis("off")
plt.tight_layout()


'''
## Example Animation of NGS Data

Below is an animation of an example play using plotly. Thanks to this notebook author for creating the awesome plotly football field:

https://www.kaggle.com/ammarnassanalhajali/nfl-big-data-bowl-2021-animating-players
'''
import plotly.express as px
import plotly.graph_objects as go
import plotly


def add_plotly_field(fig):
    # Reference https://www.kaggle.com/ammarnassanalhajali/nfl-big-data-bowl-2021-animating-players
    fig.update_traces(marker_size=20)
    
    fig.update_layout(paper_bgcolor='#29a500', plot_bgcolor='#29a500', font_color='white',
        width = 800,
        height = 600,
        title = "",
        
        xaxis = dict(
        nticks = 10,
        title = "",
        visible=False
        ),
        
        yaxis = dict(
        scaleanchor = "x",
        title = "Temp",
        visible=False
        ),
        showlegend= True,
  
        annotations=[
       dict(
            x=-5,
            y=26.65,
            xref="x",
            yref="y",
            text="ENDZONE",
            font=dict(size=16,color="#e9ece7"),
            align='center',
            showarrow=False,
            yanchor='middle',
            textangle=-90
        ),
        dict(
            x=105,
            y=26.65,
            xref="x",
            yref="y",
            text="ENDZONE",
            font=dict(size=16,color="#e9ece7"),
            align='center',
            showarrow=False,
            yanchor='middle',
            textangle=90
        )]  
        ,
        legend=dict(
        traceorder="normal",
        font=dict(family="sans-serif",size=12),
        title = "",
        orientation="h",
        yanchor="bottom",
        y=1.00,
        xanchor="center",
        x=0.5
        ),
    )
    ####################################################
        
    fig.add_shape(type="rect", x0=-10, x1=0,  y0=0, y1=53.3,line=dict(color="#c8ddc0",width=3),fillcolor="#217b00" ,layer="below")
    fig.add_shape(type="rect", x0=100, x1=110, y0=0, y1=53.3,line=dict(color="#c8ddc0",width=3),fillcolor="#217b00" ,layer="below")
    for x in range(0, 100, 10):
        fig.add_shape(type="rect", x0=x,   x1=x+10, y0=0, y1=53.3,line=dict(color="#c8ddc0",width=3),fillcolor="#29a500" ,layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=1, x1=x, y1=2,line=dict(color="#c8ddc0",width=2),layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=51.3, x1=x, y1=52.3,line=dict(color="#c8ddc0",width=2),layer="below")
    
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=20.0, x1=x, y1=21,line=dict(color="#c8ddc0",width=2),layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=32.3, x1=x, y1=33.3,line=dict(color="#c8ddc0",width=2),layer="below")
    
    
    fig.add_trace(go.Scatter(
    x=[2,10,20,30,40,50,60,70,80,90,98], y=[5,5,5,5,5,5,5,5,5,5,5],
    text=["G","1 0","2 0","3 0","4 0","5 0","4 0","3 0","2 0","1 0","G"],
    mode="text",
    textfont=dict(size=20,family="Arail"),
    showlegend=False,
    ))
    
    fig.add_trace(go.Scatter(
    x=[2,10,20,30,40,50,60,70,80,90,98], y=[48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3],
    text=["G","1 0","2 0","3 0","4 0","5 0","4 0","3 0","2 0","1 0","G"],
    mode="text",
    textfont=dict(size=20,family="Arail"),
    showlegend=False,
    ))
    
    return fig

tr_tracking["track_time_count"] = (
    tr_tracking.sort_values("time")
    .groupby("game_play")["time"]
    .rank(method="dense")
    .astype("int")
)

fig = px.scatter(
    tr_tracking.query("game_play == @game_play"),
    x="x",
    y="y",
    range_x=[-10, 110],
    range_y=[-10, 53.3],
    hover_data=["player", "s", "a", "dir"],
    color="team",
    animation_frame="track_time_count",
    text="player",
    title=f"Animation of NGS data for game_play {game_play}",
)

fig.update_traces(textfont_size=10)
fig = add_plotly_field(fig)
fig.show()


'''
# A Baseline Submission

The following code shows how to make a baseline submission. In this submission we've randomly assigned labels to the top 22 baseline helmet boxes for each frame.

First we will run this code on the training data and calculate the expected score. Next we create this submission on the test set for submission.
'''
def random_label_submission(helmets, tracks):
    """
    Creates a baseline submission with randomly assigned helmets
    based on the top 22 most confident baseline helmet boxes for
    a frame.
    """
    # Take up to 22 helmets per frame based on confidence:
    helm_22 = (
        helmets.sort_values("conf", ascending=False)
        .groupby("video_frame")
        .head(22)
        .sort_values("video_frame")
        .reset_index(drop=True)
        .copy()
    )
    # Identify player label choices for each game_play
    game_play_choices = tracks.groupby(["game_play"])["player"].unique().to_dict()
    # Loop through frames and randomly assign boxes
    ds = []
    helm_22["label"] = np.nan
    for video_frame, data in helm_22.groupby("video_frame"):
        game_play = video_frame[:12]
        choices = game_play_choices[game_play]
        np.random.shuffle(choices)
        data["label"] = choices[: len(data)]
        ds.append(data)
    submission = pd.concat(ds)
    return submission

train_submission = random_label_submission(tr_helmets, tr_tracking)
scorer = NFLAssignmentScorer(labels)
baseline_score = scorer.score(train_submission)
print(f"The score of random labels on the training set is {baseline_score:0.4f}")

te_tracking = add_track_features(te_tracking)
random_submission = random_label_submission(te_helmets, te_tracking)
# Check to make sure it meets the submission requirements.
assert check_submission(random_submission)
random_submission[ss.columns].to_csv("submission.csv", index=False)