import os
import matplotlib.pyplot as plt
from IPython.display import Video, display
import plotly.express as px

from data_loader import load_data, add_track_features
from scorer import NFLAssignmentScorer, check_submission
from visualization import (
    video_with_baseline_boxes,
    create_football_field,
    add_plotly_field,
)
from baseline_submission import create_random_label_submission

def visualize_tracking_data(tracking_data):
    """特定のゲームプレイのNGSトラッキングデータを視覚化"""
    game_play = "57584_000336"
    example_tracks = tracking_data.query("game_play == @game_play and isSnap == True")
    
    # フットボール場のプロットを作成
    ax = create_football_field()
    for team, team_data in example_tracks.groupby("team"):
        ax.scatter(team_data["x"], team_data["y"], label=team, s=65, lw=1, edgecolors="black", zorder=5)
    ax.legend().remove()
    ax.set_title(f"Tracking data for {game_play}: at snap", fontsize=15)
    plt.show()

    # NGSデータのアニメーションを作成
    create_ngs_animation(tracking_data, game_play)

def create_ngs_animation(tracking_data, game_play):
    """特定の"game_play"のNGSデータのアニメーションを作成"""
    tracking_data["track_time_count"] = (
        tracking_data.sort_values("time")
        .groupby("game_play")["time"]
        .rank(method="dense")
        .astype("int")
    )

    fig = px.scatter(
        tracking_data.query("game_play == @game_play"),
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
    
def main():
    # データ読み込み
    (
        labels,
        sample_submission,
        train_tracking,
        test_tracking,
        train_helmets,
        test_helmets,
        -
    ) = load_data()


    # スコアラーを初期化し、得点を計算する
    scorer = NFLAssignmentScorer(labels)
    sample_score = scorer.score(sample_submission)
    print(f'Sample submission scores: {sample_score:.4f}')

    perfect_impacts = labels.query('isDefinitiveImpact == True and isSidelinePlayer == False')
    impact_score = scorer.score(perfect_impacts)
    print(f'A submission with perfect predictions only for impacts scores: {impact_score:.4f}')

    perfect_non_impacts = labels.query('isDefinitiveImpact == False and isSidelinePlayer == False')
    non_impact_score = scorer.score(perfect_non_impacts)
    print(f'A submission with perfect predictions only for non-impacts scores: {non_impact_score:.4f}')

    submission_columns = sample_submission.columns
    perfect_train = labels.query('isSidelinePlayer == False')[submission_columns].copy()
    perfect_score = scorer.score(perfect_train)
    print(f'A perfect training submission scores: {perfect_score:.4f}')

    # サンプル動画を可視化
    example_video = 'inputs/train/57584_000336_Sideline.mp4'
    output_path = 'outputs/labeled_output.mp4'
    output_video = video_with_baseline_boxes(example_video, train_helmets, labels, output_path)
    display(Video(data=output_video, embed=True, height=int(720*0.65), width=int(1280*0.65)))

    # トラッキングデータの処理
    train_tracking = add_track_features(train_tracking)
    test_tracking = add_track_features(test_tracking)
    
    # NGSトラッキングデータの可視化
    visualize_tracking_data(train_tracking)

    # "baseline submission"を作成し評価
    train_submission = create_random_label_submission(train_helmets, train_tracking)
    baseline_score = scorer.score(train_submission)
    print(f"The score of random labels on the training set is {baseline_score:.4f}")

    # "test submission"の作成
    test_submission = create_random_label_submission(test_helmets, test_tracking)
    assert check_submission(test_submission)
    test_submission[submission_columns].to_csv("outputs/submission.csv", index=False)

if __name__ == "__main__":
    main()