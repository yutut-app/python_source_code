import matplotlib.pyplot as plt
from IPython.display import Video, display
import plotly.express as px

from data_loader import load_data, add_track_features
from scorer import NFLAssignmentScorer, check_submission
from visualization import video_with_baseline_boxes, create_football_field, add_plotly_field
from baseline_submission import random_label_submission

def main():
    # データを読み込む
    labels, ss, tr_tracking, te_tracking, tr_helmets, te_helmets, img_labels = load_data()

    # 提出ファイルのカラム名を定義
    SUB_COLUMNS = ss.columns 

    # サンプル提出ファイルのスコアを計算
    scorer = NFLAssignmentScorer(labels)
    ss_score = scorer.score(ss)
    print(f'Sample submission scores: {ss_score:0.4f}')

    # インパクトのみの提出ファイルのスコアを計算
    perfect_impacts = labels.query('isDefinitiveImpact == True and isSidelinePlayer == False')
    imp_score = scorer.score(perfect_impacts)
    print(f'A submission with perfect predictions only for impacts scores: {imp_score:0.4f}')

    # 非インパクトのみの提出ファイルのスコアを計算
    perfect_nonimpacts = labels.query('isDefinitiveImpact == False and isSidelinePlayer == False')
    nonimp_score = scorer.score(perfect_nonimpacts)
    print(f'A submission with perfect predictions only for non-impacts scores: {nonimp_score:0.4f}')

    # 完璧な提出ファイルのスコアを計算
    perfect_train = labels.query('isSidelinePlayer == False')[SUB_COLUMNS].copy()
    perfect_score = scorer.score(perfect_train)
    print(f'A perfect training submission scores: {perfect_score:0.4f}')

    # 可視化の例（オプション）
    example_video = 'inputs/train/57584_000336_Sideline.mp4'
    output_path = 'outputs/labeled_output.mp4'
    output_video = video_with_baseline_boxes(example_video, tr_helmets, labels, output_path)
    display(Video(data=output_video, embed=True, height=int(720*0.65), width=int(1280*0.65)))

    # トラッキングデータに特徴量を追加
    tr_tracking = add_track_features(tr_tracking)
    te_tracking = add_track_features(te_tracking)
    
    # NGSトラッキングデータの可視化
    game_play = "57584_000336"
    example_tracks = tr_tracking.query("game_play == @game_play and isSnap == True")
    ax = create_football_field()
    for team, d in example_tracks.groupby("team"):
        ax.scatter(d["x"], d["y"], label=team, s=65, lw=1, edgecolors="black", zorder=5)
    ax.legend().remove()
    ax.set_title(f"Tracking data for {game_play}: at snap", fontsize=15)
    plt.show()

    # NGSデータのアニメーション例
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

    # ベースライン提出ファイルを作成
    train_submission = random_label_submission(tr_helmets, tr_tracking)
    baseline_score = scorer.score(train_submission)
    print(f"The score of random labels on the training set is {baseline_score:0.4f}")

    # テスト提出ファイルを作成
    random_submission = random_label_submission(te_helmets, te_tracking)
    assert check_submission(random_submission)
    random_submission[SUB_COLUMNS].to_csv("outputs/submission.csv", index=False)

if __name__ == "__main__":
    main()