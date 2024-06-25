import numpy as np
import pandas as pd

def random_label_submission(helmets, tracks):
    """
    各フレームの最も自信のある22個のヘルメットボックスに基づいて
    ランダムに割り当てられたヘルメットを持つベースライン提出ファイルを作成する。
    """
    # 信頼度に基づいてフレームごとに最大22個のヘルメットを選択
    helm_22 = (
        helmets.sort_values("conf", ascending=False)
        .groupby("video_frame")
        .head(22)
        .sort_values("video_frame")
        .reset_index(drop=True)
        .copy()
    )
    # game_playごとにプレイヤーラベルの選択肢を特定
    game_play_choices = tracks.groupby(["game_play"])["player"].unique().to_dict()
    # フレームごとにループしてボックスをランダムに割り当てる
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