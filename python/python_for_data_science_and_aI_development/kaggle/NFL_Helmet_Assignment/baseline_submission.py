"""NFL Helmet Assignment Competitionのbaseline submissionを作成するためのモジュール"""

import numpy as np
import pandas as pd

def create_random_label_submission(helmets, tracks):
    """
    ランダムに割り当てられたヘルメットで baseline submission を作成
    """
    top_helmets = select_top_helmets(helmets)
    game_play_choices = create_game_play_choices(tracks)
    
    submission = assign_random_labels(top_helmets, game_play_choices)
    
    return submission

def select_top_helmets(helmets):
    """
    信頼度に基づいてフレームごとにトップ22のヘルメットを選択
    """
    return (
        helmets.sort_values("conf", ascending=False)
        .groupby("video_frame")
        .head(22)
        .sort_values("video_frame")
        .reset_index(drop=True)
        .copy()
    )

def create_game_play_choices(tracks):
    """
    各"game play"のプレイヤーの選択肢の辞書を作成
    """
    return tracks.groupby(["game_play"])["player"].unique().to_dict()

def assign_random_labels(helmets, game_play_choices):
    """
    ヘルメットの検出にランダムなラベルを割り当てる
    """
    helmets["label"] = np.nan
    submission_data = []

    for video_frame, data in helmets.groupby("video_frame"):
        game_play = video_frame[:12]
        choices = game_play_choices[game_play].copy()
        np.random.shuffle(choices)
        data["label"] = choices[: len(data)]
        submission_data.append(data)

    return pd.concat(submission_data)