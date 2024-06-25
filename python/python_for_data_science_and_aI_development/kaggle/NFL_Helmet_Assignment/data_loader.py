"""NFLのコンペデータを読み込み、前処理するモジュール"""

import pandas as pd

def load_data():
    """
    NFL 競技に必要なデータファイルを全てロードする。
    """
    base_directory = 'inputs'

    labels = pd.read_csv(f'{base_directory}/train_labels.csv')
    sample_submission = pd.read_csv(f'{base_directory}/sample_submission.csv')
    train_tracking = pd.read_csv(f'{base_directory}/train_player_tracking.csv')
    test_tracking = pd.read_csv(f'{base_directory}/test_player_tracking.csv')
    train_helmets = pd.read_csv(f'{base_directory}/train_baseline_helmets.csv')
    test_helmets = pd.read_csv(f'{base_directory}/test_baseline_helmets.csv')
    image_labels = pd.read_csv(f'{base_directory}/image_labels.csv')

    return (
        labels,
        sample_submission,
        train_tracking,
        test_tracking,
        train_helmets,
        test_helmets,
        image_labels
    )

def add_track_features(tracks, frames_per_second=59.94, snap_frame=10):
    """
    ビデオデータとトラッキングデータの同期に役立つカラム機能を追加
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dictionary = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dictionary)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).dt.total_seconds()
    tracks["estimated_frame"] = (
        ((tracks["snap_offset"] * frames_per_second) + snap_frame).round().astype("int")
    )
    return tracks