import pandas as pd

def load_data():
    BASE_DIR = 'inputs'
    
    # 各種データをCSVファイルから読み込む
    labels = pd.read_csv(f'{BASE_DIR}/train_labels.csv')
    ss = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
    tr_tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
    te_tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')
    tr_helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
    te_helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')
    img_labels = pd.read_csv(f'{BASE_DIR}/image_labels.csv')

    return labels, ss, tr_tracking, te_tracking, tr_helmets, te_helmets, img_labels


def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    ビデオデータと同期するために便利な特徴量をトラッキングデータに追加する。
    """
    tracks = tracks.copy()
    
    # ゲームとプレイのIDを結合してgame_play列を作成
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    
    # スナップ時刻の辞書を作成
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    # スナップ時刻からのオフセットを計算
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).dt.total_seconds()
    
    # 推定ビデオフレームを計算
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks