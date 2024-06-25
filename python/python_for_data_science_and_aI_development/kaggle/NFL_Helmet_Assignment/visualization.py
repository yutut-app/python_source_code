"""NFLのヘルメット検出とトラッキングデータを可視化するモジュール"""

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import pandas as pd

def video_with_baseline_boxes(
    video_path: str,
    baseline_boxes: pd.DataFrame,
    ground_truth_labels: pd.DataFrame,
    output_path: str,
    verbose: bool = True
) -> str:
    """
    ベースラインモデルのboxとground truthのboxで動画に注釈をつける
    """
    video_codec = "mp4v"
    helmet_color = (0, 0, 0)  # Black
    baseline_color = (255, 255, 255)  # White
    impact_color = (0, 0, 255)  # Red

    video_name = os.path.basename(video_path).replace(".mp4", "")
    if verbose:
        print(f"Processing video: {video_name}")

    baseline_boxes = preprocess_boxes(baseline_boxes.copy(), "baseline")
    ground_truth_labels = preprocess_boxes(ground_truth_labels.copy(), "ground_truth")

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_output_path = os.path.join('outputs', "tmp_" + os.path.basename(output_path))
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*video_codec), fps, (width, height)
    )

    frame_count = 0
    while True:
        success, image = video_capture.read()
        if not success:
            break

        frame_count += 1
        annotate_frame(
            image, video_name, frame_count, baseline_boxes, ground_truth_labels,
            helmet_color, baseline_color, impact_color
        )
        output_video.write(image)

    video_capture.release()
    output_video.release()
    
    return tmp_output_path

def preprocess_boxes(boxes: pd.DataFrame, box_type: str) -> pd.DataFrame:
    """
    可視化のためにボックスデータを前処理する
    """
    boxes["video"] = boxes["video_frame"].str.split("_").str[:3].str.join("_")
    boxes["frame"] = boxes["video_frame"].str.split("_").str[-1].astype("int")
    return boxes

def annotate_frame(
    image, video_name, frame_count, baseline_boxes, ground_truth_labels,
    helmet_color, baseline_color, impact_color
):
    """
    単一のフレームに bounding boxes と labels でアノテーションを付ける
    """
    frame_name = f"{video_name}_frame{frame_count}"
    cv2.putText(
        image, frame_name, (0, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, helmet_color, thickness=2
    )

    draw_boxes(
        image, baseline_boxes, video_name, frame_count,
        baseline_color, is_baseline=True
    )
    draw_boxes(
        image, ground_truth_labels, video_name, frame_count,
        helmet_color, impact_color, is_baseline=False
    )

def draw_boxes(
    image, boxes, video_name, frame_count, color, impact_color=None, is_baseline=False
):
    """
     画像に bounding boxes を描画
    """
    frame_boxes = boxes.query("video == @video_name and frame == @frame_count")
    for box in frame_boxes.itertuples(index=False):
        if not is_baseline and box.isDefinitiveImpact:
            box_color, thickness = impact_color, 3
        else:
            box_color, thickness = color, 1

        cv2.rectangle(
            image,
            (box.left, box.top),
            (box.left + box.width, box.top + box.height),
            box_color,
            thickness=thickness,
        )

        if is_baseline:
            label = f"{box.conf:0.2}"
        else:
            label = box.label

        cv2.putText(
            image,
            label,
            (box.left + 1, max(0, box.top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            thickness=1,
        )

def create_football_field(
    line_numbers=True,
    end_zones=True,
    highlight_line=False,
    highlight_line_number=50,
    highlighted_name="Line of Scrimmage",
    fifty_is_los=False,
    figure_size=(12, 6.33),
    field_color="lightgreen",
    end_zone_color='forestgreen',
    axis=None,
):
    """
    プレーを可視化するためにフットボール場のプロットを作成
    """
    if axis is None:
        _, axis = plt.subplots(1, figsize=figure_size)

    rect = patches.Rectangle(
        (0, 0), 120, 53.3, linewidth=0.1,
        edgecolor="r", facecolor=field_color, zorder=0
    )
    axis.add_patch(rect)

    plt.plot(
        [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
         80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
        [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
         53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
        color='black'
    )

    if fifty_is_los:
        axis.plot([60, 60], [0, 53.3], color="gold")
        axis.text(62, 50, "<- Player Yardline at Snap", color="gold")

    if end_zones:
        create_end_zones(axis, end_zone_color)

    axis.axis("off")

    if line_numbers:
        add_yard_lines(axis)

    add_hash_marks(axis, end_zones)

    if highlight_line:
        highlight_specific_line(axis, highlight_line_number, highlighted_name)

    add_border(axis)

    axis.set_xlim((0, 120))
    axis.set_ylim((0, 53.3))
    return axis

def create_end_zones(axis, end_zone_color):
    """フットボール場にエンドゾーンを作成"""
    ez1 = patches.Rectangle(
        (0, 0), 10, 53.3, linewidth=0.1,
        edgecolor="black", facecolor=end_zone_color, alpha=0.6, zorder=0
    )
    ez2 = patches.Rectangle(
        (110, 0), 120, 53.3, linewidth=0.1,
        edgecolor="black", facecolor=end_zone_color, alpha=0.6, zorder=0
    )
    axis.add_patch(ez1)
    axis.add_patch(ez2)

def add_yard_lines(axis):
    """フットボール場にヤードラインの番号を追加"""
    for x in range(20, 110, 10):
        numb = x if x <= 50 else 120 - x
        axis.text(
            x, 5, str(numb - 10),
            horizontalalignment="center",
            fontsize=20, color="black"
        )
        axis.text(
            x - 0.95, 53.3 - 5, str(numb - 10),
            horizontalalignment="center",
            fontsize=20, color="black", rotation=180
        )

def add_hash_marks(axis, end_zones):
    """フットボール場にハッシュマークを追加"""
    hash_range = range(11, 110) if end_zones else range(1, 120)
    for x in hash_range:
        axis.plot([x, x], [0.4, 0.7], color="black")
        axis.plot([x, x], [53.0, 52.5], color="black")
        axis.plot([x, x], [22.91, 23.57], color="black")
        axis.plot([x, x], [29.73, 30.39], color="black")

def highlight_specific_line(axis, highlight_line_number, highlighted_name):
    """フットボール場の特定の行をハイライトする"""
    hl = highlight_line_number + 10
    axis.plot([hl, hl], [0, 53.3], color="yellow")
    axis.text(hl + 2, 50, f"<- {highlighted_name}", color="yellow")

def add_border(axis):
    """フットボール場に境界線を追加"""
    border = patches.Rectangle(
        (-5, -5), 120 + 10, 53.3 + 10, linewidth=0.1,
        edgecolor="orange", facecolor="white", alpha=0, zorder=0
    )
    axis.add_patch(border)

def add_plotly_field(figure):
    """
    フットボール場を Plotly 図に追加
    """
    figure.update_traces(marker_size=20)
    
    figure.update_layout(
        paper_bgcolor='#29a500',
        plot_bgcolor='#29a500',
        font_color='white',
        width=800,
        height=600,
        title="",
        xaxis=dict(nticks=10, title="", visible=False),
        yaxis=dict(scaleanchor="x", title="Temp", visible=False),
        showlegend=True,
        annotations=[
            dict(
                x=-5,
                y=26.65,
                xref="x",
                yref="y",
                text="ENDZONE",
                font=dict(size=16, color="#e9ece7"),
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
                font=dict(size=16, color="#e9ece7"),
                align='center',
                showarrow=False,
                yanchor='middle',
                textangle=90
            )
        ],
        legend=dict(
            traceorder="normal",
            font=dict(family="sans-serif", size=12),
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.00,
            xanchor="center",
            x=0.5
        ),
    )
    
    add_field_elements(figure)
    add_yard_numbers(figure)
    
    return figure

def add_field_elements(figure):
    """Plotly図にフィールド要素を追加"""
    figure.add_shape(type="rect", x0=-10, x1=0, y0=0, y1=53.3,
                     line=dict(color="#c8ddc0", width=3), fillcolor="#217b00", layer="below")
    figure.add_shape(type="rect", x0=100, x1=110, y0=0, y1=53.3,
                     line=dict(color="#c8ddc0", width=3), fillcolor="#217b00", layer="below")
    
    for x in range(0, 100, 10):
        figure.add_shape(type="rect", x0=x, x1=x+10, y0=0, y1=53.3,
                         line=dict(color="#c8ddc0", width=3), fillcolor="#29a500", layer="below")
    
    for x in range(0, 100, 1):
        figure.add_shape(type="line", x0=x, y0=1, x1=x, y1=2,
                         line=dict(color="#c8ddc0", width=2), layer="below")
        figure.add_shape(type="line", x0=x, y0=51.3, x1=x, y1=52.3,
                         line=dict(color="#c8ddc0", width=2), layer="below")
        figure.add_shape(type="line", x0=x, y0=20.0, x1=x, y1=21,
                         line=dict(color="#c8ddc0", width=2), layer="below")
        figure.add_shape(type="line", x0=x, y0=32.3, x1=x, y1=33.3,
                         line=dict(color="#c8ddc0", width=2), layer="below")

def add_yard_numbers(figure):
    """Plotly図にヤード番号を追加"""
    yard_numbers = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98]
    yard_labels = ["G", "1 0", "2 0", "3 0", "4 0", "5 0", "4 0", "3 0", "2 0", "1 0", "G"]
    
    figure.add_trace(go.Scatter(
        x=yard_numbers,
        y=[5] * len(yard_numbers),
        text=yard_labels,
        mode="text",
        textfont=dict(size=20, family="Arial"),
        showlegend=False,
    ))
    
    figure.add_trace(go.Scatter(
        x=yard_numbers,
        y=[48.3] * len(yard_numbers),
        text=yard_labels,
        mode="text",
        textfont=dict(size=20, family="Arial"),
        showlegend=False,
    ))