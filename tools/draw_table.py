import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_track_records(trackers):
    track_records = {}
    for tracker in trackers:
        for t in tracker.tracked_stracks:
            track_record = {
                "track_id": t.global_id,
                "X": int(t.location[0][0]),
                "Y": int(t.location[0][1]),
                "distance": int(np.linalg.norm(np.array([0, 0]) - t.location[0])),
                "score": f"{t.score:.2%}"
            }
            if t.global_id not in track_records:
                track_records[t.global_id] = track_record
                track_records[t.global_id]["duration"] = 1
            else:
                track_records[t.global_id].update(track_record)
                track_records[t.global_id]["duration"] += 1
    return track_records


def generate_track_records(track_records, top_n=7):
    column_names = ["track_id", "X", "Y", "distance", "score"]
    if not track_records:
        return pd.DataFrame(columns=column_names)
    track_records = pd.DataFrame(track_records).T
    # 필터링: track_id가 음수가 아닌 데이터만 선택
    filtered_data = track_records[track_records["track_id"] > 0]
    # 정렬: duration을 기준으로 오름차순
    sorted_data = filtered_data.sort_values(by="track_id", ascending=True)
    # 상위 7개의 데이터 선택
    top_data = sorted_data.head(top_n)
    # 5개 컬럼 선택
    result = top_data[column_names]
    return result


def draw_table(
    image, table, start_x, start_y, cell_width, cell_height, top_n=7, font_scale=0.6
):
    """
    Draws a Pandas DataFrame table on an OpenCV image.

    Args:
        image (numpy.ndarray): OpenCV image to draw on.
        table (pandas.DataFrame): Table to draw.
        start_x (int): X-coordinate to start drawing.
        start_y (int): Y-coordinate to start drawing.
        cell_width (int): Width of each cell.
        cell_height (int): Height of each cell.
        font_scale (float): Font size for text.
    """
    # Table dimensions
    rows, cols = table.shape
    end_x = start_x + cols * cell_width
    end_y = start_y + (top_n + 1) * cell_height
    line_color = (0, 0, 0)
    text_color = (0, 0, 0)

    # Draw horizontal lines
    for i in range(top_n + 2):
        y = start_y + i * cell_height
        cv2.line(image, (start_x, y), (end_x, y), line_color, 3)

    # Draw vertical lines
    for j in range(cols + 1):
        x = start_x + j * cell_width
        cv2.line(image, (x, start_y), (x, end_y), line_color, 3)

    # Add text in cells
    columns_names = ["Track ID", "X", "Y", "Distance", "Score"]
    for j, column_name in enumerate(columns_names):
        text_size = cv2.getTextSize(
            column_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )[0]
        text_x = start_x + j * cell_width + (cell_width - text_size[0]) // 2
        text_y = start_y + (cell_height + text_size[1]) // 2
        cv2.putText(
            image,
            column_name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2,
        )

    for i in range(rows):
        for j in range(cols):
            text = str(table.iloc[i, j])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[
                0
            ]
            text_x = start_x + j * cell_width + (cell_width - text_size[0]) // 2
            text_y = start_y + (i + 1) * cell_height + (cell_height + text_size[1]) // 2
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                2,
            )


if __name__ == "__main__":
    track_records = {}  # draw empty table
    table_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    record_table = generate_track_records(track_records)
    draw_table(
        table_frame,
        record_table,
        start_x=210,
        start_y=80,
        cell_width=300,
        cell_height=120,
        font_scale=1.8,
    )
    cv2.imshow("table", table_frame)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
