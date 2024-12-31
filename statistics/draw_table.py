import cv2
import numpy as np
import pandas as pd
import random
import time

# 표 데이터를 랜덤으로 생성하는 함수
def generate_random_table(rows, cols):
    data = {f"Col {j+1}": [random.randint(1, 100) for _ in range(rows)] for j in range(cols)}
    return pd.DataFrame(data)

# OpenCV 화면에 표를 그리는 함수
def draw_table(image, table, start_x, start_y, cell_width, cell_height, font_scale=0.6):
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
    end_y = start_y + rows * cell_height

    # Draw horizontal lines
    for i in range(rows + 1):
        y = start_y + i * cell_height
        cv2.line(image, (start_x, y), (end_x, y), (255, 255, 255), 1)

    # Draw vertical lines
    for j in range(cols + 1):
        x = start_x + j * cell_width
        cv2.line(image, (x, start_y), (x, end_y), (255, 255, 255), 1)

    # Add text in cells
    for i in range(rows):
        for j in range(cols):
            text = str(table.iloc[i, j])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = start_x + j * cell_width + (cell_width - text_size[0]) // 2
            text_y = start_y + i * cell_height + (cell_height + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

# 화면 설정
window_name = "Dynamic Table"
cv2.namedWindow(window_name)
screen_width, screen_height = 640, 480
fps = 30

while True:
    # Create a blank screen
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Generate a random table
    rows, cols = 5, 3
    table = generate_random_table(rows, cols)

    # Draw the table in the bottom-left corner
    draw_table(screen, table, start_x=10, start_y=200, cell_width=100, cell_height=40)

    # Display the frame
    cv2.imshow(window_name, screen)

    # Wait for 1 second (simulate 1 FPS update for the table)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q' key
        break
    
    time.sleep(1)

cv2.destroyAllWindows()