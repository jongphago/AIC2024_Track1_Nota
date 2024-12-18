import pickle
import cv2

import cv2
import numpy as np


def change_fov(
    frame, dist_coeffs=None, original_focal_length=740, new_focal_length=1200
):
    """
    화각(FOV)을 변경하는 함수.

    Args:
        frame (numpy.ndarray): 입력 이미지 (광각 렌즈 이미지).
        original_focal_length (float): 원본 초점 거리 (기존 카메라 매트릭스의 focal length).
        new_focal_length (float): 새 초점 거리 (표준 렌즈 효과를 위한 focal length).

    Returns:
        numpy.ndarray: 화각이 변경된 이미지.
    """
    # 이미지 크기 가져오기
    h, w = frame.shape[:2]
    if isinstance(original_focal_length, int):
        original_focal_length_x = original_focal_length
        original_focal_length_y = original_focal_length
    else:
        original_focal_length_x, original_focal_length_y = original_focal_length

    # 기존 카메라 매트릭스 정의
    camera_matrix = np.array(
        [
            [original_focal_length_x, 0, w / 2],
            [0, original_focal_length_y, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # 왜곡 계수 (왜곡 없는 경우로 가정)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    # 새로운 카메라 매트릭스 정의
    new_camera_matrix = np.array(
        [
            [new_focal_length, 0, w / 2],
            [0, new_focal_length, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # 리매핑을 위한 변환 매트릭스 계산
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    # 리매핑 적용
    remapped_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    return remapped_frame


def undistort(frame, camera_matrix, dist, new_camera_matrix, roi):
    distorted = cv2.undistort(frame, camera_matrix, dist, None, new_camera_matrix)
    x, y, w, h = roi
    distorted = distorted[y : y + h, x : x + w]
    distorted = cv2.resize(distorted, frame.shape[:2][::-1])
    return distorted


if __name__ == "__main__":
    original_focal_length = 740  # DJI Osomo Action 5Pro 초점 거리
    new_focal_length = 1200  # 표준 렌즈 효과를 위한 초점 거리

    h, w = 1080, 1920
    camera_matrix = pickle.load(open("calibration/osmo_cameraMatrix.pkl", "rb"))
    print(camera_matrix)
    dist = pickle.load(open("calibration/osmo_dist.pkl", "rb"))
    print(dist)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist, (w, h), 1, (w, h)
    )

    capture = cv2.VideoCapture(2)

    while True:
        ret, frame = capture.read()
        undistorted_frame = undistort(
            frame, camera_matrix, dist, new_camera_matrix, roi
        )
        fov_changed_frame = change_fov(
            undistorted_frame, dist, original_focal_length, new_focal_length
        )
        frame = cv2.hconcat([frame, fov_changed_frame])
        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
