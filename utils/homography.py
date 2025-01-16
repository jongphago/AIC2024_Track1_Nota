import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


class PointSelector:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.points = []
        self.current_point = None
        self.load_image()

    def load_image(self):
        cv2.imshow("Image", self.image)
        cv2.setMouseCallback("Image", self.mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_point = (x, y)
            cv2.circle(self.image, (x, y), radius=10, color=(0, 255, 0), thickness=-1)
            cv2.imshow("Image", self.image)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.current_point:
                self.points.append(self.current_point)
                self.current_point = None

    def get_points(self):
        return self.points


class HomographyFinder:
    def __init__(self, camera_id, cam_path, map_path):
        self.camera_id = camera_id
        self.cam_path = cam_path
        self.map_path = map_path
        self.osmo_action_5pro_camera_projection_matrix = [
            [740, 0, 960],
            [0, 740, 540],
            [0, 0, 1],
        ]
        self.osmo_action_5pro_dist = [
            [0.01306166, -0.01449382, -0.00023464, 0.00115228, 0.00826946]
        ]

    def preprocess_images(self, save=True):
        """카메라 이미지(cam_image)와 평면 이미지(map_image)의 크기 차이가 많이 나는 경우 올바른 변환 행렬(H)을 찾을 수 없다.
        이 경우 이미지를 resize하여 두 이미지의 크기를 비슷하게 맞춘다.
        """
        original_map_image = self.original_map_image

        ch, cw, _ = self.cam_image.shape
        oh, ow, _ = original_map_image.shape
        m_ratio = ow / oh

        mw = cw
        mh = int(mw / m_ratio)

        map_image = cv2.resize(
            original_map_image, (mw, mh), interpolation=cv2.INTER_AREA
        )
        self.size = (ch, cw, mh, mw)
        
        return original_map_image

    def initialize_images(self):
        """이미지 경로를 읽고 BGR에서 RGB로 변환하여 반환한다.

        Returns:
            tuple[np.ndarray]: cam_image, map_image
        """
        cam_image = self.read_image(self.cam_path)
        _map_image = self.read_image(self.map_path)
        images = cam_image, _map_image
        return images

    def find_homography(self):
        cam_points = np.array(self.cam_points)
        map_points = np.array(self.map_points)
        H, _ = cv2.findHomography(cam_points, map_points)
        return H

    def warp_cam_image(self):
        cam_image = self.draw_points(self.cam_image, self.cam_points)
        h, w = cam_image.shape[:2]
        warped_image = cv2.warpPerspective(
            cam_image,
            self.H,
            (w, h),
        )
        return warped_image

    def draw_points(self, image, points):
        drawn = cv2.polylines(
            image,
            [np.array(points)],
            True,
            (255, 0, 0),
            2,
        )
        return drawn

    def visualize_results(self):
        cam_image = self.draw_points(self.cam_image, self.cam_points)
        map_image = self.draw_points(self.map_image, self.map_points)
        warped_image = self.warped_image

        _, axes = plt.subplots(1, 3)
        axes[0].imshow(cam_image)
        axes[1].imshow(map_image)
        axes[2].imshow(warped_image)

        plt.show()

    def find(self):
        # Initialize the images
        images = self.initialize_images()
        self.cam_image, self.original_map_image = images
        self.original_cam_image = self.cam_image.copy()
        # Resize the map image
        self.map_image = self.preprocess_images()
        # Get the points
        self.cam_points = self.get_points(self.cam_image)
        self.map_points = self.get_points(self.map_image)
        # Find the homography
        self.H = self.find_homography()
        # Warp the image
        self.warped_image = self.warp_cam_image()
        # Visualize the points
        self.visualize_results()
        logging.info(f"Homography found successfully.\n{self.H}")
        logging.info(f"cam_points = {self.cam_points}")
        logging.info(f"map_points = {self.map_points}")

        self.write_homography()

    def __call__(self):
        self.find()

    @staticmethod
    def read_image(path):
        image = cv2.imread(path)
        return image

    @staticmethod
    def get_points(image: np.ndarray):
        selector = PointSelector(image)
        return selector.get_points()

    def write_homography(self):
        with open(Path(self.cam_path).parent / f"homography.json", "w") as f:
            json.dump(
                {
                    "camera projection matrix": self.osmo_action_5pro_camera_projection_matrix,
                    "homography matrix": self.H.tolist(),
                    "dist": self.osmo_action_5pro_dist,
                    "cap_points": self.cam_points,
                    "map_points": self.map_points,
                },
                f,
            )


def load_cam_image(camera_index, camera_path):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        logging.error("Failed to open the camera.")
        return None
    ret, frame = capture.read()
    if not ret:
        logging.error("Failed to read the frame.")
        return None
    os.makedirs(Path(camera_path).parent, exist_ok=True)
    cv2.imwrite(camera_path, frame)
    capture.release()
    return frame


if __name__ == "__main__":
    # Define the paths
    scene = "scene_042"
    camera_id = 2
    camera_path = f"videos/val/{scene}/camera{camera_id:02d}/cam.png"
    map_path = f"maps/val/{scene}/map_ces.png"

    load_cam_image(camera_id, camera_path)

    homography_finder = HomographyFinder(camera_id, camera_path, map_path)
    homography_finder()
