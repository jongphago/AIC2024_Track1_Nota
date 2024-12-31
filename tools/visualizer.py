import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def visualize_tracker(img, tracker, is_label=False):
    """
    Visualize tracker on the image
    :type img: np.ndarray
    :type tracker: Tracker
    """
    img = img.copy()
    for t in tracker.tracked_stracks:
        if not t.is_activated:
            continue
        labels = [{"track_id": t.track_id, "global_id": t.t_global_id}]
        bbox = t.tlbr.astype(int)
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[2], bbox[3])
        bbox_color = (255, 0, 0)
        cv2.rectangle(img, top_left, bottom_right, bbox_color, 2)
        # draw label
        if is_label:
            for label_index, label in enumerate(labels):
                draw_label(
                    img,
                    f"{label['track_id']}_{label['global_id']}",
                    (bbox[0], bbox[1] + 10 * (label_index + 1)),
                    bbox_color,
                )
    return img


def visualize_trackers(imgs, trackers, draw_label=False):
    result_imgs = []
    for img, tracker in zip(imgs, trackers):
        img = visualize_tracker(img, tracker, draw_label=draw_label)
        result_imgs.append(img)
    return np.vstack(result_imgs)

def visualize_dets(img, dets):
    img = img.copy()
    for det_index, det in enumerate(dets):
        bbox = det[:4].astype(int)
        score = det[4]
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            4,
        )
        cv2.putText(
            img,
            f"{det_index:2d}",
            (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            2,
        )
    return img
   
def visualize_detss(imgs, detss):
    result_imgs = []
    for img, dets in zip(imgs, detss):
        img = visualize_dets(img, dets)
        result_imgs.append(img)
    return result_imgs, np.vstack(result_imgs)

def visualize_groups():
    pass