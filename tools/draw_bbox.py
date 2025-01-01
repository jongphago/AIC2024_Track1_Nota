import cv2
import numpy as np
import matplotlib.pyplot as plt


class DrawSCBoundingBox:
    """ Draw single-camera bounding box
    Example:
    for t in tracks:
        draw_bbox = DrawSCBouondingBox(
            img, t, colors, gid_2_lenfeats=gid_2_lenfeats
        )
        draw_bbox.draw()
        img = draw_bbox.get_image()
    result_imgs.append(img)
    """
    def __init__(self, image, track, colors, m=2, gid_2_lenfeats=None):
        self.m = m
        self.image = image
        self.colors = colors
        self.gid_2_lenfeats = gid_2_lenfeats
        self.is_mc = False
        self.initialize_track(track)
        self.initialize_text()

    def initialize_track(self, track):
        self.track = track
        self.tlbr = self.track.tlbr.astype(int)
        self.score = self.track.score
        self.track_id = self.track.global_id
        self.len_feats = self.gid_2_lenfeats.get(self.track.global_id, -1)

    def initialize_text(self):
        self.text = "S{} : {:.1f}% | {}".format(
            self.track_id, self.score * 100, self.len_feats
        )
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_size = cv2.getTextSize(
            self.text, self.font, 0.4 * self.m, 1 * self.m
        )[0]
        self.text_background_color = (
            (self.colors[self.track_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
        )
        self.text_color = (
            (0, 0, 0)
            if np.mean(self.colors[self.track_id % 80]) > 0.5
            else (255, 255, 255)
        )

    def draw_text_background(self):
        x0, y0 = self.tlbr[:2]
        pt1 = (x0, y0 - 2)
        pt2 = (x0 + self.text_size[0] + 1, y0 - int(1.5 * self.text_size[1]))
        cv2.rectangle(self.image, pt1, pt2, self.text_background_color, -1)

    def draw_text(self):
        x0, y0 = self.tlbr[:2]
        org = (x0, int(y0 - self.text_size[1] / 2))
        font_scale = 0.4 * self.m
        thickness = 1 * self.m
        cv2.putText(
            self.image,
            self.text,
            org,
            self.font,
            font_scale,
            self.text_color,
            thickness=thickness,
        )

    def draw_label(self):
        self.draw_text_background()
        self.draw_text()

    def draw_bounding_box(self, line_thickness=2):
        color = (self.colors[self.track_id % 80] * 255).astype(np.uint8).tolist()
        top_left, bottom_right = self.tlbr[:2], self.tlbr[2:]
        cv2.rectangle(self.image, top_left, bottom_right, color, line_thickness)

    def draw(self):
        self.draw_bounding_box()
        self.draw_label()

    def get_image(self):
        return self.image


class DrawMCBoundingBox(DrawSCBoundingBox):
    """ Draw multi-camera bounding box
    Example:
        copied_imgs = [img.copy() for img in imgs]
        for mtrack in mc_tracker.tracked_mtracks:
            img_paths = mtrack.img_paths
            path_tlbr = mtrack.path_tlbr
            track_id = mtrack.global_id
            for img_path in img_paths:
                img = copied_imgs[img_path]
                tlbr = path_tlbr[img_path].astype(int)
                draw_bbox = DrawMCBoundingBox(img, tlbr, track_id, colors)
                draw_bbox.draw()
                img = draw_bbox.get_image()
        result_imgs = copied_imgs
    """
    def __init__(self, image, tlbr, track_id, colors, m=2):
        self.image = image
        self.tlbr = tlbr.astype(int)
        self.track_id = track_id
        self.colors = colors
        self.m = m
        self.is_mc = True
        self.initialize_text()

    def initialize_text(self):
        self.text = "M{}".format(self.track_id)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_size = cv2.getTextSize(
            self.text, self.font, 0.4 * self.m, 1 * self.m
        )[0]
        self.text_background_color = (
            (self.colors[self.track_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
        )
        self.text_color = (
            (0, 0, 0)
            if np.mean(self.colors[self.track_id % 80]) > 0.5
            else (255, 255, 255)
        )


if __name__ == "__main__":
    _img = np.random.randint(0, 255, (1080, 1920, 3)).astype(np.uint8)
    img_paths = [1]
    path_tlbr = {1: np.array([408.79, 547.86, 501.56, 840.7])}
    track_id = 0
    score = 0.9
    len_features = 10
    m = 2
    text = f"{track_id}"
    txt_color = (255, 255, 255)
    colors = [np.array((0, 0, 255))]
    color = np.array((0, 0, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 0.4 * m, 1 * m)[0]
    txt_bk_color = (color * 0.7).astype(np.uint8).tolist()
    color = (color).astype(np.uint8).tolist()

    for img_path in img_paths:
        draw_bounding_box = DrawMCBoundingBox(
            _img.copy(),
            path_tlbr[img_path],
            track_id,
            score,
            len_features,
            colors,
        )
        draw_bounding_box.draw()

    cv2.imshow("result", draw_bounding_box.get_image())
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
