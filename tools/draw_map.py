import cv2
import numpy as np


class DrawMap:
    """Draw a track on a map image
    Example:
        for t in tracks:
            draw_map = DrawMap(sc_map_image, t, tracker_index, colors, is_mc=is_mc)
            draw_map.draw()
            sc_map_image = draw_map.get_image()
    """

    def __init__(self, map_image, track, tracker_index, colors, is_mc):
        self.track = track
        self.map_image = map_image
        self.is_mc = is_mc
        self.initialize_track(colors)
        self.initialize_text(tracker_index)

    def initialize_location(self):
        if self.is_mc:
            self.location = self.track.curr_coords[0].astype(int).tolist()
        else:
            self.location = tuple(map(int, self.track.location[0]))

    def initialize_track(self, colors):
        self.initialize_location()
        self.track_id = self.track.global_id
        self.color = (colors[self.track_id % 80] * 255).astype(np.uint8).tolist()

    def initialize_text(self, tracker_index):
        prefix = "M" if self.is_mc else "S"
        self.text = f"{prefix}{str(self.track_id)}[{tracker_index}]"
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1

    def draw_map_position(self):
        radius = 10
        thickness = -1
        cv2.circle(
            self.map_image,
            self.location,
            radius,
            self.color,
            thickness,
        )

    def draw_map_label(self, thickness=2):
        cv2.putText(
            self.map_image,
            self.text,
            self.location,
            self.font_face,
            self.font_scale,
            self.color,
            thickness,
        )

    def draw(self):
        self.draw_map_position()
        self.draw_map_label()

    def get_image(self):
        return self.map_image
