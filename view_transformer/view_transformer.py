import cv2
import numpy as np


class ViewTransformer:
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [100, 700],  # todo: make the script find the pixel values
            [80, 285],
            [1078, 236],
            [1260, 700]
        ], np.float32)

        self.target_vertices = np.array([
            [0, court_width],
            [0,0],
            [court_length, 0],
            [court_length, court_width]
        ], np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                        tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
