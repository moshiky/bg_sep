import cv2
import numpy as np


class PixelManager:

    def __init__(self, height: int, width: int, sample_size: int):

        self._frame_bank = np.zeros((sample_size, int(height), int(width), 3), dtype=np.float64)
        self._next_idx = 0

        self._centroid_img = None
        self._max_dist = None

    @staticmethod
    def _normalize_frame(frame):
        return frame.astype(np.float64)

    def get_centroid(self):
        return self._centroid_img.round().astype(np.uint8)

    def add_frame(self, frame: np.array):

        if self._next_idx < self._frame_bank.shape[0]:
            self._frame_bank[self._next_idx, :, :, :] = self._normalize_frame(frame)
            self._next_idx += 1

            return True

        else:
            return False

    @staticmethod
    def _get_dist(im0: np.array, im1: np.array):
        return np.sqrt(np.power(im0 - im1, 2).sum(axis=-1))

    def calculate_stats(self):

        centroid_img_base = self._frame_bank.mean(axis=0)

        self._centroid_img = centroid_img_base.copy()

        self._centroid_img += cv2.blur(centroid_img_base, (3, 3))
        self._centroid_img += cv2.blur(centroid_img_base, (5, 5))
        self._centroid_img += cv2.blur(centroid_img_base, (9, 9))
        self._centroid_img += cv2.blur(centroid_img_base, (15, 15))
        self._centroid_img += cv2.blur(centroid_img_base, (27, 27))
        self._centroid_img += cv2.blur(centroid_img_base, (41, 41))

        self._centroid_img /= 7

        dist_arr = self._get_dist(self._frame_bank, self._centroid_img)
        self._max_dist = dist_arr.max(axis=0)

    def get_mask(self, frame):

        # calculate diff for each pixel
        norm_frame = self._normalize_frame(frame)
        dist_arr = self._get_dist(norm_frame, self._centroid_img)

        # convert to color mask
        output_mask = np.zeros_like(frame)
        output_mask[:, :, 2] = np.round(255 * ((dist_arr - self._max_dist) > 60)).astype(frame.dtype)

        return output_mask
