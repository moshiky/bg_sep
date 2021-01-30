
import numpy as np


class PixelManager:

    def __init__(self, height: int, width: int, sample_size: int):

        self._frame_bank = np.zeros((sample_size, int(height), int(width), 3), dtype=np.float32)
        self._next_idx = 0

        self._centroid_bank = None
        self._max_dist = None

    @staticmethod
    def _normalize_frame(frame):
        cn_min = frame.min(axis=0).min(axis=0)
        cn_max = frame.max(axis=0).max(axis=0)
        cn_range = cn_max - cn_min

        return (frame - cn_min) / cn_range

    def get_centroid(self):
        return (self._centroid_bank * np.array([255, 255, 255])).astype(np.uint8)

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

        self._centroid_bank = self._frame_bank.mean(axis=0)

        dist_arr = self._get_dist(self._frame_bank, self._centroid_bank)
        self._max_dist = dist_arr.max(axis=0)

    def get_mask(self, frame):

        # calculate diff for each pixel
        dist_arr = self._get_dist(self._normalize_frame(frame), self._centroid_bank)

        # convert to color mask
        output_mask = np.zeros_like(frame)
        output_mask[:, :, 2] = np.round(255 * (dist_arr > self._max_dist * 4).astype(int)).astype(frame.dtype)

        return output_mask
