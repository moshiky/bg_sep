
import numpy as np


class PixelManager:

    def __init__(self, height: int, width: int, sample_size: int):

        self._frame_bank = np.zeros((sample_size, int(height), int(width), 3), dtype=np.uint8)
        self._next_idx = 0

        self._centroid_bank = None
        self._max_dist = None

    def add_frame(self, frame: np.array):

        if self._next_idx < self._frame_bank.shape[0]:
            self._frame_bank[self._next_idx, :, :, :] = frame
            self._next_idx += 1

            return True

        else:
            return False

    @staticmethod
    def _get_dist(im0: np.array, im1: np.array):
        return np.log(np.sqrt(np.power(im0 - im1, 2).sum(axis=-1)))

    def calculate_stats(self):

        self._centroid_bank = self._frame_bank.mean(axis=0)

        dist_arr = self._get_dist(self._frame_bank, self._centroid_bank)
        self._max_dist = dist_arr.max(axis=0)

    def get_mask(self, frame):

        # calculate diff for each pixel
        dist_arr = self._get_dist(frame, self._centroid_bank)
        max_dist = dist_arr.max()
        dist_arr /= max_dist

        # convert to color mask
        output_mask = np.zeros_like(frame)
        output_mask[:, :, 2] = np.round(255 * dist_arr).astype(frame.dtype)

        return output_mask
