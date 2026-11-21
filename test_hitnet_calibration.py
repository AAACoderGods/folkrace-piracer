import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from depth_camera import DepthCamera
from test_hitnet_debug import _fit_calibration


class HitNetCalibrationTests(unittest.TestCase):
    def test_fit_recovers_scale_and_offset(self):
        scale, offset = 24000.0, 1.5
        samples = [(distance, scale / distance + offset)
                   for distance in (400.0, 600.0, 900.0, 1400.0)]

        fitted_scale, fitted_offset, r_squared = _fit_calibration(samples)

        self.assertAlmostEqual(fitted_scale, scale, places=7)
        self.assertAlmostEqual(fitted_offset, offset, places=7)
        self.assertAlmostEqual(r_squared, 1.0, places=7)

    def test_fit_requires_three_distances(self):
        with self.assertRaises(ValueError):
            _fit_calibration([(500.0, 40.0), (1000.0, 20.0)])

    def test_applies_hitnet_offset_and_model_resolution_scale(self):
        cam = DepthCamera.__new__(DepthCamera)
        cam.config = SimpleNamespace(
            stereo_swap_eyes=False,
            stereo_median_blur=0,
            stereo_fast_stereonet_offset_px=1.0,
        )
        cam._using_fast_stereonet = True
        cam._fast_stereonet_input_width = 4
        cam._disparity_from_fast_stereonet = lambda left, right: np.full(
            (2, 2), 5.0, dtype=np.float32)

        depth = cam._stereo_frame_to_depth(np.zeros((8, 16, 3), dtype=np.uint8), 1000.0)

        np.testing.assert_allclose(depth, 125.0)

    def test_sgbm_fallback_keeps_native_scale_when_hitnet_fails(self):
        class Matcher:
            def compute(self, left, right):
                return np.full(left.shape, 1600, dtype=np.int16)

        cam = DepthCamera.__new__(DepthCamera)
        cam.config = SimpleNamespace(
            stereo_swap_eyes=False,
            stereo_median_blur=0,
            stereo_fast_stereonet_offset_px=1.0,
            stereo_clahe_clip_limit=0.0,
            stereo_num_disparities=16,
            stereo_block_size=5,
            stereo_pre_filter_cap=31,
            stereo_use_sgbm=True,
            stereo_min_disparity=0,
            stereo_disp12_max_diff=0,
            stereo_uniqueness_ratio=0,
            stereo_speckle_window_size=0,
            stereo_speckle_range=0,
            stereo_sgbm_mode="3way",
        )
        cam._using_fast_stereonet = True
        cam._fast_stereonet_input_width = 4
        cam._disparity_from_fast_stereonet = lambda left, right: None

        with patch.object(cv2, "StereoSGBM_create", return_value=Matcher()):
            depth = cam._stereo_frame_to_depth(
                np.zeros((8, 16, 3), dtype=np.uint8), 1000.0)

        np.testing.assert_allclose(depth, 10.0)


if __name__ == "__main__":
    unittest.main()
