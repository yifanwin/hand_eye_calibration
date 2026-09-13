from __future__ import annotations

import cv2
import numpy as np

from ..interfaces import TargetDetector
from ..models import ColorProjection, DetectionResult


class ChessboardDetector(TargetDetector):
    def __init__(self, columns: int = 11, rows: int = 8, square_size_m: float = 0.01):
        if columns < 2 or rows < 2 or square_size_m <= 0:
            raise ValueError("invalid chessboard geometry")
        self.pattern_size = (columns, rows)
        self.square_size_m = square_size_m
        points = np.zeros((columns * rows, 3), dtype=np.float64)
        points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        self.object_points = points * square_size_m

    def detect(self, color_bgr, color_projection: ColorProjection) -> DetectionResult | None:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        success, rvec, tvec = cv2.solvePnP(
            self.object_points, corners, color_projection.camera_matrix,
            color_projection.distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        T_camera_target = np.eye(4, dtype=np.float64)
        T_camera_target[:3, :3] = rotation
        T_camera_target[:3, 3] = tvec.reshape(3)
        projected, _ = cv2.projectPoints(
            self.object_points, rvec, tvec, color_projection.camera_matrix,
            color_projection.distortion_coeffs,
        )
        errors = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        debug = color_bgr.copy()
        cv2.drawChessboardCorners(debug, self.pattern_size, corners, True)
        cv2.drawFrameAxes(
            debug, color_projection.camera_matrix, color_projection.distortion_coeffs,
            rvec, tvec, 5 * self.square_size_m,
        )
        return DetectionResult(
            T_camera_target=T_camera_target,
            image_points_px=corners.reshape(-1, 2),
            reprojection_rmse_px=rmse,
            debug_image_bgr=debug,
        )

