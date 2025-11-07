import json
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np


@dataclass
class StereoCalibrationParams:
  camera_matrix_left: np.ndarray
  camera_matrix_right: np.ndarray
  dist_coeffs_left: np.ndarray
  dist_coeffs_right: np.ndarray
  rotation_matrix: np.ndarray
  translation_vector: np.ndarray
  essential_matrix: np.ndarray
  fundamental_matrix: np.ndarray
  rectification_transform_left: np.ndarray
  rectification_transform_right: np.ndarray
  projection_matrix_left: np.ndarray
  projection_matrix_right: np.ndarray
  disparity_to_depth_mapping: np.ndarray
  roi_left: Tuple[int, int, int, int]
  roi_right: Tuple[int, int, int, int]


class StereoVisionProcessor:

  def __init__(self, calibration_file: Optional[str] = None):
    self.calibration_params: Optional[StereoCalibrationParams] = None
    self.stereo_matcher = None
    self.wls_filter = None
    self.stereo_bm = None
    self.stereo_sgbm = None

    if calibration_file:
      self.load_calibration(calibration_file)

    self._setup_stereo_matcher()

  def _setup_stereo_matcher(self):
    self.stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    self.stereo_sgbm = cv2.StereoSGBM_create(
      minDisparity=0,
      numDisparities=64,
      blockSize=11,
      P1=8 * 3 * 11 ** 2,
      P2=32 * 3 * 11 ** 2,
      disp12MaxDiff=1,
      uniquenessRatio=15,
      speckleWindowSize=0,
      speckleRange=2,
      preFilterCap=63,
      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_sgbm)
    self.wls_filter.setLambda(8000)
    self.wls_filter.setSigmaColor(1.2)

  def calibrate_stereo_cameras(self,
                               left_images: List[np.ndarray],
                               right_images: List[np.ndarray],
                               chessboard_size: Tuple[int, int] = (9, 6),
                               square_size: float = 1.0) -> StereoCalibrationParams:
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    for left_img, right_img in zip(left_images, right_images):
      gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
      gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

      ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
      ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

      if ret_left and ret_right:
        objpoints.append(objp)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

    if len(objpoints) < 10:
      raise ValueError(
        "No se encontraron suficientes imágenes válidas para la calibración. Se necesitan al menos 10 pares.")

    image_size = gray_left.shape[::-1]

    ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
      objpoints, imgpoints_left, image_size, None, None)

    ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
      objpoints, imgpoints_right, image_size, None, None)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
      objpoints, imgpoints_left, imgpoints_right,
      mtx_left, dist_left, mtx_right, dist_right,
      image_size, criteria=criteria_stereo, flags=flags)

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
      mtx_left, dist_left, mtx_right, dist_right,
      image_size, R, T, alpha=0)

    calibration_params = StereoCalibrationParams(
      camera_matrix_left=mtx_left,
      camera_matrix_right=mtx_right,
      dist_coeffs_left=dist_left,
      dist_coeffs_right=dist_right,
      rotation_matrix=R,
      translation_vector=T,
      essential_matrix=E,
      fundamental_matrix=F,
      rectification_transform_left=R1,
      rectification_transform_right=R2,
      projection_matrix_left=P1,
      projection_matrix_right=P2,
      disparity_to_depth_mapping=Q,
      roi_left=roi_left,
      roi_right=roi_right
    )

    self.calibration_params = calibration_params
    return calibration_params

  def save_calibration(self, filepath: str) -> None:
    if not self.calibration_params:
      raise ValueError("No hay parámetros de calibración para guardar")

    calibration_data = {
      'camera_matrix_left': self.calibration_params.camera_matrix_left.tolist(),
      'camera_matrix_right': self.calibration_params.camera_matrix_right.tolist(),
      'dist_coeffs_left': self.calibration_params.dist_coeffs_left.tolist(),
      'dist_coeffs_right': self.calibration_params.dist_coeffs_right.tolist(),
      'rotation_matrix': self.calibration_params.rotation_matrix.tolist(),
      'translation_vector': self.calibration_params.translation_vector.tolist(),
      'essential_matrix': self.calibration_params.essential_matrix.tolist(),
      'fundamental_matrix': self.calibration_params.fundamental_matrix.tolist(),
      'rectification_transform_left': self.calibration_params.rectification_transform_left.tolist(),
      'rectification_transform_right': self.calibration_params.rectification_transform_right.tolist(),
      'projection_matrix_left': self.calibration_params.projection_matrix_left.tolist(),
      'projection_matrix_right': self.calibration_params.projection_matrix_right.tolist(),
      'disparity_to_depth_mapping': self.calibration_params.disparity_to_depth_mapping.tolist(),
      'roi_left': self.calibration_params.roi_left,
      'roi_right': self.calibration_params.roi_right
    }

    with open(filepath, 'w') as f:
      json.dump(calibration_data, f, indent=2)

  def load_calibration(self, filepath: str) -> None:
    with open(filepath, 'r') as f:
      calibration_data = json.load(f)

    self.calibration_params = StereoCalibrationParams(
      camera_matrix_left=np.array(calibration_data['camera_matrix_left']),
      camera_matrix_right=np.array(calibration_data['camera_matrix_right']),
      dist_coeffs_left=np.array(calibration_data['dist_coeffs_left']),
      dist_coeffs_right=np.array(calibration_data['dist_coeffs_right']),
      rotation_matrix=np.array(calibration_data['rotation_matrix']),
      translation_vector=np.array(calibration_data['translation_vector']),
      essential_matrix=np.array(calibration_data['essential_matrix']),
      fundamental_matrix=np.array(calibration_data['fundamental_matrix']),
      rectification_transform_left=np.array(calibration_data['rectification_transform_left']),
      rectification_transform_right=np.array(calibration_data['rectification_transform_right']),
      projection_matrix_left=np.array(calibration_data['projection_matrix_left']),
      projection_matrix_right=np.array(calibration_data['projection_matrix_right']),
      disparity_to_depth_mapping=np.array(calibration_data['disparity_to_depth_mapping']),
      roi_left=tuple(calibration_data['roi_left']),
      roi_right=tuple(calibration_data['roi_right'])
    )

  def rectify_stereo_pair(self,
                          left_image: np.ndarray,
                          right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not self.calibration_params:
      raise ValueError("Las cámaras estéreo deben estar calibradas primero")

    map1_left, map2_left = cv2.initUndistortRectifyMap(
      self.calibration_params.camera_matrix_left,
      self.calibration_params.dist_coeffs_left,
      self.calibration_params.rectification_transform_left,
      self.calibration_params.projection_matrix_left,
      left_image.shape[:2][::-1],
      cv2.CV_32FC1
    )

    map1_right, map2_right = cv2.initUndistortRectifyMap(
      self.calibration_params.camera_matrix_right,
      self.calibration_params.dist_coeffs_right,
      self.calibration_params.rectification_transform_right,
      self.calibration_params.projection_matrix_right,
      right_image.shape[:2][::-1],
      cv2.CV_32FC1
    )

    rectified_left = cv2.remap(left_image, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map1_right, map2_right, cv2.INTER_LINEAR)

    return rectified_left, rectified_right

  def compute_disparity_map(self,
                            left_image: np.ndarray,
                            right_image: np.ndarray,
                            method: str = 'sgbm',
                            use_wls_filter: bool = True) -> np.ndarray:
    if len(left_image.shape) == 3:
      left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
      right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
      left_gray = left_image
      right_gray = right_image

    if method == 'bm':
      disparity = self.stereo_bm.compute(left_gray, right_gray)
    elif method == 'sgbm':
      disparity = self.stereo_sgbm.compute(left_gray, right_gray)
    else:
      raise ValueError("El método debe ser 'bm' o 'sgbm'")

    if use_wls_filter and method == 'sgbm':
      right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_sgbm)
      disparity_right = right_matcher.compute(right_gray, left_gray)
      disparity = self.wls_filter.filter(disparity, left_gray, None, disparity_right)

    return disparity

  def disparity_to_depth(self, disparity_map: np.ndarray) -> np.ndarray:
    if not self.calibration_params:
      raise ValueError("Las cámaras estéreo deben estar calibradas primero")

    disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
    depth_map = cv2.reprojectImageTo3D(disparity_map, self.calibration_params.disparity_to_depth_mapping)
    depth = depth_map[:, :, 2]
    depth = np.where((depth > 0) & (depth < 10000), depth, 0)

    return depth

  def get_3d_coordinates(self,
                         disparity_map: np.ndarray,
                         x: int,
                         y: int) -> Optional[Tuple[float, float, float]]:
    if not self.calibration_params:
      return None

    disparity_value = disparity_map[y, x]
    if disparity_value <= 0:
      return None

    Q = self.calibration_params.disparity_to_depth_mapping
    point_2d = np.array([x, y, disparity_value, 1.0])
    point_3d = Q.dot(point_2d)

    if point_3d[3] != 0:
      X = point_3d[0] / point_3d[3]
      Y = point_3d[1] / point_3d[3]
      Z = point_3d[2] / point_3d[3]
      return X, Y, Z

    return None

  def calculate_3d_distance(self,
                            disparity_map: np.ndarray,
                            point1: Tuple[int, int],
                            point2: Tuple[int, int]) -> Optional[float]:
    coords1 = self.get_3d_coordinates(disparity_map, point1[0], point1[1])
    coords2 = self.get_3d_coordinates(disparity_map, point2[0], point2[1])

    if coords1 and coords2:
      distance = np.sqrt(
        (coords1[0] - coords2[0]) ** 2 +
        (coords1[1] - coords2[1]) ** 2 +
        (coords1[2] - coords2[2]) ** 2
      )
      return distance

    return None

  def visualize_disparity(self, disparity_map: np.ndarray) -> np.ndarray:
    disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    return disparity_colored
