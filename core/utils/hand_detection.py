import math
from typing import List, Tuple, Dict, Optional, Any

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
  def __init__(self) -> None:
    self.mp_hands = mp.solutions.hands
    self.hands = self.mp_hands.Hands(
      static_image_mode=False,
      max_num_hands=2,
      min_detection_confidence=0.7,
      min_tracking_confidence=0.5
    )
    self.mp_drawing = mp.solutions.drawing_utils
    self.smoothing_factor = 0.3
    self.smoothed_landmarks: Dict = {}

  def detect_hands(self, frame: np.ndarray) -> List[Dict]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.hands.process(rgb_frame)

    hands_data = []
    if results.multi_hand_landmarks:
      for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hand_id = results.multi_handedness[idx].classification[0].index
        hand_data = {
          'landmarks': hand_landmarks,
          'handedness': results.multi_handedness[idx].classification[0].label,
          'points': self._extract_key_points(hand_id, hand_landmarks, frame.shape)
        }
        hands_data.append(hand_data)

    return hands_data

  def _extract_key_points(self, hand_id: int, landmarks: Any, frame_shape: Tuple[int, ...]) -> Dict:
    height, width = frame_shape[:2]

    raw_points = {
      'thumb': (landmarks.landmark[4].x * width, landmarks.landmark[4].y * height),
      'index': (landmarks.landmark[8].x * width, landmarks.landmark[8].y * height),
      'middle': (landmarks.landmark[12].x * width, landmarks.landmark[12].y * height),
      'ring': (landmarks.landmark[16].x * width, landmarks.landmark[16].y * height),
      'pinky': (landmarks.landmark[20].x * width, landmarks.landmark[20].y * height)
    }

    palm_center = (
      landmarks.landmark[9].x * width,
      landmarks.landmark[9].y * height
    )

    all_landmarks = [(lm.x * width, lm.y * height) for lm in landmarks.landmark]

    wrist_pos = all_landmarks[0]
    mcp_pos = all_landmarks[9]
    ref_distance = math.sqrt((wrist_pos[0] - mcp_pos[0]) ** 2 + (wrist_pos[1] - mcp_pos[1]) ** 2)

    if hand_id not in self.smoothed_landmarks:
      self.smoothed_landmarks[hand_id] = {
        'fingertips': raw_points,
        'palm_center': palm_center,
        'all_landmarks': all_landmarks,
        'ref_distance': ref_distance
      }
    else:
      self._update_smoothed_landmarks(hand_id, raw_points, palm_center, all_landmarks, ref_distance)

    return self.smoothed_landmarks[hand_id]

  def _update_smoothed_landmarks(self, hand_id: int, raw_points: Dict, palm_center: Tuple[float, float],
                                 all_landmarks: List[Tuple[float, float]], ref_distance: float) -> None:
    for finger, pos in raw_points.items():
      prev_pos = self.smoothed_landmarks[hand_id]['fingertips'][finger]
      smoothed_pos = (
        self.smoothing_factor * pos[0] + (1 - self.smoothing_factor) * prev_pos[0],
        self.smoothing_factor * pos[1] + (1 - self.smoothing_factor) * prev_pos[1]
      )
      self.smoothed_landmarks[hand_id]['fingertips'][finger] = smoothed_pos

    prev_palm = self.smoothed_landmarks[hand_id]['palm_center']
    self.smoothed_landmarks[hand_id]['palm_center'] = (
      self.smoothing_factor * palm_center[0] + (1 - self.smoothing_factor) * prev_palm[0],
      self.smoothing_factor * palm_center[1] + (1 - self.smoothing_factor) * prev_palm[1]
    )

    for i, lm in enumerate(all_landmarks):
      prev_lm = self.smoothed_landmarks[hand_id]['all_landmarks'][i]
      self.smoothed_landmarks[hand_id]['all_landmarks'][i] = (
        self.smoothing_factor * lm[0] + (1 - self.smoothing_factor) * prev_lm[0],
        self.smoothing_factor * lm[1] + (1 - self.smoothing_factor) * prev_lm[1]
      )

    prev_ref = self.smoothed_landmarks[hand_id]['ref_distance']
    self.smoothed_landmarks[hand_id]['ref_distance'] = (
      self.smoothing_factor * ref_distance + (1 - self.smoothing_factor) * prev_ref
    )

  def _update_smoothed_landmarks(self, hand_id: int, raw_points: Dict, palm_center: Tuple[float, float],
                                 all_landmarks: List[Tuple[float, float]], ref_distance: float) -> None:
    for finger, pos in raw_points.items():
      prev_pos = self.smoothed_landmarks[hand_id]['fingertips'][finger]
      smoothed_pos = (
        self.smoothing_factor * pos[0] + (1 - self.smoothing_factor) * prev_pos[0],
        self.smoothing_factor * pos[1] + (1 - self.smoothing_factor) * prev_pos[1]
      )
      self.smoothed_landmarks[hand_id]['fingertips'][finger] = smoothed_pos

    prev_palm = self.smoothed_landmarks[hand_id]['palm_center']
    self.smoothed_landmarks[hand_id]['palm_center'] = (
      self.smoothing_factor * palm_center[0] + (1 - self.smoothing_factor) * prev_palm[0],
      self.smoothing_factor * palm_center[1] + (1 - self.smoothing_factor) * prev_palm[1]
    )

    for i, lm in enumerate(all_landmarks):
      prev_lm = self.smoothed_landmarks[hand_id]['all_landmarks'][i]
      self.smoothed_landmarks[hand_id]['all_landmarks'][i] = (
        self.smoothing_factor * lm[0] + (1 - self.smoothing_factor) * prev_lm[0],
        self.smoothing_factor * lm[1] + (1 - self.smoothing_factor) * prev_lm[1]
      )

    prev_ref = self.smoothed_landmarks[hand_id]['ref_distance']
    self.smoothed_landmarks[hand_id]['ref_distance'] = (
      self.smoothing_factor * ref_distance + (1 - self.smoothing_factor) * prev_ref
    )

  def detect_pinch_gesture(self, hand_data: Dict) -> Optional[Tuple[int, int]]:
    fingertips = hand_data['points']['fingertips']
    ref_distance = hand_data['points']['ref_distance']

    thumb_pos = fingertips['thumb']
    index_pos = fingertips['index']

    distance = math.sqrt(
      (thumb_pos[0] - index_pos[0]) ** 2 +
      (thumb_pos[1] - index_pos[1]) ** 2
    )

    if distance < 0.3 * ref_distance:
      pinch_x = int((thumb_pos[0] + index_pos[0]) / 2)
      pinch_y = int((thumb_pos[1] + index_pos[1]) / 2)
      return pinch_x, pinch_y

    return None

  def detect_two_finger_span(self, hand_data: Dict) -> Optional[Dict]:
    fingertips = hand_data['points']['fingertips']
    ref_distance = hand_data['points']['ref_distance']

    thumb_pos = fingertips['thumb']
    index_pos = fingertips['index']

    distance = math.sqrt(
      (thumb_pos[0] - index_pos[0]) ** 2 +
      (thumb_pos[1] - index_pos[1]) ** 2
    )

    if distance > 0.5 * ref_distance:
      return {
        'point1': thumb_pos,
        'point2': index_pos,
        'distance': distance,
        'midpoint': (
          int((thumb_pos[0] + index_pos[0]) / 2),
          int((thumb_pos[1] + index_pos[1]) / 2)
        )
      }

    return None

  def draw_hand_landmarks(self, frame: np.ndarray, hands_data: List[Dict]) -> np.ndarray:
    for hand_data in hands_data:
      self.mp_drawing.draw_landmarks(
        frame,
        hand_data['landmarks'],
        self.mp_hands.HAND_CONNECTIONS
      )

      for finger, pos in hand_data['points']['fingertips'].items():
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 8, (0, 255, 0), -1)
        cv2.putText(frame, finger[0].upper(),
                    (int(pos[0]) + 10, int(pos[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


class GestureController:
  def __init__(self):
    self.hand_detector = HandDetector()
    self.calibration_mode = False
    self.measurement_mode = False
    self.calibration_points = []
    self.measurement_points = []
    self.pinch_threshold = 30
    self.span_threshold = 50

  def process_frame(self, frame: np.ndarray) -> Dict:
    hands_data = self.hand_detector.detect_hands(frame)

    result = {
      'frame': frame.copy(),
      'hands_detected': len(hands_data),
      'hands_data': hands_data,
      'gestures': [],
      'calibration_data': None,
      'measurement_data': None
    }

    if hands_data:
      result['frame'] = self.hand_detector.draw_hand_landmarks(result['frame'], hands_data)

    for hand_data in hands_data:
      pinch_point = self.hand_detector.detect_pinch_gesture(hand_data)
      if pinch_point:
        result['gestures'].append({
          'type': 'pinch',
          'point': pinch_point,
          'hand': hand_data['handedness']
        })

      span_data = self.hand_detector.detect_two_finger_span(hand_data)
      if span_data:
        result['gestures'].append({
          'type': 'span',
          'data': span_data,
          'hand': hand_data['handedness']
        })

    return result

  def handle_calibration_gesture(self, gesture_data: Dict) -> bool:
    if gesture_data['type'] == 'span':
      span_data = gesture_data['data']

      self.calibration_points.append({
        'point1': span_data['point1'],
        'point2': span_data['point2'],
        'distance': span_data['distance'],
        'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
      })

      if len(self.calibration_points) > 2:
        self.calibration_points = self.calibration_points[-2:]

      return True

    return False

  def handle_measurement_gesture(self, gesture_data: Dict) -> bool:
    if gesture_data['type'] == 'pinch':
      pinch_point = gesture_data['point']

      self.measurement_points.append({
        'point': pinch_point,
        'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
      })

      if len(self.measurement_points) > 2:
        self.measurement_points = self.measurement_points[-2:]

      return True

    return False

  def get_calibration_distance(self) -> Optional[float]:
    if len(self.calibration_points) >= 1:
      return self.calibration_points[-1]['distance']
    return None

  def get_measurement_points(self) -> List[Tuple[int, int]]:
    return [point['point'] for point in self.measurement_points]

  def clear_calibration(self) -> None:
    self.calibration_points = []

  def clear_measurements(self) -> None:
    self.measurement_points = []

  def set_calibration_mode(self, enabled: bool) -> None:
    self.calibration_mode = enabled
    if enabled:
      self.clear_calibration()

  def set_measurement_mode(self, enabled: bool) -> None:
    self.measurement_mode = enabled
    if enabled:
      self.clear_measurements()


class AutoCalibrator:
  def __init__(self) -> None:
    self.gesture_controller = GestureController()
    self.calibrated = False
    self.pixels_per_cm: Optional[float] = None
    self.reference_distance_cm: Optional[float] = None

  def calibrate_with_fingers(self, frame: np.ndarray, known_distance_cm: float) -> Dict:
    result = self.gesture_controller.process_frame(frame)

    for gesture in result['gestures']:
      if gesture['type'] == 'span':
        span_data = gesture['data']
        pixel_distance = span_data['distance']

        hand_ref_distance = None
        if result.get('hands_data') and len(result['hands_data']) > 0:
          hand_ref_distance = result['hands_data'][0]['points']['ref_distance']

        self.pixels_per_cm = pixel_distance / known_distance_cm
        self.reference_distance_cm = known_distance_cm
        self.calibrated = True

        result['calibration_success'] = True
        result['pixels_per_cm'] = self.pixels_per_cm
        result['calibration_distance'] = pixel_distance
        result['calibration_hand_ref_distance'] = hand_ref_distance

        return result

    result['calibration_success'] = False
    return result

  def convert_pixels_to_cm(self, pixel_distance: float) -> float:
    if self.calibrated and self.pixels_per_cm:
      return pixel_distance / self.pixels_per_cm
    return pixel_distance

  def is_calibrated(self) -> bool:
    return self.calibrated

  def get_calibration_info(self) -> Dict:
    return {
      'calibrated': self.calibrated,
      'pixels_per_cm': self.pixels_per_cm,
      'reference_distance_cm': self.reference_distance_cm
    }
