import cv2
import numpy as np


class BeaconKalmanTracker:
    def __init__(self, init_offset=(0, 0), max_coast_frames=45):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.init_offset = init_offset
        self.initialized = False
        self.max_coast_frames = max_coast_frames
        self.coast_count = 0

    def _confidence_from_covariance(self):
        trace = np.trace(self.kf.errorCovPost[:2, :2])
        confidence = 1.0 / (1.0 + trace / 50.0)
        return float(np.clip(confidence, 0.0, 1.0))

    def update(self, measurement):
        if not self.initialized and measurement is not None:
            x, y = measurement
            ox, oy = self.init_offset
            self.kf.statePre = np.array([[x + ox], [y + oy], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x + ox], [y + oy], [0], [0]], np.float32)
            self.initialized = True
            self.coast_count = 0

        if not self.initialized:
            return None, 0.0

        prediction = self.kf.predict()
        if measurement is not None:
            corrected = self.kf.correct(
                np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
            )
            pos = (int(corrected[0][0]), int(corrected[1][0]))
            self.coast_count = 0
        else:
            self.coast_count += 1
            if self.coast_count > self.max_coast_frames:
                return None, 0.0
            pos = (int(prediction[0][0]), int(prediction[1][0]))

        return pos, self._confidence_from_covariance()
