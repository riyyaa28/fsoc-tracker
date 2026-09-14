from vision.yolo_detector import YoloBeaconDetector
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGridLayout,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from vision.preprocess import denoise_for_detection
from sim.scene import VirtualScene
from sim.virtual_camera import VirtualPTZCamera
from sim.overlay import draw_crosshair
from vision.classical_detector import detect_beacon_classical, score_candidate
from vision.fusion import fuse_detection
from control.ptz_controller import compute_delta
from vision.kalman_tracker import BeaconKalmanTracker
from disturbance.manager import DisturbanceManager
from logging_.logger import PerformanceLogger


class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSOC Coarse Alignment Simulator")

        # --- pipeline setup ---
        self.scene = VirtualScene(pattern="circular", n_decoys=4)
        _, start_pos = self.scene.render()
        self.camera = VirtualPTZCamera()
        self.camera.pan_x, self.camera.tilt_y = start_pos
        self.screen_center = (self.camera.fov_w // 2, self.camera.fov_h // 2)
        self.detector = YoloBeaconDetector(
            weights_path="beacon_yolo.pt", conf_threshold=0.25
        )
        self.tracker = BeaconKalmanTracker(init_offset=(30, -25))
        self.disturbance_mgr = DisturbanceManager()
        self.logger = PerformanceLogger()
        self.frame_count = 0
        self.DETECT_EVERY_N = 2
        self.last_source = "kalman"

        # --- UI: two video panels side by side, ONE window ---
        self.full_scene_label = QLabel()
        self.crop_label = QLabel()
        video_row = QHBoxLayout()
        video_row.addWidget(self.full_scene_label)
        video_row.addWidget(self.crop_label)

        self.stats_label = QLabel("Starting...")
        self.stats_label.setStyleSheet("font-family: monospace;")

        main_layout = QVBoxLayout()
        main_layout.addLayout(video_row)
        main_layout.addWidget(self.stats_label)
        main_layout.addLayout(self._build_disturbance_panel())
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _build_disturbance_panel(self):
        panel = QGridLayout()
        self.buttons = {}
        for row, category in enumerate(["fog", "noise", "jitter", "rain"]):
            panel.addWidget(QLabel(category.capitalize()), row, 0)
            self.buttons[category] = {}
            for col, level in enumerate([0, 1, 2, 3]):
                text = "Off" if level == 0 else str(level)
                btn = QPushButton(text)
                btn.setCheckable(True)
                btn.setChecked(level == 0)
                btn.clicked.connect(self._make_handler(category, level))
                panel.addWidget(btn, row, col + 1)
                self.buttons[category][level] = btn
        return panel

    def _make_handler(self, category, level):
        def handler():
            self.disturbance_mgr.set_level(category, level)
            for lvl, btn in self.buttons[category].items():
                btn.setChecked(lvl == level)

        return handler

    def _cv_to_qpixmap(self, bgr_frame, scale=1):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if scale != 1:
            pix = pix.scaled(int(w * scale), int(h * scale))
        return pix

    def update_frame(self):
        full_frame, true_pos = self.scene.render()
        full_frame = self.disturbance_mgr.apply_to_frame(full_frame)
        cropped = self.camera.crop(full_frame)

        if self.frame_count % self.DETECT_EVERY_N == 0:
            detection_input = denoise_for_detection(cropped)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            yolo_pos, _ = self.detector.detect(cropped)
            yolo_score = (
                score_candidate(gray, yolo_pos[0], yolo_pos[1])
                if yolo_pos is not None
                else 0.0
            )
            classical_pos, classical_score = detect_beacon_classical(cropped)
            tracked_pos, final_conf, source = fuse_detection(
                yolo_pos,
                yolo_score,
                classical_pos,
                classical_score,
                self.tracker,
                min_score=0.5,
            )
            self.last_source = source
        else:
            tracked_pos, final_conf = self.tracker.update(None)
            source = self.last_source + " (coast)"

        if tracked_pos is not None:
            dx, dy = compute_delta(tracked_pos, self.screen_center)
            self.camera.apply_delta(dx, dy)
            error = (
                (tracked_pos[0] - self.screen_center[0]) ** 2
                + (tracked_pos[1] - self.screen_center[1]) ** 2
            ) ** 0.5
        else:
            error = float("nan")

        dist_m = self.scene.estimate_distance_m()
        self.frame_count += 1

        self.logger.log(
            {
                "fps": 30.0,
                "source": source,
                "confidence": final_conf,
                "error": error,
                "distance_m": dist_m,
            }
        )

        # --- full scene panel ---
        full_display = full_frame.copy()
        cv2.rectangle(
            full_display,
            (
                self.camera.pan_x - self.camera.fov_w // 2,
                self.camera.tilt_y - self.camera.fov_h // 2,
            ),
            (
                self.camera.pan_x + self.camera.fov_w // 2,
                self.camera.tilt_y + self.camera.fov_h // 2,
            ),
            (0, 255, 0),
            2,
        )
        if dist_m is not None:
            cv2.putText(
                full_display,
                f"Dist: {dist_m:.1f}m",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        cv2.putText(
            full_display,
            f"pattern: {self.scene.pattern}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        self.full_scene_label.setPixmap(self._cv_to_qpixmap(full_display, scale=1.3))

        # --- camera crop panel ---
        crop_display = cropped.copy()
        draw_crosshair(crop_display, self.screen_center, color=(0, 255, 0))
        if tracked_pos is not None:
            cv2.circle(crop_display, tracked_pos, 5, (0, 0, 255), 2)
            cv2.line(crop_display, self.screen_center, tracked_pos, (255, 255, 0), 1)
        else:
            cv2.putText(
                crop_display,
                "LOST",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        self.crop_label.setPixmap(self._cv_to_qpixmap(crop_display, scale=1.3))

        dist_str = f"{dist_m:.1f}m" if dist_m is not None else "--"
        self.stats_label.setText(
            f"Source: {source} | Score: {final_conf:.2f} | Dist: {dist_str} | Pattern: {self.scene.pattern}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())
