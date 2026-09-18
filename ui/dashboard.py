from vision.yolo_detector import YoloBeaconDetector  # must load before PyQt5
import sys
import cv2
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                              QHBoxLayout, QPushButton, QGridLayout, QComboBox,
                              QSlider, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from sim.scene import VirtualScene
from sim.virtual_camera import VirtualPTZCamera
from sim.overlay import draw_crosshair
from vision.classical_detector import detect_beacon_classical, score_candidate
from vision.fusion import fuse_detection
from vision.preprocess import denoise_for_detection
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
        self.detector = YoloBeaconDetector(weights_path="beacon_yolo.pt", conf_threshold=0.25)
        self.tracker = BeaconKalmanTracker(init_offset=(30, -25))
        self.disturbance_mgr = DisturbanceManager()
        self.logger = PerformanceLogger()
        self.frame_count = 0
        self.DETECT_EVERY_N = 2
        self.last_source = "kalman"

        # --- reacquisition state ---
        self.search_state = "LOCKED"   # LOCKED | SEEKING | SEARCHING
        self.last_known_world_pos = (float(self.camera.pan_x), float(self.camera.tilt_y))
        self.search_angle = 0.0
        self.search_radius = 0.0
        self.lost_box_size = 12.0
        self.LOST_BOX_MAX = 40.0

        # --- UI ---
        self.full_scene_label = QLabel()
        self.full_scene_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.full_scene_label.setMinimumSize(1, 1)

        self.crop_label = QLabel()
        self.crop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.crop_label.setMinimumSize(1, 1)

        video_row = QHBoxLayout()
        video_row.addWidget(self.full_scene_label, 2)
        video_row.addWidget(self.crop_label, 1)

        self.stats_label = QLabel("Starting...")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 13px;")

        main_layout = QVBoxLayout()
        main_layout.addLayout(video_row, 8)
        main_layout.addWidget(self.stats_label)
        main_layout.addLayout(self._build_controls_panel())
        main_layout.addWidget(QLabel("Set Intensity"))
        main_layout.addLayout(self._build_disturbance_panel())
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ---------------- UI builders ----------------

    def _build_controls_panel(self):
        row = QHBoxLayout()

        row.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Auto", "circular", "figure8", "straight", "random"])
        self.pattern_combo.setCurrentText("circular")
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_changed)
        row.addWidget(self.pattern_combo)

        row.addWidget(QLabel("Decoys:"))
        self.decoy_slider = QSlider(Qt.Horizontal)
        self.decoy_slider.setMinimum(0)
        self.decoy_slider.setMaximum(10)
        self.decoy_slider.setValue(self.scene.n_decoys)
        self.decoy_slider.valueChanged.connect(self._on_decoy_count_changed)
        row.addWidget(self.decoy_slider)
        self.decoy_count_label = QLabel(str(self.scene.n_decoys))
        row.addWidget(self.decoy_count_label)

        return row

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
                btn.clicked.connect(self._make_disturbance_handler(category, level))
                panel.addWidget(btn, row, col + 1)
                self.buttons[category][level] = btn
        return panel

    def _make_disturbance_handler(self, category, level):
        def handler():
            self.disturbance_mgr.set_level(category, level)
            for lvl, btn in self.buttons[category].items():
                btn.setChecked(lvl == level)
        return handler

    def _on_pattern_changed(self, text):
        if text == "Auto":
            self.scene.enable_auto_rotate()
        else:
            self.scene.disable_auto_rotate()
            self.scene.set_pattern(text)

    def _on_decoy_count_changed(self, value):
        self.decoy_count_label.setText(str(value))
        self.scene.regenerate_decoys(n_decoys=value)

    def _cv_to_qpixmap(self, bgr_frame, target_label=None):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        if target_label is not None:
            size = target_label.size()
            if size.width() > 0 and size.height() > 0:
                pix = pix.scaled(size.width(), size.height(),
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    # ---------------- main loop ----------------

    def update_frame(self):
        full_frame, true_pos = self.scene.render()
        full_frame = self.disturbance_mgr.apply_to_frame(full_frame)
        cropped = self.camera.crop(full_frame)

        if self.frame_count % self.DETECT_EVERY_N == 0:
            detection_input = denoise_for_detection(cropped)
            gray = cv2.cvtColor(detection_input, cv2.COLOR_BGR2GRAY)
            yolo_pos, _ = self.detector.detect(detection_input)
            yolo_score = score_candidate(gray, yolo_pos[0], yolo_pos[1]) if yolo_pos is not None else 0.0
            classical_pos, classical_score = detect_beacon_classical(detection_input)
            tracked_pos, final_conf, source = fuse_detection(
                yolo_pos, yolo_score, classical_pos, classical_score, self.tracker, min_score=0.5
            )
            self.last_source = source
        else:
            tracked_pos, final_conf = self.tracker.update(None)
            source = self.last_source + " (coast)"

        # ---------------- lock / seek / search state machine ----------------
        if tracked_pos is not None:
            self.search_state = "LOCKED"
            self.lost_box_size = 12.0
            self.search_radius = 0.0
            self.last_known_world_pos = (
                self.camera.pan_x + (tracked_pos[0] - self.screen_center[0]),
                self.camera.tilt_y + (tracked_pos[1] - self.screen_center[1]),
            )
            dx, dy = compute_delta(tracked_pos, self.screen_center)
            self.camera.apply_delta(dx, dy)
            error = ((tracked_pos[0] - self.screen_center[0]) ** 2 +
                     (tracked_pos[1] - self.screen_center[1]) ** 2) ** 0.5
        else:
            self.lost_box_size = min(self.lost_box_size + 1.0, self.LOST_BOX_MAX)
            tx, ty = self.last_known_world_pos
            dist_to_target = math.hypot(self.camera.pan_x - tx, self.camera.tilt_y - ty)
            if dist_to_target > 8:
                self.search_state = "SEEKING"
                dx = (tx - self.camera.pan_x) * 0.4
                dy = (ty - self.camera.tilt_y) * 0.4
            else:
                self.search_state = "SEARCHING"
                self.search_angle += 0.15
                self.search_radius = min(self.search_radius + 0.8, 60)
                sx = tx + self.search_radius * math.cos(self.search_angle)
                sy = ty + self.search_radius * math.sin(self.search_angle)
                dx = (sx - self.camera.pan_x) * 0.5
                dy = (sy - self.camera.tilt_y) * 0.5
            self.camera.apply_delta(dx, dy)
            error = float("nan")

        dist_m = self.scene.estimate_distance_m()
        self.frame_count += 1

        self.logger.log({"fps": 30.0, "source": source, "confidence": final_conf,
                          "error": error, "distance_m": dist_m})

        # ---------------- display: full scene ----------------
        full_display = full_frame.copy()
        fov_color = (0, 255, 0) if self.search_state == "LOCKED" else (0, 0, 255)
        cv2.rectangle(full_display,
                      (self.camera.pan_x - self.camera.fov_w // 2, self.camera.tilt_y - self.camera.fov_h // 2),
                      (self.camera.pan_x + self.camera.fov_w // 2, self.camera.tilt_y + self.camera.fov_h // 2),
                      fov_color, 2)
        if dist_m is not None:
            cv2.putText(full_display, f"Dist: {dist_m:.1f}m", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(full_display, f"pattern: {self.scene.pattern}  |  state: {self.search_state}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self.full_scene_label.setPixmap(self._cv_to_qpixmap(full_display, target_label=self.full_scene_label))

        # ---------------- display: camera crop ----------------
        crop_display = cropped.copy()
        if self.search_state == "LOCKED":
            draw_crosshair(crop_display, self.screen_center, color=(0, 255, 0))
            cv2.circle(crop_display, tracked_pos, 5, (0, 0, 255), 2)
            cv2.line(crop_display, self.screen_center, tracked_pos, (255, 255, 0), 1)
            cv2.putText(crop_display, f"Confidence: {final_conf:.2f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(crop_display, f"Source: {source}", (5, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        else:
            size = int(self.lost_box_size)
            cx, cy = self.screen_center
            cv2.rectangle(crop_display, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
            cv2.putText(crop_display, self.search_state, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(crop_display, f"Confidence: {final_conf:.2f}", (5, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        self.crop_label.setPixmap(self._cv_to_qpixmap(crop_display, target_label=self.crop_label))

        dist_str = f"{dist_m:.1f}m" if dist_m is not None else "--"
        self.stats_label.setText(
            f"Source: {source} | Confidence: {final_conf:.2f} | Dist: {dist_str} | "
            f"Pattern: {self.scene.pattern} | State: {self.search_state}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())