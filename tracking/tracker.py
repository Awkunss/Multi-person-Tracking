import time

import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from models.common import AutoShape, DetectMultiBackend

class YOLOv9Tracker:
    def __init__(self, weights_path, classes_path, device="cuda", conf_threshold=0.5):
        self.device = torch.device(device)
        self.model = self._load_model(weights_path)
        self.class_names = self._load_class_names(classes_path)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
        self.conf_threshold = conf_threshold
        self.tracker = DeepSort(max_age=30)
        self.prev_time = 0

    def _load_model(self, weights_path):
        model = DetectMultiBackend(weights=weights_path, device=self.device, fuse=True)
        return AutoShape(model)

    def _load_class_names(self, classes_path):
        with open(classes_path) as f:
            return f.read().strip().split('\n')

    def process_frame(self, frame):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time

        # Perform object detection
        results = self.model(frame)
        detections = self._extract_detections(results)

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Draw detections and tracker information
        self._draw_tracks(frame, tracks)
        self._draw_fps(frame, fps)

        return frame

    def _extract_detections(self, results):
        detections = []
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if class_id != 0:
                continue
            if confidence < self.conf_threshold:
                continue
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        return detections

    def _draw_tracks(self, frame, tracks):
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = tuple(map(int, self.colors[class_id % len(self.colors)]))
                label = f"ID-{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
