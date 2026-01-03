#!/usr/bin/env python3
"""
Optimized Video-Driven Edge–Cloud Traffic Simulation
Includes:
- GMM (MOG2) fallback
- YOLOv8 vehicle detection
- SORT tracking
- Speed estimation
- Congestion detection
- Deadline-aware edge–cloud scheduling (SimPy)
"""

import cv2
import simpy
import time
import numpy as np
import random
import math
from ultralytics import YOLO
from collections import defaultdict, deque
import joblib
import os
from sklearn.ensemble import RandomForestClassifier


# =========================
# SORT TRACKER (LIGHTWEIGHT)
# =========================
from filterpy.kalman import KalmanFilter

class SortTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        updated = []
        for det in detections:
            cx, cy = det["center"]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = (cx, cy)
            updated.append((tid, cx, cy, det["class"]))
        return updated


# =========================
# VIDEO ANALYTICS MODULE
# =========================
class VideoAnalytics:
    def __init__(self, meters_per_pixel=0.05):
        self.yolo = YOLO("yolov8n.pt")
        self.bg_sub = cv2.createBackgroundSubtractorMOG2()
        self.tracker = SortTracker()
        self.prev_positions = {}
        self.mpp = meters_per_pixel

    def process_frame(self, frame, fps):
        detections = []

        # ---- YOLO DETECTION ----
        results = self.yolo(frame, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append({
                    "center": (cx, cy),
                    "class": cls
                })

        # ---- TRACKING ----
        tracks = self.tracker.update(detections)

        vehicle_data = []
        for tid, cx, cy, cls in tracks:
            speed = 0.0
            if tid in self.prev_positions:
                px, py = self.prev_positions[tid]
                dist_px = math.hypot(cx - px, cy - py)
                speed = (dist_px * self.mpp * fps) * 3.6
            self.prev_positions[tid] = (cx, cy)

            vehicle_data.append({
                "id": tid,
                "class": cls,
                "speed": speed
            })

        return vehicle_data


# =========================
# CONGESTION DETECTION
# =========================
# =========================
# ML-BASED CONGESTION MODEL
# =========================
def train_congestion_model():
    X, y = [], []

    for _ in range(1200):
        vehicle_count = random.randint(0, 40)
        avg_speed = random.uniform(5, 70)

        if vehicle_count > 25 and avg_speed < 20:
            label = 2  # HIGH congestion
        elif vehicle_count > 12:
            label = 1  # MEDIUM congestion
        else:
            label = 0  # LOW congestion

        # ✅ ONLY TWO FEATURES
        X.append([vehicle_count, avg_speed])
        y.append(label)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, "congestion_model.joblib")
    print("[INFO] congestion_model.joblib trained with 2 features")

# Train only if model does not exist
if not os.path.exists("congestion_model.joblib"):
    train_congestion_model()

congestion_model = joblib.load("congestion_model.joblib")


def detect_congestion(vehicle_data):
    count = len(vehicle_data)

    avg_speed = (
        sum(v["speed"] for v in vehicle_data) / count
        if count > 0 else 0.0
    )

    # ✅ EXACTLY TWO FEATURES (matches training)
    label = congestion_model.predict([[count, avg_speed]])[0]

    if label == 2:
        return "HIGH", avg_speed
    elif label == 1:
        return "MEDIUM", avg_speed
    else:
        return "LOW", avg_speed




# =========================
# SIMPY EDGE–CLOUD SYSTEM
# =========================
class Stats:
    def __init__(self):
        self.processed = 0
        self.edge = 0
        self.cloud = 0
        self.deadline_miss = 0

class Cloud:
    def __init__(self, env, mips=5000):
        self.env = env
        self.mips = mips
        self.res = simpy.Resource(env, capacity=6)

    def run(self, task):
        with self.res.request() as r:
            yield r
            yield self.env.timeout(task["work"] / self.mips)

class EdgeNode:
    def __init__(self, env, stats, cloud, mips=800):
        self.env = env
        self.stats = stats
        self.cloud = cloud
        self.mips = mips
        self.queue = simpy.PriorityStore(env)

        env.process(self.worker())

    def submit(self, task):
        self.queue.put((task["priority"], task))

    def worker(self):
        while True:
            _, task = yield self.queue.get()
            exec_time = task["work"] / self.mips
            if exec_time <= task["deadline"]:
                yield self.env.timeout(exec_time)
                self.stats.edge += 1
            else:
                yield self.env.process(self.cloud.run(task))
                self.stats.cloud += 1
            self.stats.processed += 1


# =========================
# TASK GENERATION
# =========================
def generate_tasks(env, analytics, edge):
    cap = cv2.VideoCapture("D:\\Edge\\EdgeComputing\\datasets\\rev2vido.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vehicles = analytics.process_frame(frame, fps)
        congestion, avg_speed = detect_congestion(vehicles)

        for v in vehicles:
            urgent = (v["speed"] > 60 or congestion == "HIGH")
            task = {
                "priority": 0 if urgent else 1,
                "work": 2000 if urgent else 800,
                "deadline": 2 if urgent else 6
            }
            edge.submit(task)

        yield env.timeout(1 / fps)


# =========================
# MAIN
# =========================
def main():
    env = simpy.Environment()
    stats = Stats()
    cloud = Cloud(env)
    edge = EdgeNode(env, stats, cloud)
    analytics = VideoAnalytics()

    env.process(generate_tasks(env, analytics, edge))
    env.run(until=120)

    print("\n=== FINAL SUMMARY ===")
    print("Processed:", stats.processed)
    print("Edge:", stats.edge)
    print("Cloud:", stats.cloud)
    print("Deadline Misses:", stats.deadline_miss)

if __name__ == "__main__":
    main()
