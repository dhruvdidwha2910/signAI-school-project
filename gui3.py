import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from keras.models import load_model
from ultralytics import YOLO
import csv
from datetime import datetime
import os
from collections import deque

from sign_names import CLASS_NAMES

#init
MODEL_PATH = "gtsrb_model.h5"
YOLO_MODEL = "trainedyolo.pt"        

FRAME_W, FRAME_H = 960, 720
DISPLAY_W, DISPLAY_H = 900, 600

IMG_SIZE = 32

DETECTION_THRESHOLD = 0.35
DISPLAY_CONFIDENCE = 0.65
CSV_CONFIDENCE = 0.75

YOLO_INTERVAL = 4
STABLE_FRAMES = 6
COOLDOWN_FRAMES = 30

CSV_FILE = "detections.csv"

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

classifier = load_model(MODEL_PATH)
detector = YOLO(YOLO_MODEL)

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Sign Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")

        #ui
        self.header = Frame(root, bg="#111111", height=60)
        self.header.pack(fill="x")
        Label(
            self.header,
            text="Road Sign Recognition",
            font=("Segoe UI", 22, "bold"),  
            fg="white",
            bg="#111111"
        ).pack(pady=10)

        self.main = Frame(root, bg="#1e1e1e")
        self.main.pack(fill="both", expand=True)

        self.video_label = Label(self.main, bg="black")
        self.video_label.pack(side="left", expand=True)

        self.side = Frame(self.main, bg="#2b2b2b", width=300)
        self.side.pack(side="right", fill="y")

        Label(
            self.side,
            text="Detection Info",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).pack(pady=20)

        self.sign_label = Label(
            self.side,
            text="Sign: None",
            font=("Segoe UI", 16),
            fg="#00ff99",
            bg="#2b2b2b"
        )
        self.sign_label.pack(pady=10)

        self.conf_label = Label(
            self.side,
            text="Confidence: --",
            font=("Segoe UI", 14),
            fg="white",
            bg="#2b2b2b"
        )
        self.conf_label.pack(pady=5)

        self.status_label = Label(
            self.side,
            text="Status: Idle",
            font=("Segoe UI", 14),
            fg="#ffaa00",
            bg="#2b2b2b"
        )
        self.status_label.pack(pady=20)

        Button(
            self.side,
            text="Reset CSV",
            font=("Segoe UI", 14),
            bg="#7234A9",
            fg="black",
            command=self.reset_csv,
            width=12
        ).pack(pady=10)

        Button(
            self.side,
            text="Exit",
            font=("Segoe UI", 14),
            bg="#5233cc",
            fg="black",
            command=self.exit_app,
            width=12
        ).pack(pady=30)

        #vidinput
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        
        self.frame_count = 0
        self.cached_results = []

        self.last_prediction = None
        self.stable_count = 0
        self.logged_current_sign = False
        self.cooldown = 0

        self.conf_buffer = deque(maxlen=STABLE_FRAMES)
        self.label_buffer = deque(maxlen=STABLE_FRAMES)

        #resetcsv(startup)
        self.reset_csv()

        self.update_frame()

    
    def reset_csv(self):
        """Clear the csv file and add header."""
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Sign", "Avg_Confidence"])
        self.status_label.config(text="Status: CSV Reset", fg="#ffaa00")

    
    def preprocess(self, img):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    #logdetect
    def log_detection(self, sign, avg_conf):
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 sign,
                 round(float(avg_conf), 3)]
            )

    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        self.frame_count += 1

        #run yolo(n-frames)
        if self.frame_count % YOLO_INTERVAL == 0:
            self.cached_results = detector(frame, verbose=False)

        detected = False

        for r in self.cached_results:
            for box in r.boxes[:3]:
                if float(box.conf[0]) < DETECTION_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                preds = classifier(self.preprocess(crop), training=False)[0]

                
                top2 = np.sort(preds)[-2:]
                if (top2[1] - top2[0]) < 0.25:
                    continue

                class_id = np.argmax(preds)
                conf = preds[class_id]

                if conf < DISPLAY_CONFIDENCE:
                    continue

                sign = CLASS_NAMES.get(class_id, "unknown")
                detected = True

                #stability
                if sign == self.last_prediction:
                    self.stable_count += 1
                else:
                    self.last_prediction = sign
                    self.stable_count = 1
                    self.logged_current_sign = False
                    self.conf_buffer.clear()
                    self.label_buffer.clear()

                
                self.conf_buffer.append(conf)
                self.label_buffer.append(sign)

                
                if len(self.conf_buffer) == 0:
                    continue

                avg_conf = sum(self.conf_buffer) / len(self.conf_buffer)

                #csvlog
                if (
                    self.stable_count >= STABLE_FRAMES
                    and not self.logged_current_sign
                    and avg_conf >= CSV_CONFIDENCE
                    and self.label_buffer.count(sign) == STABLE_FRAMES
                ):
                    self.sign_label.config(text=f"Sign: {sign}")
                    self.conf_label.config(text=f"Confidence: {avg_conf:.2f}")
                    self.status_label.config(
                        text="Status: Detected", fg="#00ff99"
                    )

                    self.log_detection(sign, avg_conf)
                    self.logged_current_sign = True
                    self.cooldown = COOLDOWN_FRAMES

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    sign,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        if not detected:
            self.last_prediction = None
            self.stable_count = 0
            self.logged_current_sign = False
            self.conf_buffer.clear()
            self.label_buffer.clear()

            self.sign_label.config(text="Sign: None")
            self.conf_label.config(text="Confidence: --")
            self.status_label.config(text="Status: Idle", fg="#ffaa00")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((DISPLAY_W, DISPLAY_H))
        imgtk = ImageTk.PhotoImage(img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    
    def exit_app(self):
        self.cap.release()
        self.root.destroy()


root = tk.Tk()
TrafficSignApp(root)
root.mainloop()
 