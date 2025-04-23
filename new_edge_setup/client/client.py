import cv2
import requests
import numpy as np
import json
import base64
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


SERVER_URL = "http://127.0.0.1:8080/predict"


class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Client")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.create_control_frame()
        self.create_video_frame()
        self.create_log_frame()

        self.cap = None
        self.running = False
        self.last_send_time = 0
        self.video_thread = None
        self.camera_index = 0
        self.interval = 1.0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_control_frame(self):
        control_frame = ttk.LabelFrame(self.root, text="Controls")
        control_frame.pack(fill="x", padx=10, pady=10)

        camera_frame = ttk.Frame(control_frame)
        camera_frame.pack(side="left", padx=10, pady=5)

        ttk.Label(camera_frame, text="Camera:").pack(side="left", padx=5)
        self.camera_selector = ttk.Spinbox(camera_frame, from_=0, to=10, width=5, command=self.on_camera_change)
        self.camera_selector.set(self.camera_index)
        self.camera_selector.pack(side="left")

        interval_frame = ttk.Frame(control_frame)
        interval_frame.pack(side="left", padx=10, pady=5)

        ttk.Label(interval_frame, text="Send Interval (s):").pack(side="left", padx=5)
        self.interval_selector = ttk.Spinbox(
            interval_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            width=5,
            command=self.on_interval_change
        )
        self.interval_selector.set(self.interval)
        self.interval_selector.pack(side="left")

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_capture)
        self.start_button.pack(side="right", padx=10, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_capture, state="disabled")
        self.stop_button.pack(side="right", padx=5, pady=5)

    def create_video_frame(self):
        video_frame = ttk.LabelFrame(self.root, text="Video Feed")
        video_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

    def create_log_frame(self):
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="x", padx=10, pady=10)

        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def on_camera_change(self):
        try:
            self.camera_index = int(self.camera_selector.get())
            if self.running:
                self.log(f"Camera change will take effect after restart. New camera: {self.camera_index}")
        except ValueError:
            self.log("Invalid camera index")

    def on_interval_change(self):
        try:
            self.interval = float(self.interval_selector.get())
            self.log(f"Send interval updated to {self.interval} seconds")
        except ValueError:
            self.log("Invalid interval value")

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def start_capture(self):
        if self.running:
            return

        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera at index {self.camera_index}")
                return

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log(f"Camera initialized with resolution: {width}x{height}")
            self.log(f"Sending frames to server: {SERVER_URL}")

            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")

            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()

            self.log("Started video capture")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"Error: {e}")
            self.stop_capture()

    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.log("Stopped video capture")

    def video_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.log("Failed to capture frame")
                    break

                current_time = time.time()
                if current_time - self.last_send_time >= self.interval:
                    self.last_send_time = current_time
                    threading.Thread(target=self.process_frame, args=(frame.copy(),)).start()

                self.display_frame(frame)

                time.sleep(0.01)

            except Exception as e:
                self.log(f"Error in video loop: {e}")
                break

        if self.running:
            self.root.after(0, self.stop_capture)

    def process_frame(self, frame):
        try:
            _, jpeg_data = cv2.imencode('.jpg', frame)
            jpeg_bytes = jpeg_data.tobytes()
            base64_encoded = base64.b64encode(jpeg_bytes).decode('utf-8')

            payload = {
                'data': {
                    'image': base64_encoded,
                    'format': 'base64'
                }
            }

            response = requests.post(
                SERVER_URL,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )

            if response.status_code == 202:
                request_id = response.json().get('request_id')
                self.log(f"Request accepted: ID={request_id}")
            else:
                self.log(f"Error: {response.status_code}, {response.text}")

        except requests.RequestException as e:
            self.log(f"Request failed: {e}")
        except Exception as e:
            self.log(f"Error processing frame: {e}")

    def display_frame(self, frame):
        if not self.running:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 10 and canvas_height > 10:
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height

            if canvas_width / canvas_height > aspect_ratio:
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)

            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=img_tk, anchor="center"
        )
        self.canvas.img_tk = img_tk

    def on_close(self):
        if self.running:
            self.stop_capture()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()