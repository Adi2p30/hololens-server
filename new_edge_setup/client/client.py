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


# Default URL - can be changed in the UI
SERVER_URL = "http://192.168.4.153:8080"


class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Client")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize attributes before using them
        self.camera_index = 0
        self.interval = 1.0
        self.cap = None
        self.running = False
        self.last_send_time = 0
        self.video_thread = None
        self.pending_requests = {}
        self.results_check_thread = None
        self.results_thread_running = False

        self.create_control_frame()
        self.create_video_frame()
        self.create_log_frame()
        self.create_results_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_control_frame(self):
        control_frame = ttk.LabelFrame(self.root, text="Controls")
        control_frame.pack(fill="x", padx=10, pady=10)

        # Server URL Frame
        server_frame = ttk.Frame(control_frame)
        server_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(server_frame, text="Server URL:").pack(side="left", padx=5)
        self.server_url_var = tk.StringVar(value=SERVER_URL)
        self.server_entry = ttk.Entry(server_frame, textvariable=self.server_url_var, width=40)
        self.server_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Camera Frame
        camera_frame = ttk.Frame(control_frame)
        camera_frame.pack(side="left", padx=10, pady=5)

        ttk.Label(camera_frame, text="Camera:").pack(side="left", padx=5)
        self.camera_selector = ttk.Spinbox(camera_frame, from_=0, to=10, width=5, command=self.on_camera_change)
        self.camera_selector.set(self.camera_index)
        self.camera_selector.pack(side="left")

        # Interval Frame
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
        log_frame.pack(fill="x", padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=6, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def create_results_frame(self):
        results_frame = ttk.LabelFrame(self.root, text="Detection Results")
        results_frame.pack(fill="x", padx=10, pady=5)

        self.results_text = tk.Text(results_frame, height=6, wrap="word")
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)

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

    def show_result(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.results_text.insert("end", f"[{timestamp}] {message}\n")
        self.results_text.see("end")

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
            self.log(f"Sending frames to server: {self.server_url_var.get()}")

            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")

            # Start the results checking thread
            self.results_thread_running = True
            self.results_check_thread = threading.Thread(target=self.check_results_loop)
            self.results_check_thread.daemon = True
            self.results_check_thread.start()

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
        self.results_thread_running = False
        
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

            # Get the current server URL from the entry field
            server_url = self.server_url_var.get().rstrip('/')
            predict_url = f"{server_url}/predict"
            
            # Check if the URL uses HTTPS but is on localhost, where certificate verification might fail
            verify_ssl = True
            if predict_url.startswith('https://') and ('127.0.0.1' in predict_url or 'localhost' in predict_url):
                verify_ssl = False
                # Only log this warning once
                if not hasattr(self, '_ssl_warning_shown'):
                    self.log("Warning: SSL verification disabled for localhost HTTPS")
                    self._ssl_warning_shown = True
            
            response = requests.post(
                predict_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5.0,
                verify=verify_ssl
            )

            if response.status_code == 202:
                response_data = response.json()
                request_id = response_data.get('request_id')
                self.log(f"Request accepted: ID={request_id}")
                
                # Add to pending requests
                self.pending_requests[request_id] = {
                    'timestamp': time.time(),
                    'status': 'pending'
                }
                
            else:
                self.log(f"Error: {response.status_code}, {response.text}")

        except requests.RequestException as e:
            self.log(f"Request failed: {e}")
        except Exception as e:
            self.log(f"Error processing frame: {e}")

    def check_results_loop(self):
        """Thread to check for results of pending requests"""
        while self.results_thread_running:
            try:
                current_time = time.time()
                # Create a copy of keys to avoid modification during iteration
                pending_ids = list(self.pending_requests.keys())
                
                for request_id in pending_ids:
                    # Skip requests that are too recent (give server time to process)
                    if current_time - self.pending_requests[request_id]['timestamp'] < 0.5:
                        continue
                        
                    # Skip already completed requests
                    if self.pending_requests[request_id]['status'] != 'pending':
                        continue
                        
                    # Check if result is available
                    self.check_result(request_id)
                    
                # Remove old requests to prevent memory growth
                self.cleanup_old_requests()
                    
                # Don't hammer the server
                time.sleep(0.5)
                
            except Exception as e:
                self.log(f"Error in results checking thread: {e}")
                time.sleep(1)
                
    def check_result(self, request_id):
        """Check if a result is available for the given request ID"""
        try:
            server_url = self.server_url_var.get().rstrip('/')
            result_url = f"{server_url}/results/{request_id}"
            
            response = requests.get(
                result_url,
                timeout=2.0
            )
            
            if response.status_code == 200:
                result_data = response.json()
                self.handle_result(request_id, result_data)
                self.pending_requests[request_id]['status'] = 'completed'
                
            elif response.status_code != 404:  # 404 just means "not ready yet"
                self.log(f"Error checking result for ID {request_id}: {response.status_code}")
                
        except requests.RequestException as e:
            # Don't log timeouts or connection errors as they are expected when polling
            pass
            
    def handle_result(self, request_id, result_data):
        """Process and display the detection results"""
        try:
            model_id = result_data.get('model_id', 'unknown')
            result = result_data.get('result', {})
            
            if 'error' in result:
                self.show_result(f"Request {request_id} failed: {result['error']}")
                return
                
            # For YOLO results
            if 'detections' in result:
                detections = result['detections']
                count = result.get('count', len(detections))
                
                if count == 0:
                    self.show_result(f"No objects detected for request {request_id}")
                else:
                    self.show_result(f"Request {request_id}: Detected {count} objects:")
                    for i, detection in enumerate(detections[:5]):  # Limit to 5 for display
                        class_name = detection.get('class_name', 'unknown')
                        confidence = detection.get('confidence', 0) * 100
                        self.show_result(f"  - {class_name} ({confidence:.1f}%)")
                        
                    if count > 5:
                        self.show_result(f"  ... and {count - 5} more objects")
            else:
                # Generic result display
                self.show_result(f"Result for request {request_id} from {model_id}:")
                self.show_result(f"  {str(result)[:100]}...")
                
        except Exception as e:
            self.log(f"Error processing result: {e}")
    
    def cleanup_old_requests(self):
        """Remove old requests from the pending list"""
        current_time = time.time()
        to_remove = []
        
        for request_id, info in self.pending_requests.items():
            # Remove completed requests after 30 seconds
            if info['status'] == 'completed' and current_time - info['timestamp'] > 30:
                to_remove.append(request_id)
            # Remove pending requests after 60 seconds (they probably failed)
            elif info['status'] == 'pending' and current_time - info['timestamp'] > 60:
                to_remove.append(request_id)
                
        for request_id in to_remove:
            del self.pending_requests[request_id]