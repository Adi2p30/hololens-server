# server.py
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import socket

# Initialize Flask application
app = Flask(__name__)

# Initialize YOLO model
model = YOLO('/home/isat/Aditya Pachpande/hololens/server/object_detection.pt')  # You can choose different models: yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.

# Store frames globally
frame_buffer = None
processed_frame = None
frame_lock = threading.Lock()

def get_local_ip():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def get_network_interfaces():
    """Get all available network interfaces with IP addresses"""
    interfaces = []
    try:
        import netifaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    if 'addr' in addr:
                        interfaces.append((interface, addr['addr']))
    except ImportError:
        # Fallback if netifaces is not available
        interfaces.append(('default', get_local_ip()))
    
    return interfaces

def process_frames():
    """Process frames with YOLO model in a separate thread"""
    global frame_buffer, processed_frame
    
    while True:
        # If there's a frame to process
        if frame_buffer is not None:
            with frame_lock:
                current_frame = frame_buffer.copy()
            
            # Run YOLO detection
            results = model(current_frame)
            
            # Draw detection results on the frame
            annotated_frame = results[0].plot()
            
            # Update the processed frame
            with frame_lock:
                processed_frame = annotated_frame
        
        # Sleep to avoid CPU overload
        time.sleep(0.01)

# Start the processing thread
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate frames for streaming"""
    global processed_frame
    
    while True:
        # If there's a processed frame
        if processed_frame is not None:
            with frame_lock:
                output_frame = processed_frame.copy()
            
            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            
            if not ret:
                continue
                
            # Yield the frame in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Sleep to maintain reasonable frame rate
        time.sleep(0.03)  # ~30 FPS

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receive frames from HoloLens, process with YOLO, and return bounding boxes"""
    global frame_buffer, processed_frame
    
    # Get the frame data from the request
    if 'frame' not in request.files:
        return jsonify({"boxes": []}), 400
        
    frame_data = request.files['frame'].read()
    
    # Convert to numpy array
    nparr = np.frombuffer(frame_data, np.uint8)
    
    # Decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Update the frame buffer
    with frame_lock:
        frame_buffer = frame
    
    # Run YOLO detection
    results = model(frame)
    result = results[0]  # Get the first result
    
    # Create list to store detection results
    detection_boxes = []
    
    # Extract bounding boxes, classes and confidence scores
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        for box in result.boxes:
            # Get box coordinates (normalized format)
            x1, y1, x2, y2 = box.xyxyn[0].tolist()  # xyxyn returns normalized coordinates
            
            # Calculate center, width, height (normalized)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Get class ID and name
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            # Get confidence
            confidence = float(box.conf[0].item())
            
            # Only keep confident detections
            if confidence > 0.25:  # Adjust threshold as needed
                detection_boxes.append({
                    "className": class_name,
                    "confidence": confidence,
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height
                })
    
    # Draw detection results on the frame for server-side display
    annotated_frame = result.plot()
    
    # Update the processed frame for server display
    with frame_lock:
        processed_frame = annotated_frame
    
    # Return the bounding box data as JSON
    return jsonify({"boxes": detection_boxes})

if __name__ == '__main__':
    # Get the local IP address
    local_ip = get_local_ip()
    print(f"\n=== YOLO Vision Server ===")
    print(f"Server running at http://{local_ip}:5000")
    print("Share this URL with others on the same WiFi network")
    print("You can connect clients using:")
    print(f"python client.py --server http://{local_ip}:5000")
    print("===========================\n")
    
    # Try to get all available interfaces
    try:
        import netifaces
        print("Available network interfaces:")
        for interface, ip in get_network_interfaces():
            print(f"- {interface}: http://{ip}:5000")
        print()
    except ImportError:
        pass  # netifaces is optional
        
    # Start the server
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)