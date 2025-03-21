# server.py
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import socket
import logging
import os
import json

# Set up logging to capture server activity

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HoloLensServer")

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize YOLO model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "object_detection.pt"
)
logger.info(f"Loading YOLO model from: {model_path}")
try:
    model = YOLO(model_path)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

# Store frames globally
frame_buffer = None
processed_frame = None
frame_lock = threading.Lock()

# YOLO detection settings
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections


def get_local_ip():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
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
                    if "addr" in addr:
                        interfaces.append((interface, addr["addr"]))
    except ImportError:
        # Fallback if netifaces is not available
        interfaces.append(("default", get_local_ip()))

    return interfaces


def process_frames():
    """Process frames with YOLO model in a separate thread"""
    global frame_buffer, processed_frame

    if model is None:
        logger.error("Cannot start processing thread: YOLO model not loaded")
        return

    logger.info("Frame processing thread started")
    while True:
        try:
            # If there's a frame to process
            if frame_buffer is not None:
                with frame_lock:
                    if frame_buffer is None:
                        continue
                    current_frame = frame_buffer.copy()

                # Run YOLO detection
                results = model(current_frame)

                # Draw detection results on the frame
                annotated_frame = results[0].plot()

                # Update the processed frame
                with frame_lock:
                    processed_frame = annotated_frame
        except Exception as e:
            logger.error(f"Error in processing thread: {str(e)}")

        # Sleep to avoid CPU overload
        time.sleep(0.01)


# Start the processing thread
if model is not None:
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()


@app.route("/")
def index():
    """Render the home page"""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for connectivity testing"""
    response = jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "server_address": request.host,
            "server_version": "1.0.0",
        }
    )
    return response


@app.route("/video_feed")
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def generate_frames():
    """Generate frames for streaming"""
    global processed_frame

    while True:
        # If there's a processed frame
        if processed_frame is not None:
            with frame_lock:
                if processed_frame is None:
                    continue
                output_frame = processed_frame.copy()

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode(".jpg", output_frame)

            if not ret:
                continue

            # Yield the frame in bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

        # Sleep to maintain reasonable frame rate
        time.sleep(0.03)  # ~30 FPS


@app.route("/upload_frame", methods=["POST", "OPTIONS"])
def upload_frame():
    """Receive frames from HoloLens, process with YOLO, and return bounding boxes"""
    global frame_buffer, processed_frame

    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return response

    if model is None:
        logger.error("Received frame but YOLO model is not loaded")
        return jsonify({"error": "YOLO model not loaded", "boxes": []}), 500

    try:
        # Get the frame data from the request
        if "frame" not in request.files:
            logger.warning("No frame found in request")
            return jsonify({"error": "No frame in request", "boxes": []}), 400

        frame_data = request.files["frame"].read()
        logger.info(f"Received frame data: {len(frame_data)} bytes")

        # Convert to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)

        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode frame")
            return jsonify({"error": "Failed to decode frame", "boxes": []}), 400

        # Get dimensions - use integers to ensure exact pixel matches
        frame_height, frame_width = frame.shape[:2]
        logger.info(f"Decoded frame: {frame_width}x{frame_height}")

        # Update the frame buffer for the video feed
        with frame_lock:
            frame_buffer = frame.copy()

        # Run YOLO detection directly on the received frame without any modifications
        results = model(frame)
        result = results[0]  # Get the first result

        # Create list to store detection results
        detection_boxes = []

        # Extract bounding boxes, classes and confidence scores
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            # Process each detected box
            for box_idx, box in enumerate(result.boxes):
                # Get pixel coordinates directly from YOLO
                x1_px, y1_px, x2_px, y2_px = box.xyxy[0].tolist()
                
                # Convert to absolute integer coordinates - no scaling or transformations
                x1_abs = int(round(x1_px))
                y1_abs = int(round(y1_px))
                x2_abs = int(round(x2_px))
                y2_abs = int(round(y2_px))
                
                # Calculate center and dimensions in absolute pixels
                center_x_abs = int((x1_abs + x2_abs) // 2)
                center_y_abs = int((y1_abs + y2_abs) // 2)
                width_abs = int(x2_abs - x1_abs)
                height_abs = int(y2_abs - y1_abs)

                # Get class ID and name
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]

                # Get confidence
                confidence = float(box.conf[0].item())

                # Only keep confident detections
                if confidence > CONFIDENCE_THRESHOLD:
                    # Add detection to the list with absolute pixel coordinates only
                    detection_boxes.append(
                        {
                            "className": class_name,
                            "confidence": confidence,
                            "boxId": box_idx,
                            
                            # Absolute pixel coordinates only - no relative values
                            "x_abs": center_x_abs,
                            "y_abs": center_y_abs,
                            "width_abs": width_abs,
                            "height_abs": height_abs,
                            "x1_abs": x1_abs,
                            "y1_abs": y1_abs,
                            "x2_abs": x2_abs,
                            "y2_abs": y2_abs,
                            
                            # Include original raw values for debugging
                            "x1_raw": x1_px,
                            "y1_raw": y1_px,
                            "x2_raw": x2_px,
                            "y2_raw": y2_px
                        }
                    )
            
            # Sort by confidence (highest first)
            detection_boxes.sort(key=lambda box: box["confidence"], reverse=True)

        logger.info(f"Detected {len(detection_boxes)} objects")

        # Draw detection results on the frame for server-side display
        annotated_frame = result.plot()

        # Update the processed frame for server display
        with frame_lock:
            processed_frame = annotated_frame

        # Include original frame dimensions in the response
        response_data = {
            "boxes": detection_boxes,
            "frame_info": {
                "width": frame_width,
                "height": frame_height,
                "absolute_coordinates": True,  # Flag indicating absolute pixel coordinates
                "coordinate_system": "top_left_origin"  # Explicitly state the coordinate system
            }
        }

        # Return the bounding box data as JSON
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "boxes": []}), 500


if __name__ == "__main__":
    # Get the local IP address
    local_ip = get_local_ip()
    port = 8080

    print(f"\n=== YOLO Vision Server for HoloLens ===")
    print(f"Server running at http://{local_ip}:{port}")
    print(f"Health check: http://{local_ip}:{port}/health")
    print("Share this URL with your HoloLens app")
    print("===========================\n")

    # Try to get all available interfaces
    print("Available network interfaces:")
    for interface, ip in get_network_interfaces():
        print(f"- {interface}: http://{ip}:{port}")
    print()

    # Start the server
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)