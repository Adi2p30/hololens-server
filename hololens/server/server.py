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
    logger.error(f"Failed to load YOLO model:    {str(e)}")
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

    logger.info("Frame pimport socket processing thread started")
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


# Start the processing threadlocal_ip = get_local_ip()
        port = 8080

        print(f"\n=== YOLO Vision Server for HoloLens ===")
        print(f"Server running at http://{local_ip}:{port}")
        print(f"Health check: http://{local_ip}:{port}/health")
        print("Share this URL with your HoloLens app")
        print("===========================\n")
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

        # Log frame dimensions and format
        orig_height, orig_width, channels = frame.shape
        logger.info(f"Decoded frame: {orig_width}x{orig_height}x{channels}")

        # Calculate aspect ratio
        orig_aspect_ratio = orig_width / orig_height
        logger.info(f"Original aspect ratio: {orig_aspect_ratio:.4f}")

        # Create a copy for processing
        frame_for_detection = frame.copy()

        # Store the original dimensions for coordinate mapping
        detection_height, detection_width = frame_for_detection.shape[:2]

        # Update the frame buffer for the video feed
        with frame_lock:
            frame_buffer = frame.copy()

        # Run YOLO detection
        results = model(frame_for_detection)
        result = results[0]  # Get the first result

        # Create list to store detection results
        detection_boxes = []

        # Extract bounding boxes, classes and confidence scores
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            # Log raw detection details for the first box to help with debugging
            if len(result.boxes) > 0:
                first_box = result.boxes[0]
                x1, y1, x2, y2 = first_box.xyxyn[0].tolist()
                class_id = int(first_box.cls[0].item())
                class_name = model.names[class_id]
                confidence = float(first_box.conf[0].item())
                logger.info(
                    f"First detection: {class_name} ({confidence:.2f}) at [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]"
                )

            for box in result.boxes:
                # Get box coordinates (normalized format)
                x1, y1, x2, y2 = box.xyxyn[
                    0
                ].tolist()  # xyxyn returns normalized coordinates (0-1)

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
                if confidence > CONFIDENCE_THRESHOLD:
                    # Add to detection boxes with all the metadata needed for client-side rendering
                    detection_boxes.append(
                        {
                            "className": class_name,
                            "confidence": confidence,
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height,
                            # Original normalized coordinates (x1,y1,x2,y2)
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

        logger.info(f"Detected {len(detection_boxes)} objects")

        # Draw detection results on the frame for server-side display
        annotated_frame = result.plot()

        # Update the processed frame for server display
        with frame_lock:
            processed_frame = annotated_frame

        # Create the response with frame info
        response_data = {
            "boxes": detection_boxes,
            "frame_info": {
                "width": orig_width,
                "height": orig_height,
                "aspect_ratio": orig_aspect_ratio,
                "detection_width": detection_width,
                "detection_height": detection_height,
            },
        }

        # Log the full response for debugging (truncate if too large)
        response_json = json.dumps(response_data)
        if len(response_json) > 1000:
            logger.debug(f"Response: {response_json[:1000]}...")
        else:
            logger.debug(f"Response: {response_json}")

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
