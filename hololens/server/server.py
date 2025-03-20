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
        orig_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_frame is None:
            logger.error("Failed to decode frame")
            return jsonify({"error": "Failed to decode frame", "boxes": []}), 400

        # Get original dimensions
        orig_height, orig_width, channels = orig_frame.shape
        orig_aspect_ratio = orig_width / orig_height
        
        # Determine orientation
        orientation = "landscape" if orig_width >= orig_height else "portrait"
        logger.info(f"Original frame: {orig_width}x{orig_height} ({orientation})")

        # Create a copy for YOLO processing
        # YOLO works best with square-ish images, so we'll resize to a standard size
        yolo_size = 640  # Standard YOLO input size
        
        # Resize while maintaining aspect ratio
        if orig_width >= orig_height:
            # Landscape orientation
            yolo_width = yolo_size
            yolo_height = int(yolo_size / orig_aspect_ratio)
        else:
            # Portrait orientation
            yolo_height = yolo_size
            yolo_width = int(yolo_size * orig_aspect_ratio)
            
        # Ensure dimensions are even (some models prefer this)
        yolo_width = (yolo_width // 2) * 2
        yolo_height = (yolo_height // 2) * 2
        
        # Resize frame for YOLO processing
        yolo_frame = cv2.resize(orig_frame, (yolo_width, yolo_height))
        
        logger.info(f"YOLO input: {yolo_width}x{yolo_height}")

        # Update the frame buffer for the video feed display
        with frame_lock:
            frame_buffer = orig_frame.copy()

        # Run YOLO detection on the resized frame
        results = model(yolo_frame)
        result = results[0]  # Get the first result

        # Draw on the original frame for visualization
        annotated_frame = result.plot()
        
        # Update the processed frame
        with frame_lock:
            processed_frame = annotated_frame

        # Create list to store detection results
        detection_boxes = []
        
        # Calculate scale factors to map from YOLO frame to original frame
        scale_x = orig_width / yolo_width
        scale_y = orig_height / yolo_height

        # Extract bounding boxes, classes, and confidence scores
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box_idx, box in enumerate(result.boxes):
                # Get normalized coordinates from YOLO (0-1 range)
                x1_norm, y1_norm, x2_norm, y2_norm = box.xyxyn[0].tolist()
                
                # Convert normalized to pixel coordinates in YOLO frame
                x1_yolo = x1_norm * yolo_width
                y1_yolo = y1_norm * yolo_height
                x2_yolo = x2_norm * yolo_width
                y2_yolo = y2_norm * yolo_height
                
                # Scale coordinates to original frame
                x1_orig = x1_yolo * scale_x
                y1_orig = y1_yolo * scale_y
                x2_orig = x2_yolo * scale_x
                y2_orig = y2_yolo * scale_y
                
                # Calculate center and size in original frame
                center_x_orig = (x1_orig + x2_orig) / 2
                center_y_orig = (y1_orig + y2_orig) / 2
                width_orig = x2_orig - x1_orig
                height_orig = y2_orig - y1_orig
                
                # Re-normalize for client use (0-1 range in original frame)
                center_x_norm = center_x_orig / orig_width
                center_y_norm = center_y_orig / orig_height
                width_norm = width_orig / orig_width
                height_norm = height_orig / orig_height
                
                # Get class ID and name
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]

                # Get confidence
                confidence = float(box.conf[0].item())
                
                # Only keep confident detections
                if confidence > CONFIDENCE_THRESHOLD:
                    # Create client-ready detection object with all coordinates pre-calculated
                    detection_box = {
                        "className": class_name,
                        "confidence": confidence,
                        "boxId": box_idx,
                        
                        # Client-ready normalized coordinates (already aspect-ratio corrected)
                        "x": center_x_norm,                  # Center X (0-1)
                        "y": center_y_norm,                  # Center Y (0-1)
                        "width": width_norm,                 # Width (0-1)
                        "height": height_norm,               # Height (0-1)
                        "x1": x1_orig / orig_width,          # Top-left X (0-1)
                        "y1": y1_orig / orig_height,         # Top-left Y (0-1)
                        "x2": x2_orig / orig_width,          # Bottom-right X (0-1)
                        "y2": y2_orig / orig_height,         # Bottom-right Y (0-1)
                        
                        # Exact pixel coordinates in original frame
                        "x_px": center_x_orig,               # Center X in pixels
                        "y_px": center_y_orig,               # Center Y in pixels
                        "width_px": width_orig,              # Width in pixels
                        "height_px": height_orig,            # Height in pixels
                        "x1_px": x1_orig,                    # Top-left X in pixels
                        "y1_px": y1_orig,                    # Top-left Y in pixels
                        "x2_px": x2_orig,                    # Bottom-right X in pixels
                        "y2_px": y2_orig,                    # Bottom-right Y in pixels
                        
                        # Original YOLO coordinates (for debugging)
                        "yolo_x1": x1_yolo,
                        "yolo_y1": y1_yolo,
                        "yolo_x2": x2_yolo,
                        "yolo_y2": y2_yolo
                    }
                    
                    detection_boxes.append(detection_box)
            
            # Sort by confidence (highest first) for consistent ordering
            detection_boxes.sort(key=lambda box: box["confidence"], reverse=True)
            
            # Log detection info
            if detection_boxes:
                logger.info(f"Detected {len(detection_boxes)} objects with confidence >= {CONFIDENCE_THRESHOLD}")
                first_box = detection_boxes[0]
                logger.info(f"Top detection: {first_box['className']} ({first_box['confidence']:.2f})")
                logger.info(f"  Original frame: ({first_box['x1_px']:.1f}, {first_box['y1_px']:.1f}) → ({first_box['x2_px']:.1f}, {first_box['y2_px']:.1f})")

        # Create the response with complete frame info and pre-calculated coordinates
        response_data = {
            "boxes": detection_boxes,
            "frame_info": {
                "width": orig_width,                 # Original width in pixels
                "height": orig_height,               # Original height in pixels
                "aspect_ratio": orig_aspect_ratio,   # Original aspect ratio
                "orientation": orientation,          # Landscape or portrait
                "yolo_width": yolo_width,            # YOLO input width
                "yolo_height": yolo_height,          # YOLO input height
                "scale_x": scale_x,                  # Scale factor from YOLO to original (X)
                "scale_y": scale_y,                  # Scale factor from YOLO to original (Y)
                "processing_info": {
                    "coordinates_in_original_frame": True,  # Coordinates are already mapped to original frame
                    "aspect_ratio_corrected": True,         # Aspect ratio correction has been applied
                    "requires_client_correction": False     # Client doesn't need to apply corrections
                }
            },
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