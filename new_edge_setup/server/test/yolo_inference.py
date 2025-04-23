import time
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace with your model path if different

# Open a video capture (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

try:
    while True:
        start_time = time.time()

        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        results = model(frame)

        # Display the results
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Inference", annotated_frame)

        # Wait for 0.1 seconds
        elapsed_time = time.time() - start_time
        time_to_wait = max(0.1 - elapsed_time, 0)
        time.sleep(time_to_wait)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()