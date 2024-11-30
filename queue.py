# Import necessary libraries
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model (pretrained on COCO dataset)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for the nano version (lightweight)

# Function to determine sentiment based on queue length
def get_sentiment(queue_count):
    if queue_count <= 5:  # Short queue
        sentiment = "Positive - Manageable Queue"
        color = (0, 255, 0)  # Green
    elif queue_count <= 10:  # Medium queue
        sentiment = "Neutral - Queue is Growing"
        color = (0, 255, 255)  # Yellow
    else:  # Long queue
        sentiment = "Negative - Long Queue"
        color = (0, 0, 255)  # Red
    return sentiment, color

# Function to process frames for queue management
def process_frame(frame):
    # Perform inference
    results = model(frame)
    
    # Extract detections
    detections = results[0].boxes
    count = 0

    for box in detections:
        # Extract label and confidence
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Check if detected object is 'person' (class ID for person in COCO is 0)
        if class_id == 0 and conf > 0.5:  # Adjust confidence threshold as needed
            count += 1

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Determine sentiment
    sentiment, color = get_sentiment(count)

    # Add the queue count and sentiment to the frame
    cv2.putText(frame, f'Queue Count: {count}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, sentiment, (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame, count, sentiment

# Capture video feed from webcam or file
video_source = 0  # Set to 0 for webcam or provide the path to a video file
cap = cv2.VideoCapture(video_source)

# Check if the video source is available
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

while True:
    # Read a frame from the video source
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Process the frame
    processed_frame, people_count, sentiment = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Queue Management with Sentiment - YOLOv8', processed_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
