import cv2
import time
from playsound import playsound

# Constants for mock testing
LONG_QUEUE_THRESHOLD = 15  # Mock threshold for long queues
MOCK_FIGHT_DETECTED = True  # Simulate a fight
MOCK_FAINTING_DETECTED = True  # Simulate fainting

# Load a sample video (can be replaced with a path to any video file)
video_path = "C:/Users/HP/OneDrive/Desktop/codenovate/mock.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Function to simulate the alarm system
def ring_alarm(message):
    print(f"ALERT: {message}")
    playsound('mixkit-alert-alarm-1005.wav')  # Replace with your alarm sound file

# Main loop to process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or no input. Exiting...")
        break

    # Simulate anomaly conditions for testing
    mock_num_persons = 20  # Mock number of people in queue
    mock_fight_detected = MOCK_FIGHT_DETECTED  # Simulated fight detection
    mock_fainting_detected = MOCK_FAINTING_DETECTED  # Simulated fainting detection

    # Simulated long queue detection
    if mock_num_persons > LONG_QUEUE_THRESHOLD:
        ring_alarm("Long queue detected! Please address the situation.")
        # Reset or disable this condition after testing
        mock_num_persons = 0  # Reset mock value after the alarm

    # Simulated fight detection
    if mock_fight_detected:
        ring_alarm("Fight detected! Alert security.")
        # Reset or disable this condition after testing
        mock_fight_detected = False

    # Simulated fainting detection
    if mock_fainting_detected:
        ring_alarm("Person fainted! Call for medical assistance.")
        # Reset or disable this condition after testing
        mock_fainting_detected = False

    # Display the video frame (optional, for visualization)
    cv2.imshow('Mock Anomaly Detection', frame)

    # Press 'q' to quit the simulation
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
