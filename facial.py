from IPython.display import Javascript
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import face_recognition

def take_photo():
    js = """
        async function takePhoto() {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Wait for Capture to be clicked.
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg');
        }
    """
    display(Javascript(js))
    data = eval_js('takePhoto()')
    binary = b64decode(data.split(',')[1])
    return np.frombuffer(binary, dtype=np.uint8)

# Known faces and encodings
known_face_encodings = []
known_face_names = []

# Load a sample image and encode it
known_image = face_recognition.load_image_file("/content/WhatsApp Image 2024-11-30 at 8.13.01 PM.jpeg")  # Replace with your image path
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_encodings.append(known_face_encoding)
known_face_names.append("Akif")  # Replace with the name

print("Starting real-time face recognition...")
while True:
    try:
        # Capture a photo
        photo = take_photo()
        frame = cv2.imdecode(photo, cv2.IMREAD_COLOR)

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame
        cv2_imshow(frame)

    except Exception as e:
        print(f"Error: {e}")
        break
