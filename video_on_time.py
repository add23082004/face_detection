import cv2
import streamlit as st
import time
import os
import numpy as np

# Get the path to the script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Use the script's directory to load the cascade classifier
face_cascade = cv2.CascadeClassifier(os.path.join(script_dir, '/Users/amadoudiakhadiop/Downloads/haarcascade_frontalface_default.xml'))


def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    # Wait for 2 seconds to allow the webcam to initialize
    time.sleep(2)

    # Set the frame width and height
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            st.error("Error: Failed to capture frame from the webcam.")
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frames using Streamlit as a video
        st.video(frame, format="BGR")  # Use format="BGR" for color images

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces()


if __name__ == "__main__":
    app()
