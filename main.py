import cv2
import streamlit as st
import time
import os

# Get the path to the script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Use the script's directory to load the cascade classifier
face_cascade = cv2.CascadeClassifier(os.path.join(script_dir, '/Users/amadoudiakhadiop/Downloads/haarcascade_frontalface_default.xml'))

def detect_faces(min_neighbors, scale_factor, rectangle_color):
    # Initialize the webcam
    cap = cv2.VideoCapture(1)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    # Wait for 2 seconds to allow the webcam to initialize
    time.sleep(2)

    frame_count = 0
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        frame_count += 1
        st.write(f"Frames processed: {frame_count}")

        # Check if the frame was successfully captured
        if not ret:
            st.error("Error: Failed to capture frame from the webcam.")
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frames using Streamlit
        st.image(rgb_frame, channels="RGB", use_column_width=True)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Add instructions
        st.write("Use 'q' to stop detecting faces.")

        # Add sliders for minNeighbors and scaleFactor
        min_neighbors = st.slider("minNeighbors (Adjust to reduce false positives)", 1, 10, 5)
        scale_factor = st.slider("scaleFactor (Adjust for sensitivity)", 1.1, 2.0, 1.3)

        # Add color picker for rectangle color
        rectangle_color = st.color_picker("Rectangle Color (Pick the color for rectangles)", "#00ff00")

        # Call the detect_faces function with user-selected parameters
        detect_faces(min_neighbors, scale_factor, rectangle_color)

if __name__ == "__main__":
    app()






