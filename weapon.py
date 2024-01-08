import cv2
from roboflow import Roboflow

# Initialize Roboflow API
rf = Roboflow(api_key="vNuH3IiQvIqq0fvGssUE")

# Specify the Roboflow project and model version
project = rf.workspace().project("gun-knife-thesis")
model = project.version(11).model

# Open a video capture from your webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Perform inference on the current frame
    predictions = model.predict(frame, confidence=50, overlap=30)

    # Visualize predictions on the frame
    for prediction in predictions:
        # Extract bounding box information
        x, y, width, height = (
            int(prediction['x']),
            int(prediction['y']),
            int(prediction['width']),
            int(prediction['height'])
        )
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0,0,255), 2)
        
        # Display class and confidence information
        label = prediction['class']
        confidence = prediction['confidence']
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)

    # Display the frame with predictions
    cv2.imshow("Live Fire Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
