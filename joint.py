import cv2
from roboflow import Roboflow

# Initialize Roboflow API
rf = Roboflow(api_key="vNuH3IiQvIqq0fvGssUE")

# Specify the Roboflow projects and model versions
projects = [
    {"project_name": "", "model_version":3},
    {"project_name": "fire-smoke-detection-odvk6", "model_version": 1},
    {"project_name": "gun-knife-thesis", "model_version": 11},
]

# Load models
models = []
for project_info in projects:
    project = rf.workspace().project(project_info["project_name"])
    model = project.version(project_info["model_version"]).model
    models.append(model)

# Open a video capture from your webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Perform inference on each model for the current frame
    all_predictions = []
    for model in models:
        confidence = 40  # default confidence value

        if model.classes is not None:
            if "fire" in model.classes:
                confidence = 80
            elif any(c in model.classes for c in ["knife"]):
                confidence = 40  # set different confidence for knife

        predictions = model.predict(frame, confidence=confidence, overlap=30)
        all_predictions.extend(predictions)

    # Visualize predictions on the frame
    for prediction in all_predictions:
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
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Display the frame with predictions
    cv2.imshow("Combined Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
