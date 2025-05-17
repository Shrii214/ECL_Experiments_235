import edgeimpulse_linux
import cv2
import numpy as np
# Load the trained model from Edge Impulse
model = edgeimpulse_linux.EimModel("path_to_your_model.eim") # Replace with your
model's path
# Initialize the camera (use 0 for default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
print("Error: Could not access the camera.")
exit()
print("Running Inference...")
# Inference loop (run inference on live video feed)
while True:
ret, frame = cap.read()
if not ret:
print("Error: Failed to capture image.")
Break
# Preprocess image (resize, normalize)
normalized_frame = resized_frame.astype(np.float32) / 255.0 # Normalize to [0,1] range
# Run inference using the Edge Impulse model
result = model.classify(normalized_frame)
# Print the result (predicted class and confidence)
print(f"Predicted Class: {result['label']} with Confidence: {result['confidence']}")
# Display the live feed and inference result on the frame
cv2.putText(frame, f"Class: {result['label']} ({result['confidence']:.2f})",
(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("Live Object Detection", frame)
# Exit the loop when 'q' is pressed
if cv2.waitKey(1) & 0xFF == ord('q'):
break
# Clean up
cap.release()
cv2.destroyAllWindows()