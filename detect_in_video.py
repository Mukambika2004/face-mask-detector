# detect_in_video.py
import tensorflow as tf
import cv2
import numpy as np
import os

# --- 1. Load Pre-trained Face Detector ---
# We use OpenCV's built-in deep learning face detector
print("[INFO] Loading face detector model...")
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# --- 2. Load the Face Mask Detector Model ---
print("[INFO] Loading face mask detector model...")
# MAKE SURE you updated this line to .h5!
maskNet = tf.keras.models.load_model("mask_detector.h5")

# --- 3. Initialize Video Stream ---
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0) # '0' means the default webcam

# --- 4. Loop Over Frames ---
while True:
    # Read the next frame from the video stream
    ret, frame = vs.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    # Create a blob from the image for the face detector
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network to detect faces
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5: # Filter out weak detections
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box is within the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # --- 5. Predict Mask ---
            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            
            # Check if the face ROI is valid
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Predict
            (mask, withoutMask) = maskNet.predict(face)[0]

            # Determine label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display label and bounding box
            label_with_prob = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
            cv2.putText(frame, label_with_prob, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# --- 6. Cleanup ---
vs.release()
cv2.destroyAllWindows()