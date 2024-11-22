import os
import tensorflow as tf
import cv2
import numpy as np

# Check if the model file exists
model_path = '/Users/keshavbisht/PycharmProjects/MotionLink/.venv/Model/keras_model.h5'
if os.path.exists(model_path):
    print("Model file found!")
    # Load the trained model
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
else:
    print(f"Model file not found at {model_path}. Please check the path and filename.")

# Proceed only if the model was loaded successfully
if 'model' in locals():
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame to 300x300 for the model
        img_resized = cv2.resize(frame, (300, 300))

        # Normalize the image (scale pixel values to [0, 1])
        img_resized = img_resized.astype('float32') / 255.0

        # Expand dimensions to match the model input shape (1, 300, 300, 3)
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Make predictions
        predictions = model.predict(img_expanded)

        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Real-Time Sign Language Prediction', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when everything is done
    cap.release()
    cv2.destroyAllWindows()