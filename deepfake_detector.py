import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained deepfake detection model
model = load_model('deepfake_detector_advanced_v2.h5')

# Initialize face detector
face_detector = MTCNN()

# Initialize DeepFace for face recognition
from deepface import DeepFace

class DeepFakeDetector:
    def __init__(self):
        # Initialize the base model (MobileNetV2)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        # Preprocess and predict
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        return float(prediction[0][0])

    def detect_faces(self, image_path):
        # Detect faces using MTCNN
        try:
            if detector is None:
                detector = MTCNN()
            
            # Load and preprocess the image
            img = Image.open(image_path)
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Detect faces
            faces = detector.detect_faces(img_array)
            return faces
        except Exception as e:
            logger.error(f"Failed to detect faces: {e}")
            return []

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame temporarily and process
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            prediction = self.predict(temp_path)
            frame_predictions.append(prediction)
            
        cap.release()
        return np.mean(frame_predictions)

def main():
    # Initialize detector
    detector = DeepFakeDetector()
    
    # Example usage
    print("DeepFake Detector initialized!")
    print("To analyze an image, use: detector.predict('path_to_image')")
    print("To analyze a video, use: detector.analyze_video('path_to_video')")

if __name__ == "__main__":
    main()
