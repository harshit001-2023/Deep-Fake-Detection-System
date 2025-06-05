import cv2
import numpy as np
import os
import logging
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from mtcnn import MTCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MTCNN detector
detector = None  # Will be initialized when needed

def initialize_detector():
    """Initialize MTCNN detector with error handling"""
    global detector
    try:
        if detector is None:
            logger.info("Initializing MTCNN detector...")
            detector = MTCNN()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MTCNN detector: {e}")
        return False

def build_model(load_saved=True):
    """Build or load the model with error handling"""
    try:
        model_path = "deepfake_model.h5"
        if load_saved and os.path.exists(model_path):
            logger.info("Loading saved model...")
            return load_model(model_path)

        logger.info("Building new model...")
        # Base model: MobileNetV2 for feature extraction
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create a model for processing individual frames
        frame_input = Input(shape=(224, 224, 3))
        x = base_model(frame_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.3)(x)  # Increased dropout
        frame_features = Dense(256, activation='relu', kernel_regularizer='l2')(x)

        # Create a sequence model with LSTM
        sequence_input = Input(shape=(5, 224, 224, 3))  # 5 frames
        time_distributed = TimeDistributed(Model(inputs=frame_input, outputs=frame_features))(sequence_input)
        lstm_out = LSTM(128, return_sequences=False)(time_distributed)
        x = Dense(64, activation='relu')(lstm_out)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=sequence_input, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        logger.info("Model built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building/loading model: {e}")
        return None

def process_video(video_path):
    """Process video with improved error handling and logging"""
    try:
        if not initialize_detector():
            return None

        logger.info(f"Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 5:
            logger.warning(f"Video too short (frames: {total_frames})")
            return None

        frames = []
        frame_indices = np.linspace(0, total_frames-1, 5, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {idx}")
                continue
                
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(frame_rgb)
                
                if detections:
                    # Get the largest face
                    max_area = 0
                    max_box = None
                    for detection in detections:
                        box = detection['box']
                        area = box[2] * box[3]
                        if area > max_area:
                            max_area = area
                            max_box = box
                    
                    if max_box is not None:
                        # Extract and preprocess face
                        x, y, w, h = max_box
                        face = frame[max(0, y):min(frame.shape[0], y+h), 
                                   max(0, x):min(frame.shape[1], x+w)]
                        face = cv2.resize(face, (224, 224))
                        face = face.astype('float32') / 255.0
                        frames.append(face)
                else:
                    logger.warning(f"No face detected in frame {idx}")
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}")
                continue
        
        cap.release()
        
        if len(frames) == 5:
            return np.array(frames)
        else:
            logger.warning(f"Insufficient frames processed: {len(frames)}/5")
            return None
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return None

def load_videos(folder_path, label, max_videos=50):
    """Load videos with improved error handling and progress tracking"""
    videos = []
    labels = []
    
    try:
        if not os.path.exists(folder_path):
            logger.error(f"Path does not exist: {folder_path}")
            return videos, labels
            
        video_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            logger.warning(f"No video files found in {folder_path}")
            return videos, labels
            
        logger.info(f"Found {len(video_files)} videos in {folder_path}")
        processed = 0
        
        for video_file in video_files:
            if processed >= max_videos:
                break
                
            video_path = os.path.join(folder_path, video_file)
            processed_frames = process_video(video_path)
            
            if processed_frames is not None:
                videos.append(processed_frames)
                labels.append(label)
                processed += 1
                logger.info(f"Processed {processed}/{min(max_videos, len(video_files))} "
                          f"videos from {os.path.basename(folder_path)}")
        
        logger.info(f"Successfully loaded {len(videos)} videos from {folder_path}")
        return videos, labels
    except Exception as e:
        logger.error(f"Error loading videos from {folder_path}: {e}")
        return videos, labels

def predict_deepfake(video_path, model):
    """Predict with improved error handling and validation"""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return "Error: Video file not found", 0
            
        if model is None:
            logger.error("Model not initialized")
            return "Error: Model not initialized", 0
            
        frames = process_video(video_path)
        if frames is None:
            logger.error("Could not process video frames")
            return "Error: Could not process video", 0
            
        # Stack frames into a sequence
        sequence = np.array(frames)
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        
        prediction = model.predict(sequence, verbose=0)[0][0]
        logger.info(f"Raw prediction value: {prediction}")
        
        result = "Real" if prediction > 0.5 else "Fake"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        confidence = float(confidence * 100)  # Convert to percentage
        
        logger.info(f"Final prediction: {result} with {confidence:.2f}% confidence")
        return result, confidence
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return "Error: An unexpected error occurred", 0

def train_model(incremental=True):
    logger.info("Starting DeepFake detection model training...")
    
    try:
        model = build_model(load_saved=True)  # Load existing model
        if model is None:
            logger.warning("No saved model found, building new one...")
            model = build_model(load_saved=False)
        
        # Load original dataset
        celeb_real_path = os.path.join("dataset", "Celeb-real")
        youtube_real_path = os.path.join("dataset", "YouTube-real")
        fake_path = os.path.join("dataset", "Celeb-synthesis")
        
        max_videos_per_class = 40
        celeb_real_videos, celeb_real_labels = load_videos(celeb_real_path, 1, max_videos_per_class)
        youtube_real_videos, youtube_real_labels = load_videos(youtube_real_path, 1, max_videos_per_class)
        fake_videos, fake_labels = load_videos(fake_path, 0, max_videos_per_class * 2)

        # Load user-contributed data
        real_user_path = os.path.join("dataset", "Celeb-real")
        fake_user_path = os.path.join("dataset", "Celeb-synthesis")
        user_real_videos, user_real_labels = load_videos(real_user_path, 1, max_videos_per_class)
        user_fake_videos, user_fake_labels = load_videos(fake_user_path, 0, max_videos_per_class * 2)

        # Combine all data (no feedback data)
        X_train = np.array(celeb_real_videos + youtube_real_videos + fake_videos + 
                          user_real_videos + user_fake_videos)
        y_train = np.array(celeb_real_labels + youtube_real_labels + fake_labels + 
                          user_real_labels + user_fake_labels)
        
        if len(X_train) == 0:
            logger.error("No training data available")
            return False

        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        logger.info(f"Training with {len(X_train)} videos (Real: {np.sum(y_train == 1)}, Fake: {np.sum(y_train == 0)})")

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        
        if incremental and os.path.exists("deepfake_model.h5"):
            logger.info("Fine-tuning existing model...")
            history = model.fit(
                X_train, y_train,
                epochs=5,  # Fewer epochs for fine-tuning
                batch_size=8,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
        else:
            logger.info("Training new model from scratch...")
            history = model.fit(
                X_train, y_train,
                epochs=15,
                batch_size=8,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
        
        model.save("deepfake_model.h5")
        val_accuracy = max(history.history['val_accuracy'])
        logger.info(f"Best validation accuracy: {val_accuracy:.2%}")
        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False

if __name__ == "__main__":
    if train_model():
        logger.info("Training completed successfully")
    else:
        logger.error("Training failed")