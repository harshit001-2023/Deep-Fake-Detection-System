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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = None


def initialize_detector():
    global detector
    try:
        if detector is None:
            logger.info("Initializing MTCNN detector...")
            detector = MTCNN()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MTCNN detector: {e}")
        return False


def _build_architecture(use_pretrained=True):
    """Build the model architecture, with fallback to no pretrained weights."""
    weights = 'imagenet' if use_pretrained else None
    base_model = MobileNetV2(weights=weights, include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    frame_input = Input(shape=(224, 224, 3))
    x = base_model(frame_input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.3)(x)
    frame_features = Dense(256, activation='relu', kernel_regularizer='l2')(x)

    sequence_input = Input(shape=(5, 224, 224, 3))
    time_distributed = TimeDistributed(Model(inputs=frame_input, outputs=frame_features))(sequence_input)
    lstm_out = LSTM(128, return_sequences=False)(time_distributed)
    x = Dense(64, activation='relu')(lstm_out)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model(load_saved=True):
    """Build or load the model. Falls back gracefully if weights download fails."""
    try:
        model_path = "deepfake_model.h5"
        if load_saved and os.path.exists(model_path):
            logger.info("Loading saved model from disk...")
            return load_model(model_path)

        # Try with pretrained ImageNet weights first
        try:
            logger.info("Building model with pretrained ImageNet weights...")
            model = _build_architecture(use_pretrained=True)
            logger.info("Model built successfully with pretrained weights.")
            return model
        except Exception as e:
            logger.warning(f"Could not load pretrained weights ({e}). Falling back to random init.")

        # Fallback: build without pretrained weights (random initialisation)
        logger.info("Building model with random initialisation (no pretrained weights)...")
        model = _build_architecture(use_pretrained=False)
        logger.info("Model built successfully (random weights — needs training for real accuracy).")
        return model

    except Exception as e:
        logger.error(f"Error building/loading model: {e}")
        return None


def process_video(video_path, progress_callback=None):
    """Process video frames, detecting faces with MTCNN."""
    try:
        if not initialize_detector():
            if progress_callback:
                progress_callback(0, 'Failed to initialize face detector')
            return None

        logger.info(f"Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            if progress_callback:
                progress_callback(0, 'Could not open video file')
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 5:
            logger.warning(f"Video too short (frames: {total_frames})")
            if progress_callback:
                progress_callback(0, 'Video is too short for analysis')
            cap.release()
            return None

        frames = []
        frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)

        for idx_num, idx in enumerate(frame_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            progress_percent = 10 + int((idx_num / len(frame_indices)) * 50)
            if progress_callback:
                progress_callback(progress_percent, f'Processing frame {idx_num} of {len(frame_indices)}')

            if not ret:
                logger.warning(f"Could not read frame {idx}")
                continue

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(frame_rgb)

                if detections:
                    max_area, max_box = 0, None
                    for detection in detections:
                        box = detection['box']
                        area = box[2] * box[3]
                        if area > max_area:
                            max_area = area
                            max_box = box

                    if max_box is not None:
                        x, y, w, h = max_box
                        face = frame[max(0, y):min(frame.shape[0], y + h),
                                     max(0, x):min(frame.shape[1], x + w)]
                        face = cv2.resize(face, (224, 224))
                        face = face.astype('float32') / 255.0
                        frames.append(face)
                else:
                    # No face detected — use the full resized frame as a fallback
                    logger.warning(f"No face in frame {idx}, using full frame as fallback")
                    fallback = cv2.resize(frame, (224, 224)).astype('float32') / 255.0
                    frames.append(fallback)

            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}")
                continue

        cap.release()

        if len(frames) >= 5:
            if progress_callback:
                progress_callback(65, 'Video frames prepared for prediction')
            return np.array(frames[:5])
        else:
            logger.warning(f"Insufficient frames processed: {len(frames)}/5")
            if progress_callback:
                progress_callback(0, 'Insufficient frames detected')
            return None

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        if progress_callback:
            progress_callback(0, 'Error processing video')
        return None


def predict_deepfake(video_path, model, frames=None, progress_callback=None):
    """Run inference and return (result_label, confidence_percent)."""
    try:
        if frames is None:
            if progress_callback:
                progress_callback(65, 'Extracting frames for prediction')
            frames = process_video(video_path, progress_callback=progress_callback)

        if frames is None:
            return "Error: Could not process video", 0

        if model is None:
            return "Error: Model not initialized", 0

        if progress_callback:
            progress_callback(75, 'Running sequence prediction')

        sequence = np.expand_dims(np.array(frames), axis=0)
        prediction = float(model.predict(sequence, verbose=0)[0][0])
        logger.info(f"Raw prediction value: {prediction:.4f}")

        result = "Real" if prediction > 0.5 else "Fake"
        confidence = float((prediction if prediction > 0.5 else 1 - prediction) * 100)

        if progress_callback:
            progress_callback(90, 'Prediction complete')

        logger.info(f"Final: {result} ({confidence:.2f}% confidence)")
        return result, confidence

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        if progress_callback:
            progress_callback(0, 'Error during prediction')
        return "Error: An unexpected error occurred", 0


def load_videos(folder_path, label, max_videos=50):
    videos, labels = [], []
    try:
        if not os.path.exists(folder_path):
            return videos, labels
        video_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        processed = 0
        for video_file in video_files:
            if processed >= max_videos:
                break
            frames = process_video(os.path.join(folder_path, video_file))
            if frames is not None:
                videos.append(frames)
                labels.append(label)
                processed += 1
        return videos, labels
    except Exception as e:
        logger.error(f"Error loading videos: {e}")
        return videos, labels


def train_model(incremental=True):
    logger.info("Starting model training...")
    try:
        model = build_model(load_saved=True)
        if model is None:
            model = build_model(load_saved=False)

        celeb_real   = load_videos(os.path.join("dataset", "Celeb-real"),      1, 40)
        youtube_real = load_videos(os.path.join("dataset", "YouTube-real"),    1, 40)
        fake         = load_videos(os.path.join("dataset", "Celeb-synthesis"), 0, 80)

        all_videos = celeb_real[0] + youtube_real[0] + fake[0]
        all_labels = celeb_real[1] + youtube_real[1] + fake[1]

        if not all_videos:
            logger.error("No training data available")
            return False

        X = np.array(all_videos)
        y = np.array(all_labels)
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        epochs = 5 if (incremental and os.path.exists("deepfake_model.h5")) else 15
        model.fit(X, y, epochs=epochs, batch_size=8, validation_split=0.2, callbacks=[cb], verbose=1)
        model.save("deepfake_model.h5")

        # Persist history so the analytics dashboard can plot it
        try:
            import json as _json
            with open('training_history.json', 'w') as _f:
                _json.dump({k: [float(v) for v in vals]
                            for k, vals in history.history.items()}, _f)
        except Exception as _e:
            logger.warning(f"Could not save training history: {_e}")

        return True
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False


if __name__ == "__main__":
    train_model()
