import os
import json
import numpy as np
import logging
from tensorflow.keras.callbacks import EarlyStopping

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model import build_model, load_videos
except ImportError as e:
    logger.error(f"Failed to import required modules from 'model': {e}")
    exit(1)

def train_deepfake_detector():
    logger.info("Starting DeepFake detection model training...")
    
    try:
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Load or build the model
        model = build_model(load_saved=True)
        if model is None:
            logger.warning("No saved model found, building new one...")
            model = build_model(load_saved=False)
        
        if model is None:
            logger.error("Failed to build or load the model.")
            return False
        
        # Define dataset paths
        celeb_real_path = os.path.join("dataset", "Celeb-real")
        youtube_real_path = os.path.join("dataset", "YouTube-real")
        fake_path = os.path.join("dataset", "Celeb-synthesis")
        
        # Check if dataset directories exist
        for path in [celeb_real_path, youtube_real_path, fake_path]:
            if not os.path.exists(path):
                logger.error(f"Dataset directory not found: {path}")
                return False
        
        # Load dataset
        max_videos_per_class = 40
        logger.info(f"Loading videos from {celeb_real_path}...")
        celeb_real_videos, celeb_real_labels = load_videos(celeb_real_path, 1, max_videos_per_class)
        logger.info(f"Loading videos from {youtube_real_path}...")
        youtube_real_videos, youtube_real_labels = load_videos(youtube_real_path, 1, max_videos_per_class)
        logger.info(f"Loading videos from {fake_path}...")
        fake_videos, fake_labels = load_videos(fake_path, 0, max_videos_per_class * 2)

        # Combine all data
        X_train = np.array(celeb_real_videos + youtube_real_videos + fake_videos)
        y_train = np.array(celeb_real_labels + youtube_real_labels + fake_labels)
        
        if len(X_train) == 0:
            logger.error("No training data available. Check dataset directories and video files.")
            return False

        # Shuffle data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        logger.info(f"Training with {len(X_train)} videos (Real: {np.sum(y_train == 1)}, Fake: {np.sum(y_train == 0)})")

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        
        # Train the model
        logger.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=8,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model
        model_path = 'models/deepfake_detector.h5'
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save training history
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],  # Ensure float values
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        history_path = 'models/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)  # Pretty print for readability
        logger.info(f"Training history saved to {history_path}")
        logger.debug(f"History data: {json.dumps(history_dict, indent=4)}")  # Log the data for verification

        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False

if __name__ == "__main__":
    success = train_deepfake_detector()
    if success:
        logger.info("Training completed successfully.")
    else:
        logger.error("Training failed.")