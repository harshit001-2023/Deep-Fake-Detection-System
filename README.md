# DeepFake Detection Project

A Flask-based DeepFake detection application that analyzes uploaded videos using MTCNN face detection and a deep learning model built with MobileNetV2 and LSTM.

## What this project does
- Shows a welcome page first, then moves to the upload page after clicking **Start Detecting**.
- Accepts MP4, AVI, MOV, and MKV video uploads.
- Extracts face frames from video using MTCNN.
- Uses a sequence model to classify the uploaded video as **Real** or **Fake**.
- Displays model confidence, face quality, frame consistency, and a progress indicator during analysis.

## Project structure
- `app.py`: Flask web server, upload and analysis routes, progress API.
- `model.py`: Model building, video frame extraction, and prediction logic.
- `templates/`: HTML templates for the welcome, upload, about, and analytics pages.
- `static/`: CSS, JavaScript, and image assets.
- `deepfake_model.h5`, `deepfake_detector_advanced_v2.h5`: Trained model weights used by the application.

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the app
From the project root:
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Usage
1. Open the main page.
2. Click **Start Detecting**.
3. Upload a video file.
4. Click **Analyze Video**.
5. Watch the progress bar and see the final classification result.

## Notes
- The app uses only CPU by default on native Windows for TensorFlow 2.12+.
- The upload page is the correct analysis entry point; the welcome page only introduces the workflow.
- The About page now links to the upload page rather than a missing `/detect` route.

## Requirements
The project depends on:
- Flask
- TensorFlow
- OpenCV
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- MTCNN
- pandas
- plotly

3. **Run the Flask application**:
   ```bash
   python app.py
   ```
   The application will start running on http://127.0.0.1:5000.

4. **Accessing on Mobile**:
   - Ensure your computer and mobile device are connected to the same Wi-Fi network.
   - Find your local IP address (e.g., 192.168.1.10) by running `ipconfig` (Windows) or `ifconfig` (Linux/Mac) in the terminal.
   - Replace `127.0.0.1` in the URL with your local IP address:
     - For example: http://192.168.1.10:5000
   - Open this URL in the mobile browser to access the application.

## Note
This implementation uses transfer learning with MobileNetV2 for efficient detection. The model needs to be trained with a dataset of real and fake images/videos for optimal performance.
