# DeepFake Detection Project

This project implements a deep learning-based DeepFake detection system using MobileNetV2 architecture.

## Features
- Image-based DeepFake detection
- Video analysis capabilities
- Pre-trained MobileNetV2 base model
- Easy-to-use Python interface

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the detector:
```bash
python deepfake_detector.py
```

## Usage
```python
from deepfake_detector import DeepFakeDetector

# Initialize the detector
detector = DeepFakeDetector()

# Analyze an image
result = detector.predict('path_to_image.jpg')
print(f"Probability of being fake: {result:.2f}")

# Analyze a video
video_result = detector.analyze_video('path_to_video.mp4')
print(f"Average probability of video being fake: {video_result:.2f}")
```

## Running the Application Manually

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd New DeepFake
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

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

![Welcome Pange](static/images/Screenshot (187).png)
