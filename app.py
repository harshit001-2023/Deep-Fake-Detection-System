from flask import Flask, request, render_template, Response, session, jsonify, flash
import os
import cv2
from model import build_model, predict_deepfake, process_video, train_model
import numpy as np
from mtcnn import MTCNN
import base64
from io import BytesIO
import json
import pandas as pd
import plotly
import plotly.express as px
from datetime import datetime
import logging
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

detector = None
model = None
analysis_data = {'confidence_scores': [], 'user_feedback': [], 'detection_metrics': {}}

def save_analysis_data():
    with open('analysis_data.json', 'w') as f:
        json.dump({'confidence_scores': analysis_data['confidence_scores'], 
                   'detection_metrics': analysis_data['detection_metrics']}, f, default=str)

def load_analysis_data():
    global analysis_data
    try:
        with open('analysis_data.json', 'r') as f:
            data = json.load(f)
            analysis_data = {'confidence_scores': data.get('confidence_scores', []), 
                             'detection_metrics': data.get('detection_metrics', {})}
    except FileNotFoundError:
        analysis_data = {'confidence_scores': [], 'detection_metrics': {}}

def initialize_app():
    global detector, model
    try:
        required_dirs = ['uploads', 'static/frames', 'dataset/Celeb-real', 'dataset/Celeb-synthesis']
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        load_analysis_data()
        
        logger.info("Initializing MTCNN detector...")
        detector = MTCNN()
        
        logger.info("Loading deep learning model...")
        model = build_model(load_saved=True)
        if model is None:
            logger.warning("No saved model found. Building new model...")
            model = build_model(load_saved=False)
            
        if model is None:
            logger.error("Failed to initialize model")
            return False
            
        logger.info("Application initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False

def to_base64(image):
    try:
        if isinstance(image, np.ndarray):
            return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        return ''
    except Exception as e:
        logger.error(f"Error in to_base64: {e}")
        return ''

app.jinja_env.filters['to_base64'] = to_base64

@app.route('/')
def index():
    if not model:
        flash("Warning: Model not fully initialized.", "warning")
    return render_template('index.html')

# @app.route('/detect')
# def detect():
#     session.pop('video_path', None)
#     if not model:
#         flash("Warning: Model not initialized.", "warning")
#     return render_template('detect.html')

@app.route('/train_page')
def train_page():
    stats = get_dataset_stats()
    return render_template('train.html', real_videos=stats['real_videos'], fake_videos=stats['fake_videos'])

@app.route('/about')
def about():
    if not model:
        flash("Warning: Model not fully initialized.", "warning")
    team = [
        {'name': 'Achal S. Surandase', 'photo': 'achal.jpg'},
        {'name': 'Sanika N. Tole', 'photo': 'sanika.jpg'},
        {'name': 'Pawan S. Bhandekar', 'photo': 'pawan.jpg'},
        {'name': 'Swaraj P. Patil', 'photo': 'swaraj.jpg'},
        {'name': 'Harshit M. Pande', 'photo': 'harshit.jpg'}
    ]
    return render_template('about.html', team=team)

@app.route('/upload', methods=['POST'])
def upload_file():
    if not model:
        flash("Error: Model not initialized.", "error")
        return render_template('upload.html')
        
    if 'file' not in request.files:
        flash("No file uploaded.", "error")
        return render_template('upload.html')
    
    file = request.files['file']
    if not file.filename:
        flash("No file selected.", "error")
        return render_template('upload.html')
        
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        flash("Invalid format. Use MP4, AVI, MOV, or MKV.", "error")
        return render_template('upload.html')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    secure_filename = f"{timestamp}_{os.path.splitext(file.filename)[0][:50]}{os.path.splitext(file.filename)[1]}"
    video_path = os.path.join("uploads", secure_filename)
    
    file.save(video_path)
    session['video_path'] = video_path
    flash("Video uploaded successfully.", "success")
    return render_template('upload.html', video_path=video_path)

@app.route('/upload', methods=['GET'])
def upload_page():
    if not model:
        flash("Warning: Model not initialized.", "warning")
    video_path = session.get('video_path')
    return render_template('upload.html', video_path=video_path)

@app.route('/cancel_upload', methods=['POST'])
def cancel_upload():
    video_path = session.pop('video_path', None)
    if video_path and os.path.exists(video_path):
        os.remove(video_path)
    return jsonify({'success': True})

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model:
        flash("Error: Model not initialized.", "error")
        return render_template('upload.html')

    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        flash("No video loaded.", "error")
        return render_template('upload.html')

    start_time = datetime.now()  # Start timing
    frames = process_video(video_path)
    if frames is None:
        flash("Error: Could not process video frames.", "error")
        return render_template('upload.html', video_path=video_path)

    result, confidence = predict_deepfake(video_path, model)
    if result.startswith("Error"):
        flash(f"Error during analysis: {result}", "error")
        return render_template('upload.html', video_path=video_path)

    display_frames = [(frame * 255).astype(np.uint8) for frame in frames]

    face_detections = [detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in display_frames]
    face_quality = (sum(1 for det in face_detections if det) / len(face_detections)) * 100 if face_detections else 0
    face_quality_details = f"{sum(1 for det in face_detections if det)} of {len(face_detections)} frames had faces"

    frame_diffs = [np.mean(np.abs(frames[i] - frames[i+1])) for i in range(len(frames)-1)]
    frame_consistency = 100 - (np.std(frame_diffs) * 100 if frame_diffs else 0)
    frame_consistency = max(0, min(100, frame_consistency))
    frame_analysis_details = f"Temporal stability (std: {np.std(frame_diffs):.2f})"

    model_confidence = confidence
    model_analysis_details = f"Deep learning confidence: {confidence:.1f}%"

    analysis_data['confidence_scores'].append({
        'video': os.path.basename(video_path),
        'confidence': confidence,
        'timestamp': datetime.now()
    })
    save_analysis_data()
    logger.info(f"Analysis data updated: {analysis_data['confidence_scores'][-1]}")

    end_time = datetime.now()  # End timing
    analysis_time = (end_time - start_time).total_seconds()  # Calculate time in seconds

    flash("Analysis completed.", "success")
    return render_template('upload.html', 
                         video_path=video_path, 
                         result=result,
                         confidence=confidence,
                         frames=display_frames,
                         face_quality=face_quality,
                         frame_consistency=frame_consistency,
                         model_confidence=model_confidence,
                         face_quality_details=face_quality_details,
                         frame_analysis_details=frame_analysis_details,
                         model_analysis_details=model_analysis_details,
                         analysis_time=f"{analysis_time:.2f} seconds")
                         
@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return "No video loaded", 400
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics')
def analytics():
    confidence_graph = create_confidence_graph()
    training_history_graph = create_training_history_graph()
    detection_metrics = get_detection_metrics()
    return render_template('analytics.html', 
                          confidence_graph=confidence_graph,
                          training_history_graph=training_history_graph,
                          metrics=detection_metrics)

def retrain_model_async():
    global model
    if train_model(incremental=True):
        logger.info("Model retrained successfully")
        model = build_model(load_saved=True)  # Reload updated model
    else:
        logger.error("Retraining failed")

def load_training_history():
    history_path = 'models/training_history.json'
    try:
        with open(history_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded training history from {history_path}")
            return data
    except FileNotFoundError:
        logger.warning(f"Training history file not found at {history_path}")
        return {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding training history JSON: {e}")
        return {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

def create_training_history_graph():
    history = load_training_history()
    if not history.get('accuracy') or len(history['accuracy']) == 0:
        logger.info("No training history data available to plot.")
        return None
    
    epochs = list(range(1, len(history['accuracy']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Training Accuracy': history['accuracy'],
        'Validation Accuracy': history['val_accuracy']
    })
    fig = px.line(df, x='Epoch', y=['Training Accuracy', 'Validation Accuracy'], 
                  title='Training History - Accuracy',
                  labels={'value': 'Accuracy', 'variable': 'Metric'})
    fig.update_layout(
        width=800, 
        height=400,
        plot_bgcolor='white',  # Ensure visibility on light backgrounds
        paper_bgcolor='white',
        font=dict(color='black')
    )
    fig.update_traces(line=dict(width=2))  # Thicker lines for clarity
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({'success': False, 'error': 'Invalid format'})
    
    label = request.form.get('label')
    if label not in ['real', 'fake']:
        return jsonify({'success': False, 'error': 'Specify real or fake'})
    
    target_dir = os.path.join('dataset', 'Celeb-real' if label == 'real' else 'Celeb-synthesis')
    os.makedirs(target_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    secure_filename = f"{timestamp}_{os.path.splitext(file.filename)[0][:50]}{os.path.splitext(file.filename)[1]}"
    video_path = os.path.join(target_dir, secure_filename)
    
    file.save(video_path)
    logger.info(f"Training video contributed: {video_path}, Label: {label}")

# Check dataset size and retrain
    stats = get_dataset_stats()
    if stats['total_videos'] % 5 == 0:  # Retrain every 20 new videos
        Thread(target=retrain_model_async).start()

    return jsonify({'success': True})

@app.route('/dataset_stats')
def dataset_stats():
    return jsonify(get_dataset_stats())

def get_dataset_stats():
    try:
        real_path = os.path.join('dataset', 'Celeb-real')
        fake_path = os.path.join('dataset', 'Celeb-synthesis')
        real_videos = len([f for f in os.listdir(real_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]) if os.path.exists(real_path) else 0
        fake_videos = len([f for f in os.listdir(fake_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]) if os.path.exists(fake_path) else 0
        return {'real_videos': real_videos, 'fake_videos': fake_videos, 'total_videos': real_videos + fake_videos}
    except Exception as e:
        logger.error(f"Error calculating dataset stats: {e}")
        return {'real_videos': 0, 'fake_videos': 0, 'total_videos': 0}

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

def create_confidence_graph():
    if not analysis_data['confidence_scores']:
        return None
    df = pd.DataFrame(analysis_data['confidence_scores'])
    fig = px.line(df, x='timestamp', y='confidence', title='Confidence Scores Over Time')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_feedback_graph():
    if not analysis_data['user_feedback']:
        return None
    df = pd.DataFrame(analysis_data['user_feedback'])
    fig = px.pie(df, names='label', title='User Feedback Distribution')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_detection_metrics():
    try:
        total_videos = len(analysis_data['confidence_scores'])
        avg_confidence = np.mean([score['confidence'] for score in analysis_data['confidence_scores']]) if total_videos > 0 else 0
        return {
            'total_videos': total_videos,
            'avg_confidence': f"{avg_confidence:.1f}%",
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {'total_videos': 0, 'avg_confidence': "0.0%"}

if __name__ == '__main__':
    if initialize_app():
        app.run(debug=False, host='0.0.0.0', port=5000)