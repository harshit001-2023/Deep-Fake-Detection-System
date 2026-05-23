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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

detector = None
model = None
analysis_data = {'confidence_scores': [], 'user_feedback': [], 'detection_metrics': {}}


# ─── Persistence ────────────────────────────────────────────────────────────

def save_analysis_data():
    try:
        with open('analysis_data.json', 'w') as f:
            json.dump({
                'confidence_scores': analysis_data['confidence_scores'],
                'detection_metrics': analysis_data['detection_metrics']
            }, f, default=str)
    except Exception as e:
        logger.error(f"Error saving analysis data: {e}")


def load_analysis_data():
    global analysis_data
    try:
        with open('analysis_data.json', 'r') as f:
            data = json.load(f)
            analysis_data = {
                'confidence_scores': data.get('confidence_scores', []),
                'detection_metrics': data.get('detection_metrics', {}),
                'user_feedback': []
            }
    except FileNotFoundError:
        analysis_data = {'confidence_scores': [], 'detection_metrics': {}, 'user_feedback': []}


# ─── Initialisation ─────────────────────────────────────────────────────────

def initialize_app():
    global detector, model
    try:
        for directory in ['uploads', 'static/frames', 'dataset/Celeb-real', 'dataset/Celeb-synthesis']:
            os.makedirs(directory, exist_ok=True)

        load_analysis_data()

        logger.info("Initializing MTCNN detector...")
        detector = MTCNN()

        logger.info("Loading deep learning model...")
        model = build_model(load_saved=True)
        if model is None:
            logger.warning("No saved model found — building new model...")
            model = build_model(load_saved=False)

        if model is None:
            logger.error("Failed to initialize model")
            return False

        logger.info("Application initialised successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False


# ─── Helpers ────────────────────────────────────────────────────────────────

def to_base64(image):
    try:
        if isinstance(image, np.ndarray):
            return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        return ''
    except Exception as e:
        logger.error(f"Error in to_base64: {e}")
        return ''


app.jinja_env.filters['to_base64'] = to_base64


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if not model:
            flash("Model not yet initialised — please try again in a moment.", "error")
            return render_template('upload.html')

        if 'file' not in request.files:
            flash("No file uploaded.", "error")
            return render_template('upload.html')

        file = request.files['file']
        if not file.filename:
            flash("No file selected.", "error")
            return render_template('upload.html')

        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            flash("Invalid format. Please use MP4, AVI, MOV, or MKV.", "error")
            return render_template('upload.html')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = f"{timestamp}_{os.path.splitext(file.filename)[0][:50]}{os.path.splitext(file.filename)[1]}"
        video_path = os.path.join("uploads", safe_name)
        file.save(video_path)

        session['video_path'] = video_path
        flash("Video uploaded successfully.", "success")
        return render_template('upload.html', video_path=video_path)

    video_path = session.get('video_path')
    return render_template('upload.html', video_path=video_path)


@app.route('/about')
def about():
    team = [
        {'name': 'Achal S. Surandase',  'photo': 'achal.jpg'},
        {'name': 'Sanika N. Tole',      'photo': 'sanika.jpg'},
        {'name': 'Pawan S. Bhandekar',  'photo': 'pawan.jpg'},
        {'name': 'Swaraj P. Patil',     'photo': 'swaraj.jpg'},
        {'name': 'Harshit M. Pande',    'photo': 'harshit.jpg'},
    ]
    return render_template('about.html', team=team)


@app.route('/cancel_upload', methods=['POST'])
def cancel_upload():
    video_path = session.pop('video_path', None)
    if video_path and os.path.exists(video_path):
        os.remove(video_path)
    return jsonify({'success': True})


@app.route('/analyze', methods=['POST'])
def analyze():
    if not model:
        flash("Model not initialised.", "error")
        return render_template('upload.html')

    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        flash("No video loaded.", "error")
        return render_template('upload.html')

    start_time = datetime.now()
    frames = process_video(video_path)
    if frames is None:
        flash("Error: Could not extract frames from the video.", "error")
        return render_template('upload.html', video_path=video_path)

    result, confidence = predict_deepfake(video_path, model, frames=frames)
    if result.startswith("Error"):
        flash(f"Analysis error: {result}", "error")
        return render_template('upload.html', video_path=video_path)

    display_frames = [(frame * 255).astype(np.uint8) for frame in frames]

    # Face quality
    face_detections = [detector.detect_faces(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in display_frames]
    face_quality = (sum(1 for d in face_detections if d) / len(face_detections)) * 100
    face_quality_details = f"{sum(1 for d in face_detections if d)} of {len(face_detections)} frames contained faces"

    # Frame consistency
    frame_diffs = [np.mean(np.abs(frames[i] - frames[i + 1])) for i in range(len(frames) - 1)]
    frame_consistency = max(0, min(100, 100 - np.std(frame_diffs) * 100))
    frame_analysis_details = f"Temporal stability (std: {np.std(frame_diffs):.2f})"

    analysis_data['confidence_scores'].append({
        'video': os.path.basename(video_path),
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })
    save_analysis_data()

    analysis_time = (datetime.now() - start_time).total_seconds()

    flash("Analysis completed.", "success")
    return render_template('upload.html',
                           video_path=video_path,
                           result=result,
                           confidence=confidence,
                           frames=display_frames,
                           face_quality=face_quality,
                           frame_consistency=frame_consistency,
                           model_confidence=confidence,
                           face_quality_details=face_quality_details,
                           frame_analysis_details=frame_analysis_details,
                           model_analysis_details=f"Deep learning confidence: {confidence:.1f}%",
                           analysis_time=f"{analysis_time:.2f} seconds")


@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return "No video loaded", 400
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html',
                           confidence_graph=create_confidence_graph(),
                           metrics=get_detection_metrics(),
                           training_history_graph=load_training_history_graph(),
                           error=None)


@app.route('/train_page')
def train_page():
    stats = get_dataset_stats()
    return render_template('train.html',
                           real_videos=stats['real_videos'],
                           fake_videos=stats['fake_videos'])


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
        return jsonify({'success': False, 'error': 'Label must be real or fake'})

    target_dir = os.path.join('dataset', 'Celeb-real' if label == 'real' else 'Celeb-synthesis')
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = f"{timestamp}_{os.path.splitext(file.filename)[0][:50]}{os.path.splitext(file.filename)[1]}"
    file.save(os.path.join(target_dir, safe_name))
    return jsonify({'success': True})


@app.route('/dataset_stats')
def dataset_stats():
    return jsonify(get_dataset_stats())


# ─── Utilities ──────────────────────────────────────────────────────────────

def get_dataset_stats():
    try:
        def count(path):
            return len([f for f in os.listdir(path)
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]) if os.path.exists(path) else 0
        r = count(os.path.join('dataset', 'Celeb-real'))
        f = count(os.path.join('dataset', 'Celeb-synthesis'))
        return {'real_videos': r, 'fake_videos': f, 'total_videos': r + f}
    except Exception as e:
        logger.error(f"Dataset stats error: {e}")
        return {'real_videos': 0, 'fake_videos': 0, 'total_videos': 0}


def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    cap.release()


def load_training_history_graph():
    """Load training history from disk if it exists (saved by train_model)."""
    try:
        history_path = 'training_history.json'
        if not os.path.exists(history_path):
            return None
        with open(history_path, 'r') as f:
            history = json.load(f)
        epochs = list(range(1, len(history.get('accuracy', [])) + 1))
        fig = px.line(
            x=epochs,
            y=[history.get('accuracy', []), history.get('val_accuracy', [])],
            labels={'x': 'Epoch', 'value': 'Accuracy'},
            title='Training History — Accuracy'
        )
        fig.data[0].name = 'Train'
        fig.data[1].name = 'Validation'
        fig.update_layout(legend_title_text='Split')
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        logger.error(f"Error loading training history: {e}")
        return None


def create_confidence_graph():
    if not analysis_data['confidence_scores']:
        return None
    df = pd.DataFrame(analysis_data['confidence_scores'])
    fig = px.line(df, x='timestamp', y='confidence', title='Confidence Scores Over Time',
                  labels={'confidence': 'Confidence (%)', 'timestamp': 'Time'})
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_detection_metrics():
    try:
        scores = analysis_data['confidence_scores']
        total = len(scores)
        avg = np.mean([s['confidence'] for s in scores]) if total else 0
        return {'total_videos': total, 'avg_confidence': f"{avg:.1f}%"}
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {'total_videos': 0, 'avg_confidence': "0.0%"}


# ─── Startup ────────────────────────────────────────────────────────────────

# Initialise on startup; log a warning but do NOT crash if it fails.
# This lets Flask serve pages even while the model is loading in a retry loop.
_init_ok = initialize_app()
if not _init_ok:
    logger.warning("App initialisation failed — the model may be unavailable. "
                   "Check logs for details.")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
