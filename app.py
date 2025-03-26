import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import uuid
import datetime
import json
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'blood_group_detection_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['FILE_LIFETIME'] = 15 * 60  # 15 minutes in seconds (15 * 60 = 900)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
MODEL = None
CLASS_NAMES = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    global MODEL, CLASS_NAMES
    
    # Path to the saved model
    model_path = os.path.join('models', 'final_best_model.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the model
    MODEL = tf.keras.models.load_model(model_path)
    
    # Get class names
    dataset_dir = 'dataset'
    if os.path.exists(dataset_dir):
        CLASS_NAMES = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    else:
        CLASS_NAMES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    
    print(f"Model loaded successfully. Class names: {CLASS_NAMES}")

def cleanup_old_files():
    """Delete files older than configured lifetime from uploads and reports directories"""
    with app.app_context():
        current_time = datetime.datetime.now()
        directories = [app.config['UPLOAD_FOLDER'], 'reports']
        
        for directory in directories:
            if not os.path.isdir(directory):
                continue
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    file_age = (current_time - modification_time).total_seconds()
                    
                    if file_age > app.config['FILE_LIFETIME']:
                        try:
                            os.remove(file_path)
                            app.logger.info(f"Deleted old file: {file_path}")
                        except Exception as e:
                            app.logger.error(f"Error deleting {file_path}: {e}")

def preprocess_image(image_data):
    """Preprocess image data for model prediction"""
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')
    img = img.resize((96, 103))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_blood_group(image_data):
    """Process image and predict blood group"""
    processed_image = preprocess_image(image_data)
    predictions = MODEL.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_index])
    
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        {
            "class": CLASS_NAMES[idx],
            "confidence": float(predictions[0][idx])
        }
        for idx in top_indices
    ]
    
    return {
        "blood_group": CLASS_NAMES[predicted_class_index],
        "confidence": confidence,
        "top_predictions": top_predictions
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(file_path)
            
            # Read the file for prediction
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Make prediction
            try:
                result = predict_blood_group(image_data)
                
                # Save result to session for report generation
                session['detection_result'] = {
                    'blood_group': result['blood_group'],
                    'confidence': result['confidence'],
                    'image_path': file_path,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'top_predictions': result['top_predictions']
                }
                
                return render_template(
                    'detect.html',
                    prediction=result,
                    image_file=unique_filename
                )
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg, bmp')
            return redirect(request.url)
    
    return render_template('detect.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generate-report')
def generate_report():
    if 'detection_result' not in session:
        flash('No detection result available. Please upload an image first.')
        return redirect(url_for('detect'))
    
    result = session['detection_result']
    return render_template('generate-report.html', result=result)

@app.route('/download-report')
def download_report():
    if 'detection_result' not in session:
        return jsonify({'error': 'No detection result available'}), 400
    
    result = session['detection_result']
    
    # Create a report JSON
    report_data = {
        'patient_id': request.args.get('patient_id', 'Unknown'),
        'patient_name': request.args.get('patient_name', 'Unknown'),
        'blood_group': result['blood_group'],
        'confidence': result['confidence'],
        'test_date': result['timestamp'],
        'notes': request.args.get('notes', '')
    }
    
    # Save report to file
    report_id = uuid.uuid4().hex
    report_path = os.path.join('reports', f"{report_id}.json")
    os.makedirs('reports', exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    
    return jsonify({
        'success': True,
        'report': report_data,
        'report_id': report_id
    })

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        image_data = file.read()
        
        try:
            result = predict_blood_group(image_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.svg', mimetype='image/vnd.microsoft.icon')

# Error handling
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large'}), 413

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    load_model()
    
    # Initialize and start scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_old_files, 'interval', minutes=5)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    
    app.run(debug=True, host='0.0.0.0', port=5000)