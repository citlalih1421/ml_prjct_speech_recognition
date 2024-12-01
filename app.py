import os
import sys
import torch
from transformers import pipeline
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))

from transcription_transformer_model import asr_transformer_func
from summarization_model import summarizing_func

# Initialize Flask app
app = Flask(__name__)

#file upload configuration
UPLOAD_FOLDER = 'uploads'  
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = 0 if torch.cuda.is_available() else -1

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main page route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', summary='', error='')


# Upload route for handling asynchronous file uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Please upload a .wav, .mp3, or .m4a file."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        print(f"Processing file: {file_path}")
        
        # Transcription
        print("Starting transcription...")
        transcription_file = os.path.splitext(file_path)[0] + '.txt'
        asr_transformer_func(file_path, transcription_file, device)
        print("Transcription completed.")
        
        # Summarization
        print("Starting summarization...")
        summarizing_func(transcription_file, device)
        print("Summarization completed.")
        
        # Read summarized content
        with open(transcription_file, 'r') as f:
            summarized_content = f.read()

        return jsonify({"summary": summarized_content}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing the audio file: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)