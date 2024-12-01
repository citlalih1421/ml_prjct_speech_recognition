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

#asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h",device=device)
#summarizer = pipeline("summarization", model="jordiclive/flan-t5-3b-summarizer")


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
        # Process the file: Transcription
        transcription_file = os.path.splitext(file_path)[0] + '.txt'
        asr_transformer_func(file_path, transcription_file)

        # Process the file: Summarization
        summarizing_func(transcription_file)

        # Read summarized content
        with open(transcription_file, 'r') as f:
            summarized_content = f.read()

        return jsonify({"summary": summarized_content}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing the audio file: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

'''
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', summary='', error="No file uploaded.")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', summary='', error="No file selected.")
        
        if not allowed_file(file.filename):
            return render_template('index.html', summary='', error="Unsupported file format. Please upload a .wav, .mp3, or .m4a file.")
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path) 
        

        
        try:
            
            analysis_message = "Transcribing audio file..."
            transcription_file = os.path.splitext(file_path)[0] + '.txt'
            asr_transformer_func(file_path, transcription_file)

            analysis_message = "Summarizing transcript..."
            summarizing_func(transcription_file)

            with open(transcription_file, 'r') as f:
                summarized_content = f.read()

            return render_template('index.html', summary=summarized_content, error='')
        
        except Exception as e:
            # Handle any errors during processing
            return render_template('index.html', summary='', error=f"Error processing the audio file: {str(e)}")

    return render_template('index.html', summary='', error='')



if __name__ == '__main__':
    app.run(debug=True)
    '''