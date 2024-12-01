#TODO:  Implement a bi-directional LSTM or GRU-based transcription model with a CTC loss function. 
import torch
import os
import librosa
import torch.nn as nn
from torch.nn import functional as F
import pyttsx3
from gtts import gTTS
import os

os.chdir("../../")

class CTCTranscriptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, bidirectional=True):
        super(CTCTranscriptionModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)
        output = F.log_softmax(output, dim=2)
        return output.transpose(0, 1)

    def compute_ctc_loss(self, logits, targets, input_lengths, target_lengths):
        ctc_loss = nn.CTCLoss(blank=0)
        return ctc_loss(logits, targets, input_lengths, target_lengths)

    def transcribe(self, logits, class_to_text_mapping):
        logits = logits.detach().cpu()
        pred_indices = torch.argmax(logits, dim=2)  # Greedy decoding
        transcriptions = []
        for indices in pred_indices.T:
            transcription = []
            prev_idx = None
            for idx in indices:
                idx = idx.item()
                if idx != prev_idx and idx != 0:  # Skip blanks and duplicates
                    transcription.append(class_to_text_mapping.get(idx, ""))
                prev_idx = idx
            transcriptions.append("".join(transcription))
        return transcriptions
        
    def transcribe_to_audio(self, transcription, output_file="output_audio.mp3", language="en"):
        """
        Converts transcription text into audio using gTTS.

        Args:
            transcription (str): The text to be converted into audio.
            output_file (str): Path to save the generated audio file.
            language (str): Language code for the TTS (default is 'en' for English).

        Returns:
            str: Path to the saved audio file.
        """
        # if not transcription or transcription.strip() == "":
        #     raise ValueError("Transcription is empty or invalid.")
        
        tts = gTTS(text=transcription, lang=language)
        tts.save(output_file)
        return output_file
    



def preprocess_audio(file_path, n_mfcc=40):
    """
    Preprocesses an audio file to extract MFCC features.

    Args:
    - file_path (str): Path to the audio file.
    - n_mfcc (int): Number of MFCC features to extract.

    Returns:
    - torch.Tensor: Extracted features as a PyTorch tensor.
    """
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCC features
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)  # Normalize MFCCs
    return torch.tensor(mfcc.T, dtype=torch.float32)  # Transpose to shape (time_steps, n_mfcc)


def process_audio_files(cv_train, model, class_to_text_mapping):
    """
    Processes all audio files in the cv_train directory and transcribes them.

    Args:
    - cv_train (str): Path to the directory containing audio files.
    - model (CTCTranscriptionModel): The trained transcription model.
    - class_to_text_mapping (dict): Mapping from class indices to text.

    Returns:
    - dict: A dictionary of transcriptions where keys are file names and values are the transcribed text.
    """
    # Check if directory exists
    if not os.path.exists(cv_train):
        print(f"Directory {cv_train} does not exist. Creating it.")
        os.makedirs(cv_train)  # Create directory if it doesn't exist
        return {}

    # List all audio files
    audio_files = [os.path.join(cv_train, f) for f in os.listdir(cv_train) if f.endswith(('.wav', '.mp3'))]
    if not audio_files:
        print(f"No audio files found in {cv_train}.")
        return {}

    transcriptions = {}

    for file_path in audio_files:
        try:

            # Preprocess audio
            input_features = preprocess_audio(file_path)

            # Add batch dimension
            input_tensor = input_features.unsqueeze(0)

            # Forward pass through the model
            logits = model(input_tensor)  # Shape: (T, N, C)

            # Decode transcription
            transcription_list = model.transcribe(logits, class_to_text_mapping)
            print(transcription_list)
            transcription = " ".join(transcription_list)

            if not transcription or transcription.strip() == "":
                raise ValueError("Transcription is empty or invalid.")
            
            # Store transcription
            transcriptions[file_path] = transcription
            print(f"Transcription for {file_path}: {transcription}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            raise e

    return transcriptions


if __name__ == "__main__":
    # Model parameters
    input_size = 40  # Feature dimension, e.g., MFCCs
    hidden_size = 128
    num_classes = 29  # Number of output classes, including blank
    num_layers = 2

    # Instantiate the model
    model = CTCTranscriptionModel(input_size, hidden_size, num_classes, num_layers)

    # Example class-to-text mapping
    class_to_text_mapping = {
        0: "", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g",
        8: "h", 9: "i", 10: "j", 11: "k", 12: "l", 13: "m", 14: "n",
        15: "o", 16: "p", 17: "q", 18: "r", 19: "s", 20: "t", 21: "u",
        22: "v", 23: "w", 24: "x", 25: "y", 26: "z", 27: " ", 28: "."
    }

    # Define cv_train directory path
    base_path = os.path.expanduser('./ml_prjct_speech_recognition')
    cv_train = os.path.join(base_path, 'data', 'raw', 'common-voice', 'cv-valid-train')
    # Process audio files
    transcriptions = process_audio_files(cv_train, model, class_to_text_mapping)

    print("\nFinal Transcriptions:")
    for file, transcription in transcriptions.items():
        print(f"{file}: {transcription}")
