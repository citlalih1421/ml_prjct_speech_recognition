import os
from transformers import pipeline



def asr_transformer_func(audio_path, output_txt_path):
    asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=device)
    """
    Transcribe an audio file and save the transcription to a text file.

    Args:
        audio_path (str): Path to the input audio file.
        output_txt_path (str): Path to the output text file.

    Returns:
        None
    """
    transcription = asr_pipeline(audio_path)["text"]
    
    with open(output_txt_path, 'w') as f:
        f.write(transcription)