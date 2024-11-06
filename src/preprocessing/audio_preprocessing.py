import os
import librosa 
import torch
import numpy as np


#& define the paths used (input and output)
#* input paths
cv_dev = os.path.expanduser('~/ml_prjct_speech_recognition/data/raw/common-voice/cv-valid-dev')
cv_test = os.path.expanduser('~/ml_prjct_speech_recognition/data/raw/common-voice/cv-valid-test')
cv_train = os.path.expanduser('~/ml_prjct_speech_recognition/data/raw/common-voice/cv-valid-train')

#*output paths
base_path = os.path.expanduser('~/ml_prjct_speech_recognition')
cv_dev_processed = os.path.join(base_path, "data/processed/common-voice/cv-dev")
cv_test_processed = os.path.join(base_path, "data/processed/common-voice/cv-test")
cv_train_processed = os.path.join(base_path, "data/processed/common-voice/cv-train")

#* create output directories if they don't exist
os.makedirs(cv_dev_processed, exist_ok = True)
##os.makedirs(accents_processed_dir, exist_ok = True)



#& Extracting audio features (Mel spectrogram & MFCCs)
def extract_features(audio_path):
    #* load audio
    y, sr = librosa.load(audio_path, sr = None)
    
    #* Mel Spectrogram 
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
    
    #* MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    return mel_spec_db, mfccs


#& Processing audio files
def process_dataset(dataset_dir, output_dir, save_as_tensor=True):
    # os.walk allows us to iterate through all directories and files in the dataset_dir
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(root,file)
                mel_spec_db, mfccs = extract_features(audio_path)
                
                # Generate a base filename by removing the extension
                base_filename = os.path.splitext(file)[0]
                
                #* save the processed features
                if save_as_tensor:
                    # Save as PyTorch tensors
                    torch.save(torch.tensor(mel_spec_db), os.path.join(output_dir, f"{base_filename}_mel.pt"))
                    torch.save(torch.tensor(mfccs), os.path.join(output_dir, f"{base_filename}_mfcc.pt"))
                else:
                    # Save as numpy arrays
                    np.save(os.path.join(output_dir, f"{base_filename}_mel.npy"), mel_spec_db)
                    np.save(os.path.join(output_dir, f"{base_filename}_mfcc.npy"), mfccs)

                print(f"Processed: {file}")




# Process Common Voice dataset
process_dataset(cv_dev, cv_dev_processed)

# Process Accents dataset
## process_dataset(accents_dir, accents_processed_dir)