import os
import librosa 
import torch
import numpy as np


#& define the paths used (input and output)
#* input paths
commmon_voice_dir = os.path.expanduser('~/ml_prjct/data/raw/common-voice')
accents_dir = os.path.expanduser('~/ml_prjct/data/processed')
#*output paths
common_voice_processed_dir = "data/processed/common-voice/"
accents_processed_dir = "data/processed/accents/"

#* create output directories if they don't exist
os.makedirs(common_voice_processed_dir, exist_ok = True)
os.makedirs(accents_processed_dir, exist_ok = True)



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
process_dataset(common_voice_dir, common_voice_processed_dir)

# Process Accents dataset
## process_dataset(accents_dir, accents_processed_dir)