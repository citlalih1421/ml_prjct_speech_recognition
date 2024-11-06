import os
import kaggle

# Define the datasets and their respective paths
datasets = {
    'rtatman/speech-accent-archive': 'speech-accent-archive',
    'mozillaorg/common-voice': 'common-voice',  
}

for dataset_name, folder_name in datasets.items():
    dataset_path = os.path.expanduser(f'~/ml_prjct/data/raw/{folder_name}')
    os.makedirs(dataset_path, exist_ok=True)

    # Download the dataset
    kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
    print(f"Dataset '{folder_name}' downloaded and unzipped successfully!")






'''
import os
import kaggle

# Set the path where you want to save the dataset
accent_dataset_path = os.path.expanduser('~/ml_prjct/data/raw/speech-accent-archive')
common_voice_dataset_path = os.path.expanduser('~/ml_prjct/data/raw/common-voice')

# Create the directory if it doesn't exist
os.makedirs(accent_dataset_path, exist_ok=True)
os.makedirs(common_voice_dataset_path, exist_ok=True)

# Download the dataset
kaggle.api.dataset_download_files('rtatman/speech-accent-archive', path= accent_dataset_path, unzip=True)
kaggle.api.dataset_download_files('mozillaorg/common-voice',path=common_voice_dataset_path, unzip=True)


print("Dataset downloaded and unzipped successfully!")
'''
