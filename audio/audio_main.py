import numpy as np
import os
import joblib
import librosa
import torch
import timm
import torch.nn.functional as F
import pandas as pd

# Function to compute mel spectrogram
def mel(arr, sr=32_000):
    arr = arr * 1024
    spec = librosa.feature.melspectrogram(y=arr, sr=sr,
                                          n_fft=1024, hop_length=500, n_mels=128,
                                          fmin=40, fmax=15000, power=2.0)
    spec = spec.astype('float32')
    return spec

# Paths for required files
model_paths = [
    'models_weights/effnet_seg20_80low.ckpt',
]
train_metadata_path = 'data/train_metadata.csv'

SHAPE = [48, 1, 128, 320 * 2]

# Load and initialize models
models = []
for model_path in model_paths:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model_name = 'efficientnet_b0'
    model = timm.create_model(model_name, pretrained=None, num_classes=182, in_chans=SHAPE[1])
    new_state_dict = {key[6:]: val for key, val in state_dict['state_dict'].items() if key.startswith('model.')}
    model.load_state_dict(new_state_dict)
    model.eval()
    models.append(model)

#print(f'{len(models)} models are ready')

# Load metadata for label mapping
data = pd.read_csv(train_metadata_path)
LABELS = sorted(list(data['primary_label'].unique()))

# Function to process and predict a single audio sample
def predict_single_audio(audio_path, models= models, LABELS= LABELS):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=32_000)
    chunk_size = sr * 5
    padding = chunk_size - (len(audio) % chunk_size)
    audio = np.pad(audio, (0, padding), mode='constant')
    chunks = audio.reshape(-1, chunk_size)

    # Compute mel spectrograms
    chunks_mel = mel(chunks, sr=sr)[:, :, :320]
    chunks = chunks_mel[:, np.newaxis, :, :]
    chunks = np.concatenate([chunks, chunks, chunks], axis=-1)
    chunks = librosa.power_to_db(chunks, ref=1, top_db=100.0).astype('float32')
    chunks = torch.from_numpy(chunks)

    # Run predictions
    preds = []
    with torch.no_grad():
        for model in models:
            output = model(chunks)  # Output is a PyTorch tensor
            preds.append(output.numpy())  # Convert to NumPy array

    # Aggregate predictions
    preds = np.mean(preds, axis=0)
    preds = torch.sigmoid(torch.tensor(preds)).numpy()  # Convert to NumPy array after sigmoid

    # Ensure predictions are a 1D array
    preds = preds.flatten()  # Flatten to ensure correct shape

    # Create label-score mapping
    prediction_dict = {label: score for label, score in zip(LABELS, preds)}
    names_map = {'comior1':'Common Iora', 'gloibi':'Glossy Ibis', 'houspa':'House Sparrow', 'indpit1':'Indian Pitta', 'indrol2':'Indian Roller', 'rocpig':'Rock Dove', 'lewduc1':'Lesser Whistling Duck'}
    # Print top 10 predictions
    #print(f"Predictions for '{audio_path}':")
    for label, score in sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)[:1]:
        #print(f"{names_map[label]}: {score:.4f}")
        return (names_map[label], score)

# Example usage
audio_files = ['barswa2', 'comior1', 'gloibi1', 'houspa1', 'indpit1', 'indrol1', 'rocpig1']  # Path to your audio file
# Ensure the provided audio file exists and predict
for audio_file in audio_files:
	if os.path.exists('data/AUDIO_FILES/' + audio_file+'.ogg'):
    		print(predict_single_audio('data/AUDIO_FILES/' + audio_file+'.ogg')[0])
	else:
   		print("File does not exist. Please check the path and try again.")
