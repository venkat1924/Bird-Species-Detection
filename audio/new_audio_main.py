import numpy as np
import os
import joblib
import librosa
import torch
import timm
import pandas as pd

# Function to compute mel spectrogram
def mel(arr, sr=32_000):
    arr = arr * 1024
    spec = librosa.feature.melspectrogram(y=arr, sr=sr,
                                          n_fft=1024, hop_length=500, n_mels=128,
                                          fmin=40, fmax=15000, power=2.0)
    spec = spec.astype('float32')
    return spec

# Initialize models and labels
class AudioPredictor:
    def __init__(self, model_paths= ['audio/models_weights/effnet_seg20_80low.ckpt',], train_metadata_path = 'audio/data/train_metadata.csv'):
        self.models = []
        self.SHAPE = [48, 1, 128, 320 * 2]
        
        for model_path in model_paths:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            model_name = 'efficientnet_b0'
            model = timm.create_model(model_name, pretrained=None, num_classes=182, in_chans=self.SHAPE[1])
            new_state_dict = {key[6:]: val for key, val in state_dict['state_dict'].items() if key.startswith('model.')}
            model.load_state_dict(new_state_dict)
            model.eval()
            self.models.append(model)

        data = pd.read_csv(train_metadata_path)
        self.LABELS = sorted(list(data['primary_label'].unique()))
        
        self.names_map = {
            'comior1': 'Common Iora',
            'gloibi': 'Glossy Ibis',
            'houspa': 'House Sparrow',
            'indpit1': 'Indian Pitta',
            'indrol2': 'Indian Roller',
            'rocpig': 'Rock Dove',
            'lewduc1': 'Lesser Whistling Duck'
        }

    def predict_single_audio(self, audio_path):
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
            for model in self.models:
                output = model(chunks)  # Output is a PyTorch tensor
                preds.append(output.numpy())  # Convert to NumPy array

        # Aggregate predictions
        preds = np.mean(preds, axis=0)
        preds = torch.sigmoid(torch.tensor(preds)).numpy()  # Convert to NumPy array after sigmoid

        # Ensure predictions are a 1D array
        preds = preds.flatten()  # Flatten to ensure correct shape

        # Create label-score mapping
        prediction_dict = {label: score for label, score in zip(self.LABELS, preds)}

        # Return the top prediction
        top_prediction = max(prediction_dict.items(), key=lambda x: x[1])
        label, score = top_prediction
        return self.names_map.get(label, label), score

# Example of initialization and usage
if __name__ == "__main__":
    model_paths = [
        'models_weights/effnet_seg20_80low.ckpt',
    ]
    train_metadata_path = 'data/train_metadata.csv'

    predictor = AudioPredictor(model_paths, train_metadata_path)

    audio_files = ['barswa2', 'comior1', 'gloibi1', 'houspa1', 'indpit1', 'indrol1', 'rocpig1']  # List of audio files
    for audio_file in audio_files:
        audio_path = f'data/AUDIO_FILES/{audio_file}.ogg'
        if os.path.exists(audio_path):
            label, score = predictor.predict_single_audio(audio_path)
            print(f"{label}: {score:.4f}")
        else:
            print(f"File does not exist: {audio_path}")
