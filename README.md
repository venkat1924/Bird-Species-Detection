# Ears and Eyes on the Sky: Multimodal Deep Learning for Bird Classification

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
This repository contains the code for the paper: **"Multimodal Deep Learning for Bird Species Classification: An Implementation and Analysis"**. The system implements a multimodal approach, leveraging both audio and visual data, for identifying bird species using deep learning techniques.

## Table of Contents
1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [System Architecture](#system-architecture)
4.  [Technology Stack](#technology-stack)
5.  [Setup and Installation](#setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Cloning the Repository](#cloning-the-repository)
    * [Setting up Virtual Environment](#setting-up-virtual-environment)
    * [Installing Dependencies](#installing-dependencies)
    * [Downloading Pre-trained Models](#downloading-pre-trained-models)
    * [Metadata Files](#metadata-files)
6.  [Usage](#usage)
7.  [Model Details](#model-details)
    * [Audio Model](#audio-model)
    * [Image Model](#image-model)
    * [Multimodal Fusion](#multimodal-fusion)
8.  [File Structure](#file-structure)
9.  [Limitations](#limitations)
10. [Future Work](#future-work)
11. [Citation](#citation)
12. [License](#license)
13. [Authors](#authors)

## Overview

Birds are crucial for ecological health, and their declining populations necessitate effective monitoring tools. This project implements a multimodal bird species classification system that combines audio (birdsong) and visual (images of birds) data for improved species identification. The system utilizes deep learning architectures: an EfficientNet-B0 for audio analysis via mel spectrograms and a custom ResNet34-like architecture for image analysis. These predictions are integrated within a Streamlit web application to provide a unified classification.

This system serves as a functional prototype for AI-driven ecological studies, supporting biodiversity monitoring and offering an educational tool for bird identification.

## Key Features

* **Multimodal Classification:** Identifies bird species using both audio recordings and images.
* **Deep Learning Models:**
    * **Audio:** EfficientNet-B0 processing mel spectrograms from 5-second audio chunks.
    * **Image:** Custom ResNet34-like CNN for image classification.
* **Modality-Specific Preprocessing:** Distinct pipelines for audio (resampling, chunking, mel spectrogram generation, dB conversion) and image (resizing, tensor conversion, scaling) data.
* **Multimodal Fusion:** Combines predictions from audio and image models based on the highest confidence score.
* **Interactive Web Interface:** A Streamlit application allows users to upload audio/image files and receive classification results.
* **Species Coverage:**
    * Audio model: 182 bird species.
    * Image model: 7 bird species.

## System Architecture

The system processes audio and image inputs through separate deep learning pipelines. The outputs (predicted species and confidence scores) from each pipeline are then fused to produce a final classification.

![System Architecture Diagram](https://github.com/user-attachments/assets/607a0337-60b3-466a-9e95-8f2299ec07d5)

## Technology Stack

* **Programming Language:** Python 3.10
* **Deep Learning:** PyTorch
* **Audio Processing:** Librosa
* **Image Processing:** Pillow, Torchvision
* **Numerical Operations:** NumPy
* **Data Handling:** Pandas
* **Model Zoo (EfficientNet):** `timm` (PyTorch Image Models)
* **Web Application:** Streamlit
* **Core Libraries:** See `requirements.txt` for a full list.

## Setup and Installation

### Prerequisites
* Python 3.10 or later
* `pip` (Python package installer)
* (Optional but Recommended) Conda or venv for virtual environment management

### Cloning the Repository
```bash
git clone https://github.com/venkat1924/Bird-Species-Detection
cd Bird-Species-Detection
```

### Setting up Virtual Environment
Using Conda:
```bash
conda create -n bird_classifier python=3.10
conda activate bird_classifier
```
Or using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Downloading Pre-trained Models
The pre-trained model weights are required for inference.
1.  **Audio Model:**
    * Download `effnet_seg20_80low.ckpt`.
    * Place it in the `models/audio/` directory (create this directory if it doesn't exist).
2.  **Image Model:**
    * Download `bird-resnet34best.pth`.
    * Place it in the `models/image/` directory (create this directory if it doesn't exist).

### Metadata Files
The system relies on metadata files for label mapping.
* The audio model uses label mappings derived from a `train_metadata.csv` (not directly used at inference time by the provided code snippets, but its class list is embedded).
* The image model has an internal `bird_name_map`. Ensure any such required mapping files are correctly referenced or included if used directly by the inference scripts. For this implementation, the mappings seem to be hardcoded or derived from the model structure itself during loading.

## Usage

To run the bird species classification system:

1.  Ensure you have activated your virtual environment and installed all dependencies.
2.  Navigate to the root directory of the project.
3.  Run the Streamlit application:
    ```bash
    streamlit run stream.py
    ```
4.  Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).
5.  Use the interface to:
    * Upload an audio file (e.g., .wav, .mp3).
    * Upload an image file (e.g., .jpg, .png).
    * Click the "Predict" button to get the classification results.

The application will display the individual predictions from the audio and image models, as well as the final fused prediction.

## Model Details

### Audio Model
* **Architecture:** EfficientNet-B0 (from `timm` library).
* **Input:** Mel spectrograms of shape ($1 \times 128 \times 960$) derived from 5-second audio chunks. Audio is resampled to $32 \text{ kHz}$, scaled, chunked, transformed into mel spectrograms ($128$ mel bands, $960$ time steps post-concatenation), and converted to decibels.
* **Output Classes:** 182 bird species.
* **Activation:** Sigmoid function on raw outputs to get confidence scores.
* **Note:** Currently, only the first 5-second chunk of an audio file effectively contributes to the final audio-based prediction.

### Image Model
* **Architecture:** Custom ResNet34-like CNN with residual connections.
* **Input:** RGB images resized to $128 \times 128$ pixels, with pixel values scaled to [0, 1].
* **Output Classes:** 7 bird species.
* **Activation:** Softmax function on raw logits.

### Multimodal Fusion
* **Strategy:** Late fusion based on confidence scores.
* **Logic:** The final predicted species is from the modality (audio or image) that yields the higher confidence score. The final reported confidence is this maximum score.

## File Structure
A brief overview of the key files and directories:
```
your-repo-name/
├── stream.py               # Main Streamlit application script
├── new_audio_main.py       # Audio processing and inference logic
├── new_image_main.py       # Image processing and inference logic
├── models/                 # Directory for pre-trained model weights
│   ├── audio/
│   │   └── effnet_seg20_80low.ckpt
│   └── image/
│       └── bird-resnet34best.pth
├── assets/                 # For static assets like diagrams
│   └── overallarch.png
├── requirements.txt        # Python package dependencies
├── README.md               # This file
└── references.bib          # Bibliography for the paper (if included)
└── ...                     # Other utility scripts or notebooks
```

## Limitations

As detailed in the paper, the current system has the following limitations:
1.  **Audio Chunk Processing:** The prediction for an entire audio file relies primarily on its first 5-second segment.
2.  **Simple Fusion Strategy:** Max-score fusion may not always be optimal.
   
## Future Work
Potential areas for future development include:
* Enhanced audio processing to consider entire recordings.
* Unified multimodal dataset and end-to-end model training.
* Sophisticated fusion mechanisms (e.g., attention-based).
* Expanded species coverage for the image model.
* System robustness improvements and optimization for deployment.


## Authors
* Anumaneni Venkat Balachandra (anumanenivb.cd22@rvce.edu.in)
* Prakhar Jain (prakharjain.cd22@rvce.edu.in)
* Mula Sohan (mulasohan.cd22@rvce.edu.in)
* Mohana (mohana@rvce.edu.in)

Department of CSE, R.V. College of Engineering, Bengaluru, India.
