import streamlit as st
import audio.new_audio_main as audio
from new_image_main import predict_image

# Initialize the audio predictor
predictor = audio.AudioPredictor()

# Streamlit app setup
st.title("Bird Species Prediction App")
st.write("Select an audio file and an image file to proceed.")

# File upload widgets
uploaded_audio = st.file_uploader("Upload an audio file:", type=["ogg", "mp3", "wav"])
uploaded_image = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])

# Button to trigger prediction
if st.button("Predict"):
    if uploaded_audio:
        # Save the uploaded audio file temporarily
        audio_file_path = f"temp_{uploaded_audio.name}"
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())

        # Perform prediction
        label, score = predictor.predict_single_audio(audio_file_path)

        # Display results
        st.write(f"# Prediction Result")
        st.write(f"## Label: **{label}**")
        #st.write(f"## Score: **{score:.2f}**")

        # Optionally, clean up the temporary file
        import os
        os.remove(audio_file_path)
    else:
        st.warning("Please upload an audio file before predicting.")
