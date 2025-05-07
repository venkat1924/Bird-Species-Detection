import audio.new_audio_main as audio
from new_image_main import predict_image

if __name__ == '__main__':
    predictor = audio.AudioPredictor()
    label, score = predictor.predict_single_audio('audio/data/AUDIO_FILES/gloibi1.ogg')
    print(label, score)

    bird_name = predict_image('image/test/glossyibis.jpeg')
    print(f"The predicted bird is: {bird_name}")
