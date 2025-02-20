from smth import predict_image, ResNet34, bird_name_map
import torch

# Initialize the model
num_classes = len(bird_name_map)  # Ensure this matches your trained model
model = ResNet34(in_channels=3, num_classes=num_classes)

# Load pretrained weights
model.load_state_dict(torch.load('path_to_model_weights.pth'))
model.eval()

# Call the predict_image function
image_path = 'path_to_test_image.jpg'
predicted_bird = predict_image(image_path, model, bird_name_map)
print(f"The predicted bird is: {predicted_bird}")
