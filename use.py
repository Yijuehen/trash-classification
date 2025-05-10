import torch
from torchvision import transforms, models
from PIL import Image


# Define the function to classify an image
def classify_image(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features, len(class_names)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        predicted_class = class_names[predicted.item()]

    return predicted_class


# Example usage
if __name__ == "__main__":
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    model_path = "mobilenetv3_trashnet.pth"
    image_path = "D:/download/Computer_Vision/垃圾分类/archive/dataset-resized/cardboard/cardboard6.jpg"  # Replace with the path to your image

    result = classify_image(image_path, model_path, class_names)
    print(f"The image is classified as: {result}")
