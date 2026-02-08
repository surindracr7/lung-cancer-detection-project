from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model/lung_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

classes = ["Cancer", "Normal"]

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]

# Run prediction
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    result = predict_image(img_path)
    print(f"Prediction: {result}")
