import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn

st.title("🫁 Lung Cancer Detection System")
st.write("Upload a CT scan image to detect lung cancer.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
@st.cache_resource

def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("model/lung_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

classes = ["Cancer Detected", "Normal"]

uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    st.subheader(f"Prediction: {classes[pred.item()]}")

