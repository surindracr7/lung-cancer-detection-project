import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import numpy as np
import cv2

st.title("🫁 Lung Cancer Detection System")
st.write("Upload a CT scan image to detect lung cancer.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# CT SCAN VALIDATION FUNCTION
# -------------------------------
def is_ct_scan(pil_image, color_threshold=15):
    """
    Checks if the uploaded image looks like a CT scan
    based on color variance (CT scans are grayscale).
    """

    img = np.array(pil_image)

    if len(img.shape) < 3:
        return True  # Already grayscale

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    avg_diff = (diff_rg + diff_rb + diff_gb) / 3

    if avg_diff < color_threshold:
        return True
    else:
        return False


# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("model/lung_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------
# IMAGE TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

classes = ["Cancer Detected", "Normal"]

uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg","png","jpeg"])

# -------------------------------
# PREDICTION PIPELINE
# -------------------------------
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ STEP 1 — Validate CT scan
    if not is_ct_scan(img):
        st.error("❌ Invalid Input: Please upload a Lung CT Scan image only.")
    
    else:
        # ✅ STEP 2 — Predict
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)

        st.subheader(f"Prediction: {classes[pred.item()]}")



