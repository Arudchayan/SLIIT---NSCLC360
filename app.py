import os
import numpy as np
import streamlit as st
import pydicom
import torch
import torch.nn.functional as F
import tempfile
import time
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from glob import glob
import sys
from torchvision.models.video import r2plus1d_18
import torch.nn as nn

# ✅ Add MedicalNet to system path
sys.path.append(r"E:\Researchhhh\MedicalNet")

# ✅ Import ResNet-50 for N & M Stage Predictions
from models.resnet import resnet50

# ✅ Define Class Labels
N_CLASS_LABELS = {0: "N0", 1: "N2", 2: "N1"}
M_CLASS_LABELS = {0: "M0", 1: "M1b", 2: "M1a"}
T_STAGE_LABELS = {0: "T3", 1: "T1b", 2: "T2a", 3: "T1a", 4: "Tis", 5: "T2b", 6: "T4"}
TUMOR_LOCATION_LABELS = {0: 'RUL', 1: 'RML', 2: 'LUL', 3: 'RLL', 4: 'LLL', 5: 'L Lingula'}

# ✅ Load PyTorch models
@st.cache_resource
def load_pytorch_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model = resnet50(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_r2plus1d_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = r2plus1d_18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.7),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ✅ Load models
N_model = load_pytorch_model("trained_modelN.pth", num_classes=len(N_CLASS_LABELS))
M_model = load_pytorch_model("trained_modelM.pth", num_classes=len(M_CLASS_LABELS))
T_stage_model = load_r2plus1d_model("model_TStage.pth", num_classes=len(T_STAGE_LABELS))
Tumor_model = load_r2plus1d_model("model_TumorrLocation.pth", num_classes=len(TUMOR_LOCATION_LABELS))

# ✅ Limit number of slices and files for faster processing
MAX_SLICES = 32
MAX_FILES = 50  

# ✅ Load and process DICOM images
def load_dicom_images(dicom_dir):
    dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))[:MAX_FILES]
    slices = [pydicom.dcmread(f) for f in dicom_files]

    for s in slices:
        if not hasattr(s, "SliceLocation"):
            s.SliceLocation = 0  

    slices.sort(key=lambda x: float(x.SliceLocation))
    return slices

# ✅ Convert DICOM images to a 3D volume
def convert_to_3d_volume(slices):
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = slices[0].SliceThickness
    spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
    
    image_shape = (len(slices), slices[0].pixel_array.shape[0], slices[0].pixel_array.shape[1])
    volume = np.zeros(image_shape, dtype=np.int16)
    
    for i, s in enumerate(slices):
        image_2d = s.pixel_array.astype(np.int16)
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        volume[i, :, :] = image_2d * slope + intercept
    
    return volume, spacing

# ✅ Predict class labels (T Stage, N Stage, M Stage)
def predict_class(model, input_tensor, class_labels):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return class_labels[predicted_class], confidence

# ✅ Predict tumor location
def predict_tumor_location(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return TUMOR_LOCATION_LABELS[predicted_class], confidence

# ✅ Function to visualize 3D Lung Scan using Matplotlib
def plot_3d(volume, threshold=-600):
    """Generate a static 3D visualization of the lung volume (corrected orientation)."""
    binary_image = volume > threshold  # Adjusted threshold for better lung structure
    verts, faces, _, _ = measure.marching_cubes(binary_image, level=0)  # Extract surface
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='gray', lw=0.5)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    st.pyplot(fig)  # Display in Streamlit

# ✅ Streamlit UI
st.title("🩺 Lung Cancer Staging and Tumor Location Prediction")
st.write("Upload a **folder of DICOM slices** for prediction.")

uploaded_folder = st.file_uploader("Upload a folder containing DICOM files", accept_multiple_files=True, type=["dcm"])

if uploaded_folder:
    with st.spinner("Processing DICOM files..."):
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_folder:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        slices = load_dicom_images(temp_dir)

        if slices:

            volume, spacing = convert_to_3d_volume(slices)

            input_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            input_tensor_r2plus1d = input_tensor.expand(-1, 3, -1, -1, -1)

            predicted_N_label, confidence_N = predict_class(N_model, input_tensor, N_CLASS_LABELS)
            predicted_M_label, confidence_M = predict_class(M_model, input_tensor, M_CLASS_LABELS)
            predicted_T_stage, confidence_T = predict_class(T_stage_model, input_tensor_r2plus1d, T_STAGE_LABELS)
            predicted_tumor_location, confidence_TL = predict_tumor_location(Tumor_model, input_tensor_r2plus1d)

            st.subheader("📊 Predicted Stages and Tumor Location")
            st.write(f"🩺 **Predicted T Stage:** {predicted_T_stage} (Confidence: {confidence_T:.2f})")
            st.write(f"🩺 **Predicted N Stage:** {predicted_N_label} (Confidence: {confidence_N:.2f})")
            st.write(f"🩺 **Predicted M Stage:** {predicted_M_label} (Confidence: {confidence_M:.2f})")
            st.write(f"🎯 **Predicted Tumor Location:** {predicted_tumor_location} (Confidence: {confidence_TL:.2f})")
            
            if st.button("Visualize 3D Lung Image 🫁"):
                plot_3d(volume)
