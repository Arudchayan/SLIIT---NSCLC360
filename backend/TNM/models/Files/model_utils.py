# model_utils.py
import torch
from TNM.models.resnet import resnet18  # your model architecture file
import joblib
from TNM.models.Files.dicom_processing import load_dicom_images, convert_to_3d_volume, normalize_hu, fast_resize

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_t = resnet18(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_classes=7)
    model_n = resnet18(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_classes=3)
    model_m = resnet18(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_classes=3)
    model_loc = resnet18(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_classes=6)
   
    model_loc.load_state_dict(torch.load("TNM/models/resnet18_3d_best_Location.pth", map_location=device))
    model_m.load_state_dict(torch.load("TNM/models/resnet18_3d_best_M.pth", map_location=device))
    model_n.load_state_dict(torch.load("TNM/models/resnet18_3d_best_N.pth", map_location=device))
    model_t.load_state_dict(torch.load("TNM/models/resnet18_3d_best_T.pth", map_location=device))


    return model_t.to(device), model_n.to(device), model_m.to(device), model_loc.to(device)

def preprocess_input_for_model(volume):
    # Apply normalization, resizing, add batch and channel dimensions, etc.
    # Example:
    volume_processed = normalize_hu(fast_resize(volume, (64, 128, 128)))
    tensor = torch.tensor(volume_processed).unsqueeze(0).unsqueeze(0).float()
    return tensor
