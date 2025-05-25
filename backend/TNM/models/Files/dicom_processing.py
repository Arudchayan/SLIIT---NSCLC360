# dicom_processing.py
import os
import numpy as np
import pydicom
from skimage.transform import resize
from glob import glob
import tensorflow as tf

def load_dicom_images(dicom_dir):
    """Load and sort DICOM slices by SliceLocation."""
    dicom_files = glob(os.path.join(dicom_dir, "**/*.dcm"), recursive=True)  # recursive in case subfolders
    slices = [pydicom.dcmread(f) for f in dicom_files]
    
    for s in slices:
        if not hasattr(s, "SliceLocation"):
            s.SliceLocation = 0
    slices.sort(key=lambda x: float(x.SliceLocation))
    return slices

def convert_to_3d_volume(slices):
    """Convert list of DICOM slices to 3D NumPy array with consistent size."""
    desired_shape = (512, 512)  # or your dataset-specific shape
    volume = np.zeros((len(slices), desired_shape[0], desired_shape[1]), dtype=np.int16)

    for i, s in enumerate(slices):
        img = s.pixel_array.astype(np.int16)
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else 0
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1
        img = img * slope + intercept
        img_resized = resize(img, desired_shape, order=1, mode='constant', cval=0, anti_aliasing=True)
        volume[i] = img_resized
    
    return volume, None  # You can add spacing if needed

def normalize_hu(volume, hu_min=-100, hu_max=400):
    """Normalize Hounsfield Units to range [-1, 1]."""
    volume = np.clip(volume, hu_min, hu_max)
    volume = 2 * (volume - hu_min) / (hu_max - hu_min) - 1
    return volume

def fast_resize(volume, target_shape):
    """Resize 3D volume to target shape (D, H, W) using TensorFlow."""
    volume = tf.convert_to_tensor(volume, dtype=tf.float32)
    # Resize H, W for each slice (axis=1 and 2)
    resized_slices = []
    for i in range(volume.shape[0]):
        slice_i = volume[i, :, :]
        resized_slice = tf.image.resize(slice_i[..., tf.newaxis], (target_shape[1], target_shape[2]))
        resized_slices.append(resized_slice[..., 0])
    resized_volume = tf.stack(resized_slices)
    # Resize depth (D axis)
    resized_volume = tf.image.resize(tf.transpose(resized_volume, perm=[1,2,0]), (target_shape[1], target_shape[2]))
    resized_volume = tf.transpose(resized_volume, perm=[2,0,1]).numpy()
    return resized_volume
