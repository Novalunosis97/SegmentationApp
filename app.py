import streamlit as st
import numpy as np
import nibabel as nib
import pydicom
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import tempfile
import os
import traceback
from skimage.transform import resize

# Reuse the existing metrics and loss functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.reduce_sum(y_true, [1,2,3]) + tf.reduce_sum(y_pred, [1,2,3]) - intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def pixel_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32))
    total_pixels = tf.cast(tf.reduce_prod(tf.shape(y_true)), tf.float32)
    return correct_pixels / total_pixels

def hausdorff_distance(y_true, y_pred):
    y_true = tf.cast(tf.greater(y_true, 0.5), tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    
    def compute_hd(yt, yp):
        yt_points = tf.cast(tf.where(yt > 0), tf.float32)
        yp_points = tf.cast(tf.where(yp > 0), tf.float32)
        
        def _hausdorff(set1, set2):
            dists = tf.reduce_sum(tf.square(tf.expand_dims(set1, axis=1) - tf.expand_dims(set2, axis=0)), axis=-1)
            return tf.reduce_max(tf.reduce_min(dists, axis=1))
        
        h1 = _hausdorff(yt_points, yp_points)
        h2 = _hausdorff(yp_points, yt_points)
        
        return tf.maximum(h1, h2)
    
    return tf.reduce_mean(tf.map_fn(lambda x: compute_hd(x[0], x[1]), (y_true, y_pred), fn_output_signature=tf.float32))

# Reuse the helper functions with minor modifications
TARGET_SHAPE = (256, 256)

def normalize_slice(slice):
    min_val = np.min(slice)
    max_val = np.max(slice)
    if max_val - min_val != 0:
        return (slice - min_val) / (max_val - min_val)
    else:
        return slice - min_val

def preprocess_slice(img_slice):
    img_slice = img_slice.astype(np.float32)
    resized_slice = resize(img_slice, TARGET_SHAPE, mode='constant', preserve_range=True)
    normalized_slice = normalize_slice(resized_slice)
    return normalized_slice

def load_medical_image(uploaded_file):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            if uploaded_file.name.endswith(('.nii', '.nii.gz')):
                try:
                    nii_img = nib.load(temp_file_path)
                    data = nii_img.get_fdata().astype(np.float32)
                    if data.shape[2] < data.shape[0]:
                        data = np.transpose(data, (2, 1, 0))
                    return data
                except Exception as e:
                    st.error(f"Error loading NIfTI file: {str(e)}")
                    return None
            else:  # DICOM
                try:
                    dcm = pydicom.dcmread(temp_file_path)
                    return dcm.pixel_array
                except Exception as e:
                    st.error(f"Error loading DICOM file: {str(e)}")
                    return None
    except Exception as e:
        st.error(f"Error loading medical image: {str(e)}")
        return None

def prepare_model_input(volume_data):
    preprocessed_slices = []
    for i in range(volume_data.shape[0]):
        processed_slice = preprocess_slice(volume_data[i])
        processed_slice = processed_slice[..., np.newaxis].astype(np.float32)
        preprocessed_slices.append(processed_slice)
    return np.array(preprocessed_slices)

def overlay_segmentation_liver(original_slice, liver_mask, tumor_mask, liver_alpha=0.4, tumor_alpha=0.4):
    """Modified overlay function for liver and tumor segmentation"""
    if liver_mask.shape != original_slice.shape:
        liver_mask = resize(liver_mask, original_slice.shape, mode='constant', preserve_range=True)
    
    if tumor_mask.shape != original_slice.shape:
        tumor_mask = resize(tumor_mask, original_slice.shape, mode='constant', preserve_range=True)
    
    displayed_slice = normalize_slice(original_slice)
    displayed_slice = np.stack([displayed_slice]*3, axis=-1)
    
    overlay = displayed_slice.copy().astype(np.float32)
    
    liver_binary = liver_mask > 0.5
    tumor_binary = tumor_mask > 0.5
    
    # Brown for liver
    overlay[liver_binary] = (1 - liver_alpha) * overlay[liver_binary] + \
                           liver_alpha * np.array([0.6, 0.4, 0.2])
    
    # Red for tumors
    overlay[tumor_binary] = (1 - tumor_alpha) * overlay[tumor_binary] + \
                           tumor_alpha * np.array([1, 0, 0])
    
    return overlay

@st.cache_resource
def load_pancreas_model():
    try:
        custom_objects = {
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_coef': iou_coef,
            'pixel_accuracy': pixel_accuracy,
            'hausdorff_distance': hausdorff_distance
        }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'final_model.keras')
        
        try:
            model = load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            st.error(f"Could not load pancreas model from {model_path}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        st.error(f"Error loading pancreas model: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

@st.cache_resource
def load_liver_model():
    try:
        custom_objects = {
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_coef': iou_coef,
            'pixel_accuracy': pixel_accuracy,
            'hausdorff_distance': hausdorff_distance
        }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'liver_model.keras')
        
        try:
            model = load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            st.error(f"Could not load liver model from {model_path}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        st.error(f"Error loading liver model: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def pancreas_page():
    st.title("Pancreas Segmentation")
    
    uploaded_file = st.file_uploader("Upload DICOM or NIfTI file", 
                                   type=['dcm', 'nii', 'nii.gz'],
                                   key="pancreas_uploader")
    
    model = load_pancreas_model()
    if model is None:
        st.error("Failed to load the pancreas segmentation model.")
        return
    
    if uploaded_file is not None:
        volume_data = load_medical_image(uploaded_file)
        
        if volume_data is None:
            st.error("Failed to load the image.")
            return
        
        st.write("Performing pancreas segmentation...")
        with st.spinner("Processing..."):
            preprocessed_volume = prepare_model_input(volume_data)
            predictions = model.predict(preprocessed_volume, verbose=0)
        
        st.write("### Segmentation Results")
        
        slice_idx = st.slider("Select Slice", 
                            min_value=0, 
                            max_value=volume_data.shape[0]-1, 
                            value=volume_data.shape[0]//2,
                            key="pancreas_slider")
        
        current_slice = volume_data[slice_idx]
        current_pred = predictions[slice_idx]
        
        vessel_mask = current_pred[..., 0]
        tumor_mask = current_pred[..., 1]
        
        overlay_img = overlay_segmentation(current_slice, vessel_mask, tumor_mask)
        
        st.image(overlay_img, 
                caption=f"Slice {slice_idx + 1}/{volume_data.shape[0]}: Segmentation Overlay", 
                use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Vessel Segmentation")
            st.image(vessel_mask, caption="Vessel Mask", use_container_width=True)
        with col2:
            st.write("Tumor Segmentation")
            st.image(tumor_mask, caption="Tumor Mask", use_container_width=True)

def liver_page():
    st.title("Liver Segmentation")
    
    uploaded_file = st.file_uploader("Upload DICOM or NIfTI file", 
                                   type=['dcm', 'nii', 'nii.gz'],
                                   key="liver_uploader")
    
    model = load_liver_model()
    if model is None:
        st.error("Failed to load the liver segmentation model.")
        return
    
    if uploaded_file is not None:
        volume_data = load_medical_image(uploaded_file)
        
        if volume_data is None:
            st.error("Failed to load the image.")
            return
        
        st.write("Performing liver segmentation...")
        with st.spinner("Processing..."):
            preprocessed_volume = prepare_model_input(volume_data)
            predictions = model.predict(preprocessed_volume, verbose=0)
        
        st.write("### Segmentation Results")
        
        slice_idx = st.slider("Select Slice", 
                            min_value=0, 
                            max_value=volume_data.shape[0]-1, 
                            value=volume_data.shape[0]//2,
                            key="liver_slider")
        
        current_slice = volume_data[slice_idx]
        current_pred = predictions[slice_idx]
        
        liver_mask = current_pred[..., 0]
        tumor_mask = current_pred[..., 1]
        
        overlay_img = overlay_segmentation_liver(current_slice, liver_mask, tumor_mask)
        
        st.image(overlay_img, 
                caption=f"Slice {slice_idx + 1}/{volume_data.shape[0]}: Segmentation Overlay", 
                use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Liver Segmentation")
            st.image(liver_mask, caption="Liver Mask", use_container_width=True)
        with col2:
            st.write("Tumor Segmentation")
            st.image(tumor_mask, caption="Tumor Mask", use_container_width=True)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Pancreas Segmentation", "Liver Segmentation"])
    
    if page == "Pancreas Segmentation":
        pancreas_page()
    else:
        liver_page()

if __name__ == "__main__":
    main()