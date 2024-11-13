import torch
import streamlit as st
from PIL import Image
import numpy as np
import json
from feature_extractor import resnet_feature_extractor
from data_transformation import img_transformer
import matplotlib.pyplot as plt
from plot_heamaps import plot_images_in_streamlit

# st.set_page_config("Welcome to Industrial Defective classifier")
st.title("Welcome to the application")
class_name = st.selectbox(label='Enter the class',options=['bottle','capsule','hazelnut','metal_nut','toothbrush'])
image_path = st.file_uploader(label="Upload an image: ",type = ['png','jpg'])
transform = img_transformer()
if st.button("Submit"):
    ##Load the model
    if class_name == 'bottle':
        model = torch.load("models/bottle_mb.pt")
    if class_name == 'capsule':
        model = torch.load("models/capsule_mb.pt")
    if class_name == 'hazelnut':
        model = torch.load("models/hazelnut_mb.pt")
    if class_name == 'metal_nut':
        model = torch.load("models/metal_nut_mb.pt")
    if class_name == 'toothbrush':
        model = torch.load("models/toothbrush_mb.pt")
    with open("best_thresholds.json",'r') as f:
        best_thresholds = json.load(f)
    best_threshold = best_thresholds[class_name]
    image = transform(Image.open(image_path)).unsqueeze(0)
    backbone = resnet_feature_extractor()
    with torch.no_grad():
        features = backbone(image)
    distances = torch.cdist(features,model,p = 2.0)
    dist_score,dist_score_idxs = torch.min(distances,dim = 1)
    s_star = torch.max(dist_score)
    segm_map = dist_score.view(1,1,28,28)
    segm_map = torch.nn.functional.interpolate(     # Upscale by bi-linaer interpolation to match the original input resolution
                segm_map,
                size=(224, 224),
                mode='bilinear'
            ).cpu().squeeze().numpy()

    y_score_image = s_star.cpu().numpy()

    y_pred_image = 1*(y_score_image >= best_threshold)
    # best_threshold,y_score_image
    class_label = ['Good','Defective']
    st.write(f"The object {class_name} is {class_label[y_pred_image]}")
    plot_images_in_streamlit(image=image,segm_map=segm_map,y_pred_image=y_pred_image,y_score_image=y_score_image,
                             best_threshold=best_threshold,class_label=class_label,image_path=image_path)





