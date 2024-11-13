import streamlit as st
import matplotlib.pyplot as plt

def plot_images_in_streamlit(image, segm_map, y_score_image, y_pred_image, class_label, best_threshold, image_path):
    # Create a matplotlib figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the original image
    axs[0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title(f"Original Image ")
    axs[0].axis('off')  # Hide axes for a cleaner look

    # Plot the heatmap with the segmentation map
    heat_map = segm_map
    im = axs[1].imshow(heat_map, cmap='jet', vmin=best_threshold, vmax=best_threshold * 2)
    axs[1].set_title(f'Anomaly score: {y_score_image / best_threshold:.4f} || {class_label[y_pred_image]}')
    axs[1].axis('off')
    
    # Add a color bar for reference on the heatmap
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # Display the plot in Streamlit
    st.pyplot(fig)

