#!/usr/bin/env python3
"""
Test script for the PCA visualization function
"""
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from model import DiT_B_2
from train import visualize_representations_pca
from train import Dataset
def test_pca_visualization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a small model for testing
    model = DiT_B_2(
        input_size=32,
        in_channels=4,
        num_classes=1000,
        enable_routing=False
    ).to(device)
    model.eval()
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    
    dataset = Dataset(subset="train")
    # Create synthetic test images (batch of 4 images, 256x256, RGB)
    test_images = [torch.from_numpy(dataset[i]) for i in range(4)]
    test_images = torch.stack(test_images).to(device)
    
    # Normalize to [-1, 1] range as expected by the model
    test_images = test_images.clamp(-1, 1)
    
    print("Testing PCA visualization...")
    with torch.no_grad():
        pca_grid = visualize_representations_pca(
            model=model,
            images=test_images,
            vae=vae,
            feat_layer=8,  # Use a middle layer
            batch_size=4
        )
    
    # Save the result
    pca_grid.save("test_pca_visualization.png")
    print(f"Saved visualization to test_pca_visualization.png")
    print(f"Image size: {pca_grid.size}")
    
    # Expected width: 256 * 12 = 3072 (original + 11 timesteps)
    # Expected height: 256 * 4 = 1024 (4 images)
    print(f"Expected size: (3072, 1024)")

if __name__ == "__main__":
    test_pca_visualization()
