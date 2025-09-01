# PCA Visualization of Model Representations

## Overview

The `visualize_representations_pca` function creates a visualization of how the model's internal representations change as noise is progressively added to images. It uses PCA (Principal Component Analysis) to reduce the high-dimensional feature representations to 3 dimensions (RGB) for visualization.

## Function Details

```python
visualize_representations_pca(model, images, vae, feat_layer=16, batch_size=8)
```

### Parameters:
- `model`: The DiT model to extract features from
- `images`: Input images in pixel space (B, C, H, W)
- `vae`: The VAE model for encoding images to latent space
- `feat_layer`: Which transformer layer to extract features from (default: 16)
- `batch_size`: Maximum number of images to process (default: 8)

### Process:
1. Encodes input images to latent space using VAE
2. For each timestep from 0.0 to 1.0 (in 0.1 increments):
   - Creates increasingly noisy versions of the latents
   - Feeds them through the model to get feature representations
3. **For each image separately** (to avoid feature contamination):
   - Fits PCA to reduce feature dimensions from hidden_size to 3 (RGB)
   - Normalizes each principal component independently using percentile clipping (5th to 95th percentile)
4. Creates a visualization grid where each row shows:
   - Original image
   - PCA representations at timesteps 0.0, 0.1, 0.2, ..., 1.0

### Output:
A PIL Image where:
- Each row corresponds to one input image
- Each column shows:
  - Column 0: Original image
  - Columns 1-11: PCA visualizations at increasing noise levels

## Usage in Training

The visualization is automatically generated every 1000 training steps and logged to wandb as "representations_pca".

## Interpreting the Visualization

- **Timestep 0.0**: Shows representations of clean images (no noise)
- **Timestep 1.0**: Shows representations of pure noise
- **Intermediate timesteps**: Show how representations transition as noise increases

The PCA visualization helps understand:
- What features the model focuses on at different noise levels
- How the model's internal representations change during the diffusion process
- Whether the model learns meaningful structure in its intermediate layers

## Key Improvements

1. **Per-image PCA fitting**: Each image gets its own PCA fit to prevent feature contamination between images
2. **Independent component normalization**: Each principal component is normalized independently using percentile clipping (5th-95th percentile) for better contrast and robustness to outliers

## Testing

Run the test script to verify the visualization works:
```bash
python test_pca_vis.py
```

This will create a test visualization saved as `test_pca_visualization.png`.
