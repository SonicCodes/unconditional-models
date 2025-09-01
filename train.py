from model import DiT_XXL_2
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import json
from PIL import Image
import click
import random
import numpy as np
from diffusers import AutoencoderKL
import torch.nn.functional as F
import torchvision
import wandb
import tqdm
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.decomposition import PCA
#  enable tf32 and fast math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, subset="train"):
        self.dataset_path = "./dataset/" + subset

        # open index.json in dataset_path
        with open(self.dataset_path + "/index.json", "r") as f:
            self.dataset = json.load(f)
        
        # shuffle the dataset with seed 0
        random_state = random.Random(0)
        random_state.shuffle(self.dataset)
        
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        text_path = image_path.replace(".jpg", ".txt")

        try:

            image = Image.open(image_path)
            # text = open(text_path, "r").read().replace("\n", " ").replace("The image shows ", "")
            # convert to numpy array
            # convert image to numpy array
            image = np.array(image).transpose(2, 0, 1) / 255.0
            image = ((image * 2.0) - 1.0).astype(np.float32)
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.dataset)))
        return image


def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    return dist.get_rank(), dist.get_world_size(), (dist.get_rank() % 8)


def update_ema(ema, model, alpha=0.999):
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def lerp(x0, x1, t):
    return (x1 * t[:, None, None, None]) + (x0 * (1 - t[:, None, None, None]))
@torch.no_grad()
def inference(model, x1, guidance_scale=1.0):
    xt = x1.clone()
    n_steps = 100
    dt = 1.0 / n_steps
    self_cond_feats = torch.zeros((x1.shape[0], (256//16)**2, model.hidden_size), device=x1.device)+0.0
    for i in range(n_steps, 0, -1):
        t = torch.full((x1.shape[0],), i / n_steps, device=x1.device).to(x1.dtype)
        v_cfg, _ = model(x=xt, t=t, rin_features_in=self_cond_feats, feat_layer=32)
        v_null, _ = model(x=xt, t=t,  rin_features_in=self_cond_feats, feat_layer=32, force_routing=True) #force_routing=True,
        v = v_null + ((v_cfg - v_null) * guidance_scale)
        xt -= v * dt
    return xt


def initialize_dinov2_model(device):
    # initialize dinov2 model
    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/dinov2-giant')
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def visualize_representations_pca(model, images, vae, feat_layer=16, batch_size=8, noise_x1=None):
    """
    Visualize model representations at different noise timesteps using PCA.
    
    Args:
        model: The DiT model
        images: Input images in pixel space (B, C, H, W)
        vae: The VAE model for encoding
        feat_layer: Which layer to extract features from
        batch_size: Number of images to process
        
    Returns:
        grid: A PIL image containing the visualization grid
    """
    device = images.device
    
    # Limit batch size if needed
    if images.shape[0] > batch_size:
        images = images[:batch_size]
    
    # Encode images to latent space
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    x0 = latents
    
    # Define timesteps
    timesteps = torch.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    
    # Collect all representations
    all_representations = []
    spatial_dims = None
    
    # Get model hidden size
    hidden_size = model.module.hidden_size if hasattr(model, 'module') else model.hidden_size
    
    for t_val in timesteps:
        # Create timestep tensor
        t = torch.full((x0.shape[0],), t_val, device=device)
        
        # Create noise
        x1 = torch.randn_like(x0)
        
        # Interpolate to get noisy version
        xt = lerp(x0, x1, t)
        
        # Get features from model
        self_cond_feats = torch.zeros((x0.shape[0], (256//16)**2, hidden_size), device=device)
        _, features = model(x=xt, t=t, rin_features_in=self_cond_feats, feat_layer=feat_layer, should_project=False)
   
        
        # features shape: (B, num_patches, hidden_size)
        if spatial_dims is None:
            num_patches = features.shape[1]
            H_patches = W_patches = int(np.sqrt(num_patches))
            spatial_dims = (H_patches, W_patches)
        
        all_representations.append(features)
    
    # Stack all representations: (num_timesteps, B, num_patches, hidden_size)
    all_representations = torch.stack(all_representations)
    num_timesteps, B, num_patches, hidden_dim = all_representations.shape
    
    # Process each image separately to avoid contamination
    all_pca_features = []
    for b in range(B):
        image_pca_features = []
        
        # Process each timestep separately for this image
        for t_idx in range(num_timesteps):
            # Get features for this image at this timestep only
            # Shape: (num_patches, hidden_dim)
            features_for_timestep = all_representations[t_idx, b, :, :].cpu().numpy()
            
            # Fit PCA for this specific timestep only
            pca = PCA(n_components=3)
            features_pca = pca.fit_transform(features_for_timestep)
            
            # Normalize each principal component independently using percentile clipping
            for c in range(3):
                component = features_pca[:, c]
                # Use percentiles for robust normalization
                p5 = np.percentile(component, 5)
                p95 = np.percentile(component, 95)
                # Clip and normalize to [0, 1]
                component = np.clip(component, p5, p95)
                if p95 - p5 > 1e-8:
                    component = (component - p5) / (p95 - p5)
                else:
                    component = component * 0 + 0.5
                features_pca[:, c] = component
            
            image_pca_features.append(features_pca)
        
        # Stack timesteps for this image: (num_timesteps, num_patches, 3)
        image_pca_features = np.stack(image_pca_features)
        all_pca_features.append(image_pca_features)
    
    # Stack all images: (B, num_timesteps, num_patches, 3)
    all_pca_features = np.stack(all_pca_features)
    
    # Reshape to spatial format: (B, num_timesteps, H_patches, W_patches, 3)
    all_pca_features = all_pca_features.reshape(B, num_timesteps, spatial_dims[0], spatial_dims[1], 3)
    
    # Convert to torch tensors and resize to match original image size
    visualizations = []
    
    # First add the original images
    original_imgs = images.clamp(-1, 1) * 0.5 + 0.5  # Normalize to [0, 1]
    visualizations.append(original_imgs)
    
    # Then add PCA visualizations for each timestep
    for t_idx in range(num_timesteps):
        pca_imgs = torch.from_numpy(all_pca_features[:, t_idx]).to(device).float()
        # Reshape to (B, 3, H_patches, W_patches)
        pca_imgs = pca_imgs.permute(0, 3, 1, 2)
        # Resize to match original image size
        pca_imgs = F.interpolate(pca_imgs, size=images.shape[-2:], mode='nearest')
        visualizations.append(pca_imgs)
    visualizations.append(torch.randn_like(visualizations[-1]))
    
    # Stack all visualizations: (12, B, 3, H, W) - original + 11 timesteps
    all_vis = torch.stack(visualizations)
    
    # Reshape to create grid: (B, 12*3, H, W)
    grid_images = []
    for b in range(B):
        row_images = [all_vis[i, b] for i in range(all_vis.shape[0])]
        row = torch.cat(row_images, dim=2)  # Concatenate width-wise
        grid_images.append(row)
    
    # Stack rows vertically
    final_grid = torch.cat(grid_images, dim=1)  # (3, B*H, 12*W)
    
    # Convert to uint8
    final_grid = (final_grid * 255.0).clamp(0, 255).to(torch.uint8)
    
    # Convert to PIL image
    grid_pil = Image.fromarray(final_grid.cpu().permute(1, 2, 0).numpy())
    
    return grid_pil
    

@click.command()
@click.option("--batch-size", type=int, default=32)
@click.option("--epochs", type=int, default=10)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=1e-2)
@click.option("--num-workers", type=int, default=4)
def train(batch_size, epochs, lr, weight_decay, num_workers):
    rank, world_size, local_rank = init_distributed()
    device = f"cuda:{local_rank}"
    master_process = rank == 0

    model = DiT_XXL_2(
        enable_routing=True,
        routes=[
            {'selection_ratio': 0.5, 'start_layer_idx': 4, 'end_layer_idx': 30}
        ],
        proj_dim=1536
    ).to(device)
    ema = deepcopy(model)

    # load checkpoint from "~/jack/speedy-t2i/checkpoints/train_362918/model_130000.pt"
    model.load_state_dict(torch.load("./checkpoints/train_362918/model_130000.pt"))
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # model.module = torch.compile(model.module)
    update_ema(ema, model.module, 0.0)


    if master_process:
        train_id = random.randint(0, 1000000)
        wandb.init(
            project="speedyI",
            name=f"train_{train_id}",
            id=f"train_{train_id}",
        )

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = Dataset(subset="train")
    val_dataset = Dataset(subset="valid")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    model.train()

    valid_iter_loader = iter(val_loader)
    valid_samples = next(valid_iter_loader)
    valid_samples = valid_samples.to(device)[:8]
    valid_noise = torch.randn_like(valid_samples)[:, :, :valid_samples.shape[2] // 8, :valid_samples.shape[3] // 8]

    train_loader_iter = iter(train_loader)
    N_STEPS = 5_000_000

    dinov2 = initialize_dinov2_model(device)
    # make untrainable
    for param in dinov2.parameters():
        param.requires_grad = False

    ema.eval()

    def extract_features(images):
        # torch resize to 224x224
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode='area')
        features = dinov2(pixel_values=images).last_hidden_state[:, 1:, :]
  
        return features


    def fwd_bwd(x0, dinov2_features):
        batch_size = x0.shape[0]
        t = torch.rand(batch_size).to(device)
        x1 = torch.randn_like(x0)
        v_theta = x1 - x0  # Target flow for standard flow matching
        xt = lerp(x0, x1, t)
        
        # ========== Add Contrastive Flow Matching ==========
        # Lambda parameter (as used in the paper)
        lambda_cfm = 0.05
        
        # Create negative samples by randomly permuting the batch
        # This ensures each sample gets a different negative
        perm_indices = torch.randperm(batch_size, device=device)
        # Ensure no sample is paired with itself
        for i in range(batch_size):
            if perm_indices[i] == i:
                perm_indices[i] = (i + 1) % batch_size
        
        # Get negative flows (from different samples in the batch)
        x0_neg = x0[perm_indices]
        x1_neg = x1[perm_indices]
        v_theta_neg = x1_neg - x0_neg  # Negative flow trajectories
        # ====================================================
        
        self_cond_feats = torch.zeros((batch_size, (256//16)**2, model.module.hidden_size), device=device)+0.0
        
        v_pred, feats_dirty = model(x=xt, t=t, feat_layer=32, rin_features_in=self_cond_feats.contiguous().detach())
        
        # Standard flow matching loss
        denoising_loss = F.mse_loss(v_pred, (v_theta - (v_theta_neg * lambda_cfm)))
       
        # Feature alignment loss (REPA)
        feature_loss = 1-F.cosine_similarity(feats_dirty.reshape(feats_dirty.shape[0], -1), 
                                            dinov2_features.reshape(dinov2_features.shape[0], -1), dim=1).mean()
        
        # Combine all losses
        loss = denoising_loss + feature_loss
        loss.backward()
        
        return loss, {
            "denoising_loss": denoising_loss.item(),
            "feature_loss": feature_loss.item()
        }
    # fwd_bwd = torch.compile(fwd_bwd)
    val_x1 = torch.randn((32, 4, 32, 32), device=device)
    for i in tqdm.trange(N_STEPS, disable=rank != 0):
        model.train()
        try:
            images = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            images = next(train_loader_iter)

        images = images.to(device)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        x0 = latents.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            zs = extract_features(images)
        loss, loss_dict = fwd_bwd(x0, zs)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema, model.module, 0.999)
        
        if master_process and (i % 10 == 0):
            wandb.log({
                "loss": loss.item(),
                **loss_dict
            })

        if master_process and (i % 100 == 0):
            with torch.no_grad():
                pca_vis = visualize_representations_pca(model, valid_samples, vae, feat_layer=32, batch_size=valid_samples.shape[0], noise_x1=valid_noise)
                wandb.log({
                    "representations_pca": wandb.Image(pca_vis)
                })
            

        # print(loss.item())

        if master_process and i % 1000 == 0:
            model.eval()
            with torch.no_grad():
                def get_img(guidance_scale=1.0):
                    img = inference(ema, val_x1, guidance_scale=guidance_scale) / vae.config.scaling_factor
                    img = vae.decode(img).sample.float()
                    img = img.clamp(-1, 1) * 0.5 + 0.5
                    img = (img * 255.0).to(torch.uint8)
                    # torchvision make grid
                    grid = torchvision.utils.make_grid(img).cpu().permute(1, 2, 0).numpy()
                    # concatenate images horizontally
                    img = Image.fromarray(grid)
                    return img
                
                # Create PCA visualization of representations
                
                wandb.log({
                    "image": wandb.Image(get_img(1.0)),
                    "image_1.5": wandb.Image(get_img(1.5)),
                    "image_2.0": wandb.Image(get_img(2.0)),
                    "image_3.0": wandb.Image(get_img(3.0))
                })
        if master_process and i % 5000 == 0:
            # make checkpoints, then checkpoints/train_id/
            os.makedirs(f"checkpoints/train_{train_id}", exist_ok=True)
            torch.save(model.module.state_dict(), f"checkpoints/train_{train_id}/model_{i}.pt")

    
if __name__ == "__main__":
    train()