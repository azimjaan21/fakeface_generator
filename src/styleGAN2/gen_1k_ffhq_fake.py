"""
Generate 1000 fake face images using StyleGAN2-FFHQ (pretrained).
Saves images into: data/stylegan2_ffhq_fake/
"""

import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
OUT_DIR = "data/stylegan2_ffhq_fake"
N_IMAGES = 1000
SEED = 42  # for reproducibility

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Load Pretrained StyleGAN2-FFHQ
# -----------------------------
print("📥 Loading StyleGAN2 FFHQ pretrained model...")

# official NVIDIA pretrained model
# (1024x1024 FFHQ, StyleGAN2)
network_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"

with open('stylegan2-ffhq-config-f.pkl', 'wb') as f:
    f.write(torch.hub.load_state_dict_from_url(network_url, progress=True))

with open('stylegan2-ffhq-config-f.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()     # use the EMA generator

G.eval()

print("✅ Model loaded!")


# -----------------------------
# Generate Fake Images
# -----------------------------
print(f" Generating {N_IMAGES} images...")

for i in tqdm(range(N_IMAGES)):
    # sample random latent vector z
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).cuda()

    # generate 1024×1024 RGB image (float32, range [-1,1])
    img = G(z, None, truncation_psi=0.7, noise_mode='const')

    # convert to uint8 image
    img_np = (img[0].permute(1,2,0).cpu().numpy() * 127.5 + 127.5).clip(0,255).astype(np.uint8)

    # save
    Image.fromarray(img_np).save(f"{OUT_DIR}/fake_{i:04d}.png")

print(f" DONE! Saved {N_IMAGES} images to: {OUT_DIR}")
