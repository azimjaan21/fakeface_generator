import os
import torch
import dnnlib
import legacy
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === CONFIG ===
pkl_path = "models/stylegan3-t-ffhq-1024x1024.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("🔹 Loading StyleGAN3 discriminator...")
with open(pkl_path, 'rb') as f:
    G = D = None
    data = legacy.load_network_pkl(f)
    D = data['D'].to(device)
    print("✅ Discriminator loaded.")

# === Image preprocessor ===
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_score(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = D(img_t, None)  # returns logits (higher = more real)
    return float(score.cpu().item())
