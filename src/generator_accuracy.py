"""
Evaluate Generator Accuracy (G-Accuracy) using pretrained StyleGAN3-T Discriminator
------------------------------------------------------------------------------------
This measures how many generated images fool the pretrained discriminator.

Definitions:
  - Discriminator output (ŷ_b) after sigmoid ∈ [0, 1]
  - ŷ_b > 0.5 → predicted Fake
  - ŷ_b ≤ 0.5 → predicted Real
  - G-Accuracy = (# of generated images predicted Real) / total × 100

Outputs:
  • Mean ± Std of discriminator probabilities
  • G-Accuracy (%)
  • Histogram visualization of discriminator response
  • Saves top-10 most / least realistic images for qualitative inspection
"""

import os, torch, numpy as np
import sys
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
stylegan3_path = os.path.join(project_root, 'stylegan3')
if stylegan3_path not in sys.path:
    sys.path.insert(0, stylegan3_path)

import dnnlib, legacy
import shutil

# ============ CONFIG ============
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\models\stylegan3-t-ffhq-1024x1024.pkl"
GEN_IMAGES = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\fake\run01"
OUT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\results\g_accuracy"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ LOAD DISCRIMINATOR ============
print("🔹 Loading pretrained StyleGAN3-T discriminator...")
with open(MODEL_PATH, "rb") as f:
    data = legacy.load_network_pkl(f)
    D = data["D"].eval().requires_grad_(False).to(device)
print("✅ Discriminator loaded successfully.")

# ============ PREPROCESS ============
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def get_prob(path):
    """Return discriminator probability (ŷ_b) for a single image."""
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = D(img_t, None)
        prob = torch.sigmoid(logit)
    return float(prob.cpu().item())

# ============ EVALUATE ============
files = [os.path.join(GEN_IMAGES,f) for f in os.listdir(GEN_IMAGES)
         if f.lower().endswith((".png",".jpg",".jpeg"))]

scores = []
for f in files:
    try:
        scores.append(get_prob(f))
    except Exception as e:
        print(f"⚠️ Skipped {f}: {e}")

scores = np.array(scores)
preds = (scores > 0.5).astype(int)  # 1=fake, 0=real

# === G-Accuracy ===
G_acc = np.sum(preds == 0) / len(preds) * 100
mean_score, std_score = scores.mean(), scores.std()
print(f"\n🎯 G-Accuracy (Generator Fooling Rate): {G_acc:.2f}%")
print(f"Average Discriminator Score (ŷ_b): {mean_score:.3f} ± {std_score:.3f}")
print(f"Total evaluated images: {len(scores)}")

# ============ SAVE QUALITATIVE EXAMPLES ============
sorted_indices = np.argsort(scores)
top_real_idx = sorted_indices[:10]     # lowest probs → most realistic
top_fake_idx = sorted_indices[-10:]    # highest probs → most fake

most_real_dir = os.path.join(OUT_DIR, "Top10_MostRealistic")
most_fake_dir = os.path.join(OUT_DIR, "Top10_LeastRealistic")
os.makedirs(most_real_dir, exist_ok=True)
os.makedirs(most_fake_dir, exist_ok=True)

for i in top_real_idx:
    shutil.copy(files[i], os.path.join(most_real_dir, os.path.basename(files[i])))
for i in top_fake_idx:
    shutil.copy(files[i], os.path.join(most_fake_dir, os.path.basename(files[i])))

print(f"🖼️ Saved Top-10 most / least realistic samples → {OUT_DIR}")

# ============ VISUALIZATION ============
plt.figure(figsize=(8,5))
plt.hist(scores, bins=40, color="#69b3a2", edgecolor='black', alpha=0.7)
plt.axvline(0.5, color='red', linestyle='--', linewidth=1.5)
plt.title(f"Discriminator Response on Generated Images\n"
          f"G-Accuracy={G_acc:.1f}% | Mean={mean_score:.3f} ± {std_score:.3f}")
plt.xlabel("Discriminator Probability (ŷ_b)")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "g_accuracy_histogram.png"))
plt.show()

print("\n✅ G-Accuracy evaluation completed and visualization saved!")
