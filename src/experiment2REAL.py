"""
Evaluate StyleGAN3 Discriminator on Real-Only Dataset
-----------------------------------------------------
Dataset: Only Real images (label=0)
Goal: Identify real images that are misclassified as Fake (FalseFake)

Outputs:
  - Mean ± std of discriminator logits
  - Histogram
  - Classification bar
  - Folder "output_real/FalseFake" with misclassified real images
  - Summary text with stats
"""

import os, torch, numpy as np
from tqdm import tqdm
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


# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\models\stylegan3-t-ffhq-1024x1024.pkl"

REAL_DATA = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\real"
OUT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\results_2REAL"
OUTPUT_REAL = os.path.join(OUT_DIR, "output_real")
FALSE_FAKE_DIR = os.path.join(OUTPUT_REAL, "FalseFake")
os.makedirs(FALSE_FAKE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD DISCRIMINATOR ===
print("🔹 Loading StyleGAN3-T discriminator...")
with open(MODEL_PATH, "rb") as f:
    data = legacy.load_network_pkl(f)
    D = data["D"].eval().requires_grad_(False).to(device)
print("✅ Discriminator loaded successfully!")

# === PREPROCESS ===
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def get_score(img_path):
    """Return discriminator logit."""
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = D(img_t, None)
    return float(score.cpu().item())

# === EVALUATE REAL DATASET ===
print(f"\n🧠 Evaluating Real Dataset: {REAL_DATA}")
image_paths = [os.path.join(REAL_DATA, f) for f in os.listdir(REAL_DATA)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

scores = []
false_fake_paths = []

for img_path in tqdm(image_paths, ncols=80):
    try:
        s = get_score(img_path)
        scores.append(s)
        # Real label = 0, prediction: score > 0 → Real, ≤0 → Fake
        if s <= 0:  # Misclassified as Fake
            false_fake_paths.append(img_path)
    except Exception as e:
        print(f"⚠️ Skipped {img_path}: {e}")

scores = np.array(scores)
mean, std = scores.mean(), scores.std()
num_total = len(scores)
num_false_fake = len(false_fake_paths)
num_true_real = num_total - num_false_fake
false_rate = num_false_fake / num_total * 100

print(f"\n📊 Results:")
print(f"  Total images: {num_total}")
print(f"  TrueReal: {num_true_real}")
print(f"  FalseFake (misclassified): {num_false_fake} ({false_rate:.2f}%)")
print(f"  Mean score: {mean:.3f} ± {std:.3f}")

# === SAVE MISCLASSIFIED IMAGES ===
print(f"\n💾 Saving {num_false_fake} FalseFake images → {FALSE_FAKE_DIR}")
for p in false_fake_paths:
    shutil.copy(p, os.path.join(FALSE_FAKE_DIR, os.path.basename(p)))
print("✅ All misclassified real images saved!")

# === HISTOGRAM VISUALIZATION ===
plt.figure(figsize=(8,5))
plt.hist(scores, bins=40, color="#69b3a2", edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.title(f"Discriminator Scores on Real Dataset\nFalseFake={false_rate:.1f}%, Mean={mean:.3f}")
plt.xlabel("Discriminator Score (logit)")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_dataset_histogram.png"))
plt.close()
print("📈 Saved histogram → results/real_dataset_histogram.png")

# === CLASSIFICATION BAR ===
plt.figure(figsize=(5,4))
plt.bar(["TrueReal","FalseFake"], [num_true_real, num_false_fake],
        color=["#90ee90","#ff7f7f"], edgecolor='black')
plt.title("Discriminator Classification on Real Dataset")
for i,v in enumerate([num_true_real, num_false_fake]):
    plt.text(i, v + max(1, v*0.02), f"{v}", ha='center', fontsize=10, weight='bold')
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_dataset_classification_bar.png"))
plt.close()
print("📊 Saved classification bar → results/real_dataset_classification_bar.png")

# === SUMMARY TEXT ===
summary_path = os.path.join(OUT_DIR, "real_dataset_summary.txt")
with open(summary_path, "w") as f:
    f.write("StyleGAN3 Discriminator Evaluation — Real Dataset Only\n")
    f.write(f"Total images: {num_total}\n")
    f.write(f"TrueReal: {num_true_real}\n")
    f.write(f"FalseFake: {num_false_fake} ({false_rate:.2f}%)\n")
    f.write(f"Mean score: {mean:.3f} ± {std:.3f}\n")
print(f"📝 Summary saved → {summary_path}")

print("\n✅ Real-only evaluation complete!")
