"""
Evaluate StyleGAN3 Discriminator Across 4 Experiments
-----------------------------------------------------
Experiments:
  1️⃣ Fake (Self StyleGAN3)
  2️⃣ Real (FFHQ)
  3️⃣ Mixed (StyleGAN3 + FFHQ)
  4️⃣ Unseen Generators (Diffusion, StableDiffusion, Denoising + FFHQ)

Outputs (per experiment):
  - mean ± std score
  - *_hist.png
  - *_classification_bar.png
Plus overall:
  - discriminator_scores.png (combined)
  - summary_bar.png (mean comparison, color-coded)
  - discriminator_summary.txt
"""

import sys, os, torch, numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# === Fix import path for StyleGAN3 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
stylegan3_path = os.path.join(project_root, 'stylegan3')
if stylegan3_path not in sys.path:
    sys.path.insert(0, stylegan3_path)

import dnnlib, legacy

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\models\stylegan3-t-ffhq-1024x1024.pkl"

DATA_PATHS = {
    "1-Fake(Self-StyleGAN3)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\fake\run01",
    "2-Real(FFHQ)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\real",
    "3-Mixed(StyleGAN3+FFHQ)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\mixed_half",
    "4-Unseen(DiffGens+FFHQ)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\unseen_gen_real",
}

OUT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\results"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Discriminator ===
print("🔹 Loading StyleGAN3 discriminator...")
with open(MODEL_PATH, "rb") as f:
    data = legacy.load_network_pkl(f)
    D = data["D"].eval().requires_grad_(False).to(device)
print("✅ Discriminator loaded successfully.")

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_score(img_path):
    """Return raw discriminator score (logit)."""
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = D(img_t, None)
    return float(score.cpu().item())

def evaluate_folder(folder, label):
    """Compute scores for all images in a folder."""
    if not os.path.exists(folder):
        print(f"⚠️ Missing folder: {folder}")
        return []
    paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".png"))]
    scores = []
    for p in tqdm(paths, desc=f"Evaluating {label}", ncols=80):
        try:
            scores.append(get_score(p))
        except Exception as e:
            print(f"⚠️ Skipped {p}: {e}")
    if len(scores) == 0:
        return []
    mean, std = np.mean(scores), np.std(scores)
    print(f"📊 {label}: mean={mean:.3f}, std={std:.3f}, n={len(scores)}")
    return scores

# === Evaluate all experiments ===
all_scores = {}
for label, path in DATA_PATHS.items():
    scores = evaluate_folder(path, label)
    if scores:
        all_scores[label] = scores

# === Per-experiment histograms and classification bars ===
summary_data = []
for label, scores in all_scores.items():
    mean, std = np.mean(scores), np.std(scores)
    summary_data.append((label, mean, std))

    # 1️⃣ Histogram per experiment
    plt.figure(figsize=(7,5))
    plt.hist(scores, bins=40, color="#69b3a2", alpha=0.7)
    plt.title(f"Discriminator Response — {label}\nMean={mean:.3f}, Std={std:.3f}")
    plt.xlabel("Discriminator Score (logit)")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{label}_hist.png"))
    plt.close()

    # 2️⃣ Classification summary bar
    num_real_like = np.sum(np.array(scores) > 0)
    num_fake_like = np.sum(np.array(scores) <= 0)
    plt.figure(figsize=(5,4))
    plt.bar(['Fake-like', 'Real-like'], [num_fake_like, num_real_like],
            color=['#ff7f7f', '#90ee90'])
    plt.title(f"Classification Summary — {label}")
    plt.ylabel("Image Count")
    for i, val in enumerate([num_fake_like, num_real_like]):
        plt.text(i, val + 5, f'{val}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{label}_classification_bar.png"))
    plt.close()

print("📊 Saved all individual experiment histograms and classification charts!")

# === Combined histogram for all experiments ===
plt.figure(figsize=(9,6))
for label, scores in all_scores.items():
    plt.hist(scores, bins=50, alpha=0.6, label=label)
plt.xlabel("Discriminator Score (logit)")
plt.ylabel("Image Count")
plt.title("Overall Discriminator Response — All Experiments")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "discriminator_scores.png"))
plt.close()
print("📈 Saved combined histogram → discriminator_scores.png")

# === Color-coded summary bar chart ===
labels = [x[0] for x in summary_data]
means = [x[1] for x in summary_data]
stds  = [x[2] for x in summary_data]

# Color mapping for your 4 experiments
colors = ['#FF4C4C',  # 🔴 Fake-dominant
          '#4CAF50',  # 🟢 Real
          '#9C27B0',  # 🟣 Mixed
          '#000000']  # ⚫ Unseen

plt.figure(figsize=(9,5))
bars = plt.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.9)
plt.xticks(rotation=25, ha='right')
plt.ylabel("Mean Discriminator Score")
plt.title("Summary of Mean Discriminator Scores (±Std)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "summary_bar.png"))
plt.close()
print("📊 Saved color-coded summary bar chart → summary_bar.png")

# === Save text summary ===
summary_txt = os.path.join(OUT_DIR, "discriminator_summary.txt")
with open(summary_txt, "w") as f:
    for label, mean, std in summary_data:
        f.write(f"{label}: mean={mean:.3f}, std={std:.3f}, n={len(all_scores[label])}\n")
print(f"📝 Summary saved → {summary_txt}")

print("\n✅ All 4 experiments evaluated and visualized successfully!")
