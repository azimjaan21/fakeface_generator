"""
Classify Images as Real or Fake using Pretrained StyleGAN3-T Discriminator
------------------------------------------------------------------------
Loads the pretrained StyleGAN3 discriminator and classifies each image as:
    ✅ Real  (score > 0)
    ❌ Fake  (score <= 0)

Input:
    Folder containing test images (real or fake)
Output:
    Prints classification results for each image
"""

import sys, os, torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\models\stylegan3-t-ffhq-1024x1024.pkl"
TEST_FOLDER = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\test"

# === STYLEGAN3 IMPORT PATH ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
stylegan3_path = os.path.join(project_root, 'stylegan3')
if stylegan3_path not in sys.path:
    sys.path.insert(0, stylegan3_path)

import dnnlib, legacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD DISCRIMINATOR ===
print("🔹 Loading StyleGAN3-T discriminator...")
with open(MODEL_PATH, "rb") as f:
    data = legacy.load_network_pkl(f)
    D = data["D"].eval().requires_grad_(False).to(device)
print("✅ Discriminator loaded successfully!\n")

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

# === CLASSIFY IMAGES ===
image_paths = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not image_paths:
    print(f"⚠️ No images found in {TEST_FOLDER}")
    sys.exit()

print(f"🔍 Evaluating {len(image_paths)} images from {TEST_FOLDER}...\n")

for path in tqdm(image_paths, ncols=80):
    try:
        score = get_score(path)
        label = "Real ✅" if score > 0 else "Fake ❌"
        print(f"{os.path.basename(path)} → {label}  (score={score:.3f})")
    except Exception as e:
        print(f"⚠️ Skipped {path}: {e}")

print("\n✅ Classification complete!")
