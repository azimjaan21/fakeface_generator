import os
import shutil
import random
import csv

# === Paths ===
REAL_SRC = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\ffhq\ffhq\images"
FAKE_SRC = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\fake\run01"
DST_DIR  = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\mixed_half"

# === Create destination folder ===
os.makedirs(DST_DIR, exist_ok=True)

# === Collect all available images ===
real_images = [f for f in os.listdir(REAL_SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
fake_images = [f for f in os.listdir(FAKE_SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(real_images)} real and {len(fake_images)} fake images.")

# === Randomly select 500 each ===
random.shuffle(real_images)
random.shuffle(fake_images)
real_subset = real_images[:500]
fake_subset = fake_images[:500]

# === Prepare CSV log ===
csv_path = os.path.join(DST_DIR, "labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    # === Copy real images ===
    for i, fname in enumerate(real_subset, start=1):
        src = os.path.join(REAL_SRC, fname)
        dst_name = f"real_{i:04d}.png"
        dst = os.path.join(DST_DIR, dst_name)
        shutil.copy2(src, dst)
        writer.writerow([dst_name, "real"])
        if i % 100 == 0:
            print(f"✅ Copied {i} real images")

    # === Copy fake images ===
    for i, fname in enumerate(fake_subset, start=1):
        src = os.path.join(FAKE_SRC, fname)
        dst_name = f"fake_{i:04d}.png"
        dst = os.path.join(DST_DIR, dst_name)
        shutil.copy2(src, dst)
        writer.writerow([dst_name, "fake"])
        if i % 100 == 0:
            print(f"✅ Copied {i} fake images")

print(" Done! Created mixed dataset (500 real + 500 fake):")
print(DST_DIR)
print(f" Labels saved to: {csv_path}")
