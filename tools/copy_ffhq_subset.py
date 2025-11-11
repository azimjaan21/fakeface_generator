import os
import shutil

# === Paths ===
SRC_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\ffhq\ffhq\images"
DST_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\real"

# === Ensure output folder exists ===
os.makedirs(DST_DIR, exist_ok=True)

# === Collect all image files ===
image_files = [f for f in os.listdir(SRC_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# === Sort and select first 1000 ===
image_files.sort()
subset = image_files[:1000]

print(f"Found {len(image_files)} images, copying first {len(subset)}...")

# === Copy files ===
for i, filename in enumerate(subset, start=1):
    src_path = os.path.join(SRC_DIR, filename)
    dst_path = os.path.join(DST_DIR, filename)
    shutil.copy2(src_path, dst_path)
    if i % 100 == 0:
        print(f"✅ Copied {i} images...")

print("🎉 Done! 1,000 FFHQ real face images copied to:")
print(DST_DIR)
