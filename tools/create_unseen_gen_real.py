import os
import shutil
import random
import csv

# === SOURCE FOLDERS ===
DIFFUSION_GAN_SRC = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\diffusion_gan\diff\ffhq-data\Diffusion-StyleGAN2-DiffAug-FFHQ"
STABLE_DIFF_MALE  = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\stable_diffusion\stable-face\Male"
STABLE_DIFF_FEMALE= r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\stable_diffusion\stable-face\Female"
DENOISING_DIFF_GAN= r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\denoising_diffusion_gan\dgd\denoising-diffusion-gan-data"
REAL_SRC          = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\ffhq\ffhq\images"

# === DESTINATION FOLDER ===
DST_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\unseen_gen_real"
os.makedirs(DST_DIR, exist_ok=True)

# === Helper function ===
def get_images(path, n):
    imgs = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)
    return imgs[:n]

# === Select samples ===
diffusion_gan_imgs = get_images(DIFFUSION_GAN_SRC, 200)
stable_male_imgs   = get_images(STABLE_DIFF_MALE, 100)
stable_female_imgs = get_images(STABLE_DIFF_FEMALE, 100)
denoising_imgs     = get_images(DENOISING_DIFF_GAN, 100)
real_imgs          = get_images(REAL_SRC, 500)

print(f"Fake totals → DiffusionGAN={len(diffusion_gan_imgs)}, Stable(M+F)={len(stable_male_imgs)+len(stable_female_imgs)}, DenoisingDiffGAN={len(denoising_imgs)}")
print(f"Real totals → FFHQ={len(real_imgs)}")

# === Prepare CSV log ===
csv_path = os.path.join(DST_DIR, "labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "source"])

    # --- Copy FAKE images ---
    counter = 1
    for src_folder, img_list, src_name in [
        (DIFFUSION_GAN_SRC, diffusion_gan_imgs, "diffusion_gan"),
        (STABLE_DIFF_MALE, stable_male_imgs, "stable_male"),
        (STABLE_DIFF_FEMALE, stable_female_imgs, "stable_female"),
        (DENOISING_DIFF_GAN, denoising_imgs, "denoising_diff_gan")
    ]:
        for fname in img_list:
            src = os.path.join(src_folder, fname)
            dst_name = f"fake_{counter:04d}_{src_name}.png"
            dst = os.path.join(DST_DIR, dst_name)
            shutil.copy2(src, dst)
            writer.writerow([dst_name, "fake", src_name])
            counter += 1
        print(f"✅ Copied {len(img_list)} fakes from {src_name}")

    # --- Copy REAL images ---
    for i, fname in enumerate(real_imgs, start=1):
        src = os.path.join(REAL_SRC, fname)
        dst_name = f"real_{i:04d}.png"
        dst = os.path.join(DST_DIR, dst_name)
        shutil.copy2(src, dst)
        writer.writerow([dst_name, "real", "ffhq"])
        if i % 100 == 0:
            print(f"✅ Copied {i} real images")

print("🎉 Done! Created unseen_gen_real dataset (500 fakes + 500 real)")
print("📂 Folder:", DST_DIR)
print("📝 Labels saved to:", csv_path)
