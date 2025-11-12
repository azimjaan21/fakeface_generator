"""
Evaluate StyleGAN3 Discriminator Across 4 Experiments (Probabilistic D-Accuracy)
--------------------------------------------------------------------------------
Training:
  y = 1 → Fake
  y = 0 → Real

Testing (probability threshold):
  ŷ_b > 0.5 → Fake
  ŷ_b ≤ 0.5 → Real

Outputs:
  - Mean ± Std of probabilities
  - D-Accuracy (%)
  - AUROC
  - *_hist.png
  - *_classification_bar.png (True Fake / True Real counts)
  - discriminator_scores.png
  - summary_accuracy_bar.png
  - advanced_d_accuracy.png
  - discriminator_summary.txt
"""

import sys, os, torch, numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# ============ CONFIG ============
RUN_MODE = "all"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
stylegan3_path = os.path.join(project_root, 'stylegan3')
if stylegan3_path not in sys.path:
    sys.path.insert(0, stylegan3_path)

import dnnlib, legacy

MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\models\stylegan3-t-ffhq-1024x1024.pkl"
DATA_PATHS = {
    "1-Fake(Self-StyleGAN3)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\fake\run01",
    "2-Real(FFHQ)":            r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\real",
    "3-Mixed(StyleGAN3+FFHQ)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\mixed_half",
    "4-Unseen(DiffGens+FFHQ)": r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\unseen_gen_real",
}
OUT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\results"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ LOAD DISCRIMINATOR ============
print("🔹 Loading StyleGAN3 discriminator...")
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

# --- Return discriminator probability (after sigmoid) ---
def get_score(path):
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = D(img_t, None)
        prob = torch.sigmoid(logit)        # Convert logits → probabilities [0,1]
        return float(prob.cpu().item())

# --- Assign label (0 = Real, 1 = Fake) ---
def get_label(path, exp_name):
    name = os.path.basename(path).lower()
    if "1-fake" in exp_name: return 1
    if "2-real" in exp_name: return 0
    if "fake_" in name: return 1
    if "real_" in name: return 0
    return 0

# ============ EVALUATION ============
def evaluate_folder(folder, label_name):
    if not os.path.exists(folder):
        print(f"⚠️ Missing: {folder}")
        return [], []
    files = [os.path.join(folder,f) for f in os.listdir(folder)
             if f.lower().endswith((".jpg",".png"))]
    scores, labels = [], []
    for f in tqdm(files, desc=f"Evaluating {label_name}", ncols=80):
        try:
            scores.append(get_score(f))
            labels.append(get_label(f, label_name))
        except Exception as e:
            print(f"⚠️ Skipped {f}: {e}")
    if not scores: return [], []
    scores, labels = np.array(scores), np.array(labels)
    mean, std = scores.mean(), scores.std()
    preds = (scores > 0.5).astype(int)
    acc = (preds == labels).mean()
    auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan
    print(f"📊 {label_name}: mean={mean:.3f}, std={std:.3f}, D-Acc={acc*100:.1f}%, AUROC={auroc:.3f}, n={len(scores)}")
    return scores, labels

# ============ RUN EXPERIMENTS ============
all_scores, all_labels = {}, {}
for name, path in DATA_PATHS.items():
    if RUN_MODE != "all" and not name.startswith(RUN_MODE): continue
    s, l = evaluate_folder(path, name)
    if len(s): all_scores[name], all_labels[name] = s, l

# ============ VISUALS ============
summary = []
for name, scores in all_scores.items():
    labels = all_labels[name]
    mean, std = np.mean(scores), np.std(scores)
    preds = (scores > 0.5).astype(int)
    acc = (preds == labels).mean()
    auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan
    summary.append((name, mean, std, acc, auroc))

    # Histogram
    plt.figure(figsize=(7,5))
    plt.hist(scores, bins=40, color="#69b3a2", alpha=0.7, edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=1.5)
    plt.title(f"{name}\nMean={mean:.3f}, Std={std:.3f}, D-Acc={acc*100:.1f}%, AUROC={auroc:.3f}")
    plt.xlabel("Discriminator Probability (ŷ_b)")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f"{name}_hist.png"))
    plt.close()

    # True Fake / True Real counts
    TP = np.sum((preds==1)&(labels==1))
    TN = np.sum((preds==0)&(labels==0))
    plt.figure(figsize=(5,4))
    plt.bar(["True Fake","True Real"], [TP, TN], color=["#ff7f7f","#90ee90"], edgecolor='black')
    plt.title(f"Correct Predictions — {name}")
    for i,v in enumerate([TP,TN]):
        plt.text(i, v + max(1, v*0.02), f"{v}", ha='center', fontsize=10, weight='bold')
    plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f"{name}_classification_bar.png"))
    plt.close()

print("📊 Saved per-experiment histograms and True Fake/Real counts!")

# ============ COMBINED HISTOGRAM ============
plt.figure(figsize=(9,6))
for n,s in all_scores.items():
    plt.hist(s,bins=50,alpha=0.5,label=n,edgecolor='black')
plt.axvline(0.5,color='red',ls='--',lw=1)
plt.xlabel("Discriminator Probability (ŷ_b)")
plt.ylabel("Image Count")
plt.title("Overall Discriminator Response — All Experiments")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"discriminator_scores.png"))
plt.close()
print("📈 Saved combined histogram → discriminator_scores.png")

# ============ SUMMARY BAR (Mean ± Std + D-Acc %) ============
labels  = [x[0] for x in summary]
means   = [x[1] for x in summary]
stds    = [x[2] for x in summary]
accs    = [x[3]*100 for x in summary]
colors  = ['#FF4C4C','#4CAF50','#9C27B0','#000000']

fig, ax1 = plt.subplots(figsize=(9,5))
ax1.bar(labels, means, yerr=stds, color=colors, capsize=6, alpha=0.85)
ax1.set_ylabel("Mean Discriminator Probability (ŷ_b)")
ax1.set_title("Summary of Discriminator Performance (Mean ± Std, D-Accuracy %)")
ax1.axhline(0.5,color='gray',ls='--',lw=1)

ax2 = ax1.twinx()
ax2.plot(labels, accs, color='blue', marker='o', lw=2.5, label='D-Accuracy %')
ax2.set_ylabel("Accuracy (%)"); ax2.set_ylim(0,100)
for i,a in enumerate(accs):
    ax2.text(i, a+3, f"{a:.1f}%", color='blue', ha='center', fontsize=9, weight='bold')
ax2.legend(loc='upper left')
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"summary_accuracy_bar.png"))
plt.close()
print("📊 Saved summary accuracy bar → summary_accuracy_bar.png")

# ============ ADVANCED D-ACCURACY COMPARISON ============
plt.figure(figsize=(10,6))
bars = plt.barh(labels, accs, color=colors, edgecolor='black', height=0.55)
plt.xlabel("Discriminator Accuracy (%)", fontsize=12)
plt.title("Advanced Comparison of D-Accuracy Across Experiments", fontsize=14, weight='bold')
plt.xlim(0, 100)
for bar, acc in zip(bars, accs):
    plt.text(acc + 1, bar.get_y() + bar.get_height()/2, f"{acc:.1f}%", va='center', fontsize=11, weight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "advanced_d_accuracy.png"))
plt.close()
print("🎯 Saved advanced D-Accuracy comparison → advanced_d_accuracy.png")

# ============ TEXT SUMMARY ============
txt = os.path.join(OUT_DIR,"discriminator_summary.txt")
with open(txt,"w") as f:
    for n,m,s,a,u in summary:
        f.write(f"{n}: mean={m:.3f}, std={s:.3f}, D-Acc={a*100:.1f}%, AUROC={u:.3f}, n={len(all_scores[n])}\n")
print(f"📝 Summary saved → {txt}")
print("\n✅ All experiments evaluated and probabilistic visualizations generated successfully!")
