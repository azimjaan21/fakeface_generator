import torch, dnnlib, legacy

with open("models/stylegan3-t-ffhq-1024x1024.pkl", "rb") as f:
    data = legacy.load_network_pkl(f)
    print("✅ Keys:", data.keys())
    D = data['D']
    print("Discriminator type:", type(D))
