import numpy as np
import os
from sklearn.decomposition import PCA

input_dir = "D:/Gaus/MySplat1/features"
output_dir = "D:/Gaus/MySplat1/features_pca"
os.makedirs(output_dir, exist_ok=True)

# 1. Collect all features to find global PCA
all_feats = []
files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

print("Loading features for PCA calculation...")
for f in files:
    feat = np.load(os.path.join(input_dir, f))
    # DINOv2 ViT-L usually gives [1, 1369, 1024]
    all_feats.append(feat.reshape(-1, 1024))

all_feats = np.vstack(all_feats)

# 2. Fit PCA to reduce from 1024 to 32
pca = PCA(n_components=32)
print("Fitting PCA...")
pca.fit(all_feats)

# 3. Transform and save
print("Saving compressed features...")
for f in files:
    feat = np.load(os.path.join(input_dir, f)).reshape(-1, 1024)
    compressed = pca.transform(feat)
    np.save(os.path.join(output_dir, f), compressed.astype(np.float32))

print(f"Done! Compressed features in {output_dir}")