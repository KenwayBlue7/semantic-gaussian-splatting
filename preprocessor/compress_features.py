import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# 1. Setup paths
# Update these to match your directory structure
input_dir = "D:/Gaus/MySplat1/features"
output_dir = "D:/Gaus/MySplat1/features_pca"
os.makedirs(output_dir, exist_ok=True)

# Get list of all feature files
feature_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
feature_files.sort() # Ensure consistent order

if not feature_files:
    print(f"Error: No .npy files found in {input_dir}")
    exit()

# 2. Fit PCA (Phase 1: Learning the "Semantic Essence")
# We reduce from 1024 dimensions down to 32 (32-bit float)
n_components = 32
ipca = IncrementalPCA(n_components=n_components, batch_size=10)

print(f"Phase 1: Fitting PCA on {len(feature_files)} files...")
for f in tqdm(feature_files):
    feat = np.load(os.path.join(input_dir, f))
    
    # DINOv2 ViT-L usually gives [1, 1369, 1024]
    # We flatten the spatial dimensions to treat every patch as a data point
    feat_flattened = feat.reshape(-1, 1024)
    
    # Partially fit the PCA model with this image's patches
    ipca.partial_fit(feat_flattened)

# 3. Transform and Save (Phase 2: Compressing the files)
print(f"Phase 2: Saving compressed 32-dim features to {output_dir}...")
for f in tqdm(feature_files):
    feat = np.load(os.path.join(input_dir, f))
    original_shape = feat.shape # [1, 1369, 1024]
    
    feat_flattened = feat.reshape(-1, 1024)
    
    # Transform to 32 dimensions
    compressed = ipca.transform(feat_flattened)
    
    # Reshape back to maintain spatial structure [1, 1369, 32]
    # This is critical so the training loop can resize it to [H, W]
    compressed_reshaped = compressed.reshape(original_shape[0], original_shape[1], n_components)
    
    # Explicitly save as float32 to save disk space and VRAM
    out_path = os.path.join(output_dir, f)
    np.save(out_path, compressed_reshaped.astype(np.float32))

print("All features compressed successfully!")