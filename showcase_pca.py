import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA
import os

def create_pca_visual(input_ply, output_name):
    print(f"Opening trained model: {input_ply}")
    
    plydata = PlyData.read(input_ply)
    v = plydata['vertex']
    
    s_names = [p.name for p in v.properties if p.name.startswith('f_s_')]
    s_names = sorted(s_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(s_names) == 0:
        print("Error: No semantic features found!")
        return

    features = np.zeros((v.count, len(s_names)))
    for i, name in enumerate(s_names):
        features[:, i] = v[name]

    print("Squashing 32D data into RGB colors...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min() + 1e-7)
    rgb_colors = (pca_features * 255).astype(np.uint8)

    # --- FIX STARTS HERE: Adding Scalar Field Support ---
    # We define a structure that includes x, y, z, red, green, blue AND the 32 features
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # Add each feature name to the structure so CloudCompare sees them as "Scalar Fields"
    for name in s_names:
        dtype_list.append((name, 'f4'))

    new_data = np.empty(v.count, dtype=dtype_list)
    
    new_data['x'] = v['x']
    new_data['y'] = v['y']
    new_data['z'] = v['z']
    new_data['red'] = rgb_colors[:, 0]
    new_data['green'] = rgb_colors[:, 1]
    new_data['blue'] = rgb_colors[:, 2]

    # Fill the extra columns with the actual feature data
    for i, name in enumerate(s_names):
        new_data[name] = features[:, i]
    # --- FIX ENDS HERE ---

    el = PlyElement.describe(new_data, 'vertex')
    PlyData([el]).write(output_name)
    print(f"SUCCESS! PCA Cloud saved as: {output_name}")

# IMPORTANT: Keep the 'r' before the quote to avoid the Unicode error
input_path = r"C:\Users\Naveen R\Downloads\Gaussian\gaussian-splatting\output\f723b040-d\point_cloud\iteration_30000\point_cloud.ply"
create_pca_visual(input_path, "semantic_rainbow.ply")