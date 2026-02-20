import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# 1. Setup paths
video_dir = "D:/Gaus/Buddha/input"
output_dir = "D:/Gaus/Buddha/masks"
os.makedirs(output_dir, exist_ok=True)

# Use the 2.1 versions if you downloaded them, otherwise use 2.0
# Ensure these files are in your 'preprocessor' folder
checkpoint = "./sam2.1_hiera_large.pt" 
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # Point to the actual file in the sam2 folder

# 2. Initialize SAM 2 Predictor
# We point directly to the config file inside the segment-anything-2 folder
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 3. Initialize State
inference_state = predictor.init_state(video_path=video_dir)

# 4. Add Point Prompt 
# Use Paint to find coordinates for your object!
predictor.add_new_points(inference_state, frame_idx=0, obj_id=1, 
                         points=[[500, 500]], labels=[1])

# 5. Run Propagation and Save
print("Starting mask propagation...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        mask_image = (mask * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f"mask_{frame_idx:04d}.png")
        cv2.imwrite(out_path, mask_image)

print(f"Done! Check your masks in {output_dir}")