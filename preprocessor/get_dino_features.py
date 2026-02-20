import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torchvision import transforms

# 1. Setup
input_dir = "D:/Gaus/Buddha/input"
output_dir = "D:/Gaus/Buddha/features"
os.makedirs(output_dir, exist_ok=True)

# 2. Load DINOv2 Model (ViT-L is best for 8GB VRAM balance)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
dinov2_vitl14.eval()

# 3. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((518, 518)), # DINOv2 works in multiples of 14
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Extracting DINOv2 Features...")
with torch.no_grad():
    for img_name in os.listdir(input_dir):
        if not img_name.endswith(".jpg"): continue
        
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        # Extract features (patch features)
        features = dinov2_vitl14.get_intermediate_layers(img_t, n=1)[0] 
        # features shape: [1, 1369, 1024] (for 518x518 input)
        
        # Save as compressed numpy file to save space
        out_name = os.path.splitext(img_name)[0] + ".npy"
        np.save(os.path.join(output_dir, out_name), features.cpu().numpy())
        print(f"Processed {img_name}")

print(f"Success! Features saved to {output_dir}")