# Phase 1: Image Data Exploration
# explore_data.py
import os
from PIL import Image
import matplotlib.pyplot as plt

base_dir = "dataset/cats"
img_name = os.listdir(base_dir)[0]
img = Image.open(os.path.join(base_dir, img_name))

print(f"Size: {img.size}, Mode: {img.mode}")
plt.imshow(img)
plt.axis('off')
plt.show()