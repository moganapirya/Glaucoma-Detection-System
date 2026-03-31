# SCRIPT 02: IMAGE PREPROCESSING
import pandas as pd
import os
import numpy as np
from PIL import Image

df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns: df.drop(columns=["Unnamed: 4"], inplace=True)

# 1. Link Images and Filter Quality [cite: 451, 724]
image_folder = "images"
resized_folder = "images_resized"
if not os.path.exists(resized_folder): os.makedirs(resized_folder)

# Filter images with Quality Score >= 5 [cite: 724]
df = df[df["Quality Score"] >= 5] 

# 2. Resize and Normalize [cite: 675-680, 704]
images, labels = [], []
label_map = {"GON+": 1, "GON-": 0} 
print("Processing images...")
for index, row in df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path).resize((224, 224)) 
    
    img.save(os.path.join(resized_folder, row["Image Name"]))
    
    img_array = np.array(img) / 255.0
    images.append(img_array)
    labels.append(label_map[row["Label"]])

df["label_numeric"] = df["Label"].map(label_map)
df.to_csv("glaucoma_clean_dataset.csv", index=False)
print(f"Processed Dataset Shape: {np.array(images).shape}") 