# SCRIPT 01: ENVIRONMENT AND DATA LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. Verify Environment [cite: 206-210]
print("Libraries loaded successfully!")
print(f"Pandas version: {pd.__version__}")

# 2. Load and Inspect Dataset [cite: 299-307]
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns: df.drop(columns=["Unnamed: 4"], inplace=True) 
print(f"\nDataset Shape: {df.shape}") # (747, 4) [cite: 322]
print("\nLabel Distribution:")
print(df["Label"].value_counts()) 

# 3. Patient and Quality Analysis 
print(f"\nUnique Patients: {df['Patient'].nunique()}") 
print("\nQuality Score Statistics:")
print(df["Quality Score"].describe())

# 4. Visualize Distribution 
df["Label"].value_counts().plot(kind="bar")
plt.title("Glaucoma Label Distribution")
plt.show()