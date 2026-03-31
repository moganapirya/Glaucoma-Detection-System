# SCRIPT 03: DATASET SPLITTING
import pandas as pd
from sklearn.model_selection import train_test_split 
df = pd.read_csv("glaucoma_clean_dataset.csv")

# 1. Split by Patient ID to avoid leakage 
patients = df["Patient"].unique()
train_pts, test_pts = train_test_split(patients, test_size=0.2, random_state=42)

# 2. Create DataFrames
train_df = df[df["Patient"].isin(train_pts)]
test_df = df[df["Patient"].isin(test_pts)]

# 3. Save for Training 
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print(f"Training Images: {len(train_df)}") 
print(f"Testing Images: {len(test_df)}")  