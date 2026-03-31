# SCRIPT 04: MODEL TRAINING AND EVALUATION
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report 
from PIL import Image
import os


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Run SCRIPT 03 first.") 
    
    df = pd.read_csv(csv_path)
    X, y = [], []
    print(f"Loading images from {csv_path}...")
    
    for _, row in df.iterrows():
        
        img_path = os.path.join("images_resized", row["Image Name"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            X.append(np.array(img) / 255.0) 
            y.append(row["label_numeric"]) 
            
    return np.array(X), np.array(y)

X_train, y_train = load_data("train_dataset.csv")
X_test, y_test = load_data("test_dataset.csv")


model = models.Sequential([
    layers.Input(shape=(224, 224, 3)), 
    layers.Conv2D(32, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') 
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("\nStarting training...")
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2
)

model.save("glaucoma_model.h5") 
print("\nModel saved successfully as glaucoma_model.h5")

print("\nGenerating evaluation metrics...")
predictions = model.predict(X_test)
preds = (predictions > 0.5).astype(int) 

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds)) 

print("\nClassification Report:")
print(classification_report(y_test, preds)) 