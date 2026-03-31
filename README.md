Glaucoma is a leading cause of irreversible blindness. Our mission for IDSC 2026 was to develop a robust, high-recall screening tool that ensures no positive case goes undetected, bringing "Hope in Healthcare" through accessible technology.

Dataset: Hillel Yaffe Glaucoma Dataset (HYGD).

Architecture: Convolutional Neural Network (CNN).

Primary Achievement: 100% Recall for Glaucoma cases (Zero False Negatives).

Accuracy: 79% overall on the test set.

Methodology

Our pipeline consists of four critical stages:

Input: Processing high-resolution retinal images from the HYGD dataset.

Preprocessing: Applied specialized Noise Reduction and image normalization to enhance clinical features.

Model: A custom-trained CNN optimized for spatial feature extraction in the optic disc.

Classification: Binary classification separating 'Healthy' from 'Glaucoma' with high confidence.

