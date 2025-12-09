# Spark-Based Deep Convolutional Neural Network for Pneumonia Detection in Chest X-Rays

## Background
Pneumonia remains a major cause of illness and mortality worldwide, especially among children, elderly adults, and immunocompromised patients. Chest X-rays are the most common diagnostic tool, but interpreting them requires radiology expertise and can be time-consuming in high-volume clinical settings.

This project implements a Spark-based deep learning pipeline to classify chest X-ray images as Normal, Bacterial Pneumonia, or Viral Pneumonia using a Convolutional Neural Network (CNN). By using distributed computing across a Spark cluster with one master and multiple worker nodes, the project is able to show how medical imaging workloads can be scaled efficiently on CPU-only environments. Note that even though there are 3 possible classes in this dataset, we chose to do a formal classification problem with 2 classes, Normal and Pneumonia. 

## Features
- Implemented using PySpark
- Loads and preprocesses thousands of image file paths across multiple workers
- Performs distributed resizing, normalization, and label extraction
- Handles dataset reorganization into train/validation/test splits

## Image Augmentation and Preprocessing
- Normalization of pixel intensity values
- Resizing to 64x64 uniform input size

## Model Architecture
- Block 1
      - 1 to 16 to 32 channels
      - Conv2d(1, 16, 3×3, stride=1, padding=1)
      - BatchNorm2d(16) & ReLU
      - Conv2d(16, 16, 3×3, stride=1, padding=1)
      - BatchNorm2d(16) & ReLU
      - Conv2d(16, 32, 3×3, stride=2, padding=1)
- Block 2
      - 32 to 64 to 128 channels
      - Conv2d(32, 64, 3×3, stride=1, padding=1)
      - BatchNorm2d(64) & ReLU
      - Conv2d(64, 64, 3×3, stride=1, padding=1)
      - BatchNorm2d(64) & ReLU
      - Conv2d(64, 128, 3×3, stride=2, padding=1)
- Block 3
      - 128 to 256 channels
      - Conv2d(128, 128, 3×3, stride=1, padding=1)
      - BatchNorm2d(64) & ReLU
      - Conv2d(128, 128, 3×3, stride=1, padding=1)
      - BatchNorm2d(64) & ReLU
      - Conv2d(128, 256, 3×3, stride=2, padding=1)
- Global Pooling
      - AdaptiveAvgPool2d((1, 1))
- Classifier
      - Flatten()
      - Dropout(p=0.5)
      - Linear(256 to 128) & ReLU
      - Dropout(p=0.5)
      - Linear(128 to num_classes)

## Training Configuration
- Device: CPU on the Spark master VM
- Batch size: 32
- Optimizer: Adam (lr = 1e-3)
- Loss: CrossEntropyLoss
- Regularization:
      - L2 via weight_decay=1e-4
      - Additional manual L1 penalty on all weights
- Epochs: Up to 10, with validation-based early stopping
- Best checkpoint saved automatically

## Validation and Early Stopping
- Each epoch runs:
      - train_one_epoch on the training set
      - evaluate on validation set
      - Model saved when validation accuracy improves
      - The best model occurred at Epoch 4

## Final Evaluation
To avoid VM instability, test evaluation is done in a standalone script:
- Loads Spark manifest
- Builds test DataLoader
- Re-creates CNN architecture
- Loads best checkpoint (best_pneumonia_cnn_manifest.pt)
- Computes:
      - Test loss
      - Test accuracy
      - Confusion matrix
      - Classification report

## Dataset
Source: Kaggle – Chest X-Ray Images for Pneumonia Detection with Deep Learning

Link: https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

## Cluster Setup
- 1 Master Node
- 3 Worker Nodes
- Passwordless SSH configured between nodes
- Spark Standalone mode used with the Spark master at spark://master:7077

## Requirements   
- Python 3.x
- Apache Spark installed on all VMs
- PySpark, NumPy, Pandas


## Team Members
- Andrea Wroblewski
- Olivia Gette

