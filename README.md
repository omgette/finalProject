# Spark-Based Deep Convolutional Neural Network for Pneumonia Detection in Chest X-Rays

## Background
Pneumonia remains a major cause of illness and mortality worldwide, especially among children, elderly adults, and immunocompromised patients. Chest X-rays are the most common diagnostic tool, but interpreting them requires radiology expertise and can be time-consuming in high-volume clinical settings.

This project implements a Spark-based deep learning pipeline to classify chest X-ray images as Normal, Bacterial Pneumonia, or Viral Pneumonia using a Convolutional Neural Network (CNN). By using distributed computing across a Spark cluster with one master and multiple worker nodes, the project is able to show how medical imaging workloads can be scaled efficiently on CPU-only environments.

## Features
- Implemented using PySpark
- Loads and preprocesses thousands of image file paths across multiple workers
- Performs distributed resizing, normalization, and label extraction
- Handles dataset reorganization into train/validation/test splits

## Image Augmentation and Preprocessing
- Normalization of pixel intensity values
- Resizing to 224×224 uniform input size
- Data augmentation including:
    - Random rotation (15°)
    - Horizontal flipping
    - Brightness & contrast adjustments
    - Zoom up to 10%

## Model Training
- Custom CNN architecture
- Uses Conv2D -> ReLU -> MaxPooling blocks for feature extraction
- Dense and Dropout layers for regularization
- Softmax classification head for 3 classes
- Training logs, accuracy curves, and loss curves generated for monitoring

## Distributed Hyperparameter Tuning
- Runs multiple experiments in parallel across Spark worker nodes
- Parameters tuned include:
    - Learning rate
    - Batch size
    - Number of filters
    - Regularization strength

## Machine Learning Model
### Convolutional Neural Network (CNN)
- Input: 224x224 grayscale X-ray images
- 3 convolutional blocks:
    - Filters: 32, 64, 128
    - Kernel size: 3×3
    - MaxPooling layers
- Flatten -> Dense -> Dropout (0.3)
- Softmax output layer (3 classes)
- Optimizer: Adam (lr = 0.001; tuned with Spark)
- Regularization: L2 = 0.001

#### Training Configuration
- 70/15/15 train-validation-test split
- Early stopping enabled
- Batch size: 32 (also tuned)
- about 20-30 training epochs

## Dataset
Source: Kaggle – Chest X-Ray Images for Pneumonia Detection with Deep Learning

Link: https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

## Performace Across VMS

| **Number of VMs** | **Cores Used (8 per VM)** | **Total Runtime (min)** | **Model Training + Evaluation Time (s)** |
| ----------------- | ------------------------- | ----------------------- | ---------------------------------------- |
| 1 VM              | 8                         |                         |                                          |
| 2 VMs             | 16                        |                         |                                          |
| 3 VMs             | 24                        |                         |                                          |
| 4 VMs             | 32                        |                         |                                          |


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

