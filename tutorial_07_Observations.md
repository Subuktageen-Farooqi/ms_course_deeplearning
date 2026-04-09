# Tutorial 7 — Compare Feature Extraction vs. Fine-Tuning

## Overview

This tutorial focused on comparing two transfer learning strategies on the **CIFAR-10** dataset:

- **Feature Extraction** using **VGG16**
- **Fine-Tuning** using **ResNet50**

The tutorial workflow was to load and preprocess CIFAR-10, use pretrained ImageNet models, freeze layers for feature extraction, unfreeze some layers for fine-tuning, then evaluate and compare both approaches.


## What I Learned
- the difference between **feature extraction** and **fine-tuning**
- how pretrained ImageNet models can be adapted for CIFAR-10
- why freezing layers reduces training cost (adaptation vs retraining)
- why unfreezing later layers can improve task adaptation
- how to compare transfer-learning strategies fairly
- how learning rate, overfitting control, preprocessing, epochs, and unfreezing depth affect fine-tuning results

## Transfer Learning Concepts

### 1. Feature Extraction

In feature extraction, the pretrained backbone is kept frozen and only the new classifier is trained. This is useful when:

- training data is limited
- faster training is needed
- we want to reuse general visual features learned from ImageNet

In this tutorial, **VGG16** was used for feature extraction.

### 2. Fine-Tuning

In fine-tuning, some pretrained layers are unfrozen and retrained together with the classifier head. This allows the model to adapt its learned features more specifically to the new dataset.

In this tutorial, **ResNet50** was used for fine-tuning by unfreezing the last few layers.


## Dataset and Preprocessing

The tutorial used **CIFAR-10**, which contains 10 classes of small color images. The preprocessing steps were:

- loading CIFAR-10
- normalizing pixel values to `[0, 1]`
- converting labels to one-hot encoding in the TensorFlow version

In the PyTorch implementations, CIFAR-10 was also resized and normalized to match pretrained model expectations.

## Tutorial Workflow

The overall tutorial flow was:

1. import libraries  
2. load and preprocess CIFAR-10  
3. load pretrained models  
4. perform feature extraction with VGG16  
5. perform fine-tuning with ResNet50  
6. evaluate both models  
7. compare results  


## Results and Interpretation

The tutorial’s main conceptual takeaway is that:

- **feature extraction** is simpler and cheaper
- **fine-tuning** is more flexible and can potentially achieve better task-specific performance
- the best approach depends on dataset size, preprocessing quality, number of trainable layers, regularization, and optimization setup

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/3da56180-e218-40f1-8d32-78177ca099a1" />
<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/397604a2-e699-4649-8c56-1078e5d0a959" />

## Tasks

### Task 01: PyTorch Implementation


### Task 02: Improve fine-tuning results. 
Using `Changing learning rate` `Prevent overfitting` `Early stopping` `Preprocessing formatting` `Number of Epochs` `Unfreeze more layers` `Different layers`


### Task 03: Feature extraction with Resnet50. Compare the results with your custom model and VGG16



### Task 04: Finetuning with VGG 16 and feature extraction with Resnet50



## Key Takeaways

The main things I learned were:

- transfer learning can be done in more than one way
- freezing all pretrained layers gives a fast baseline
- unfreezing some layers allows deeper adaptation
- evaluation should be done on a held-out test set
- improvements to fine-tuning often come from better hyperparameters and regularization, not just more epochs




