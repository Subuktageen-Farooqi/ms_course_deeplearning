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
This code compares two different strategies:

- Strategy 1: VGG16 feature extraction
  - pretrained backbone mostly frozen
  - train final classifier for CIFAR-10
- Strategy 2: ResNet50 staged fine-tuning
  - phase 1: train classifier head only
  - phase 2: unfreeze last block + classifier and fine-tune

So it is not just comparing two models. It is comparing two transfer-learning workflows.<br>
Conclusion: VGG16 feature extraction performed better on CIFAR-10 in this run.
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/116ff8e2-054e-499f-a1ba-29799d2e0d4a" />


### Task 02: Improve fine-tuning results. 
Using `Changing learning rate` `Prevent overfitting` `Early stopping` `Preprocessing formatting` `Number of Epochs` `Unfreeze more layers` `Different layers`
- **Baseline**

<img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/2a6af47b-b11e-458c-b25d-0bda2c3b79cc" /><br>

- **Experiment 01: More Epochs + Lower Learning Rate + Early Stopping**

  - Lower Learning Rate: a high LR can distort pretrained weights too aggressively
  - Early Stopping: prevents unnecessary training
  
<br><img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/feb8c34d-ac71-4f10-a721-8da7ae0523c9" /><br>
Only slight improvement over baseline.

- **Experiment 02: More Epochs + Unfreeze more layer + Dropout + Weight Decay**

  - Unfreezing more Layers: more of the feature hierarchy can specialize to the new dataset
  - Dropout: Because increasing trainable layers increases overfitting risk
  - Weight decay: controls model complexity by discouraging overly large weights
  
<br><img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/6672b927-c473-42bb-99b1-31000dcb26ff" /><br>
Decent improvement observed.



### Task 03: Feature extraction with Resnet50. Compare the results with your custom model and VGG16

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/6a5a61c2-3d42-4ee3-9b9b-edd253b94e7c" />

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/fc272990-1fa3-4ce7-b7d0-6eae7ba73698" />
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/58607517-e313-415e-af76-9dd48e0e6568" />

### Task 04: Finetuning with VGG 16 and feature extraction with Resnet50

<img width="1000" height="300" alt="image" src="https://github.com/user-attachments/assets/44852c9f-e352-476f-9ecb-32d0069eb8f9" />

VGG16 fine-tuning performed better than ResNet50 feature extraction in this experiment.

<img width="1000" height="550" alt="image" src="https://github.com/user-attachments/assets/f0b26e5d-fb01-4058-93ba-532af18fb166" />

## Key Takeaways

The main things I learned were:

- transfer learning can be done in more than one way
- freezing all pretrained layers gives a fast baseline
- unfreezing some layers allows deeper adaptation
- evaluation should be done on a held-out test set
- improvements to fine-tuning often come from better hyperparameters and regularization, not just more epochs




