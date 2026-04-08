## Overview

This tutorial introduced **transfer learning with pretrained image classification models**, specifically **VGG16** and **ResNet50**. 
The main workflow loaded an image, preprocessed it according to the selected model, ran inference using pretrained ImageNet weights, and printed the **top 5 predicted labels** with their probabilities.


## What I Learned
- what pretrained models are and why they are useful
- how to use **VGG16** and **ResNet50** for image classification
- how image preprocessing must match the selected model
- how to obtain and interpret the **top 5 predictions**
- how different pretrained architectures can produce slightly different predictions
- how transfer learning can be extended from inference to training on a custom dataset


## Tutorial Workflow
1. import required libraries  
2. load and preprocess an image for the chosen model  
3. load pretrained **VGG16** and **ResNet50**  
4. make predictions  
5. define the image path  
6. print the top 5 predictions for both models


## Preprocessing the Image

A key idea was that the image must be resized and preprocessed properly before it is passed into the network. The PDF uses a function that:

- loads the image
- resizes it to **224 × 224**
- converts it to an array
- adds the batch dimension
- applies model-specific preprocessing for either VGG16 or ResNet50

This was important because pretrained models expect inputs in a very specific format.

**Stub: Insert screenshot of preprocessing function or input image here**

## Using Pretrained Models

The tutorial then loads **VGG16** and **ResNet50** with pretrained **ImageNet** weights and uses them directly for prediction.

This helped me understand the practical meaning of transfer learning: instead of training a large model from scratch, I can reuse a network that already learned useful visual features from a very large dataset.

## Task 01: Pytorch Implementation



**Stub: Insert screenshot of VGG16 and ResNet50 top-5 predictions here**

## Task 02: Comparing Architectures
- **AlexNet**
- **ResNet101**
- **MobileNet**

Different architectures may agree on the same class or may rank classes differently depending on their learned representations and capacity.

## Task 03: Transfer Learning on a Custom Dataset

The final task in the PDF was to use transfer learning on a dataset of my own. 
Instead of only predicting labels from ImageNet classes, transfer learning can be used to adapt a pretrained model to a new dataset by keeping the learned backbone and replacing the final classifier.
That is an important concept, it shows how pretrained models are useful not only for ready-made predictions, but also for practical custom classification problems.

## Results and Reflection


## Key Takeaways
- **VGG16** and **ResNet50** can be used directly for classification
- the input image must be resized and preprocessed correctly
- prediction outputs should be interpreted through the top 5 labels and probabilities
- trying multiple architectures helps compare model behavior
- transfer learning can be extended to a custom dataset by fine-tuning or replacing the final layer
