# Tutorial 06 - Transfer Learning: Pre-Trained Models

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
<br>
<br><img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/34082241-3dbe-4cdc-89b8-17fdc1c42fe6" />

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/4f9e98b2-15a2-4bb6-bcd0-a486e951fad1" />

## Results
For the test Image we used one that contained multiple objects represented in the labels (daisy, bee & butterfly)
<br>
<br>![daisy](https://github.com/user-attachments/assets/68313d8f-594f-46a8-89bb-b45742e15a67)

<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/aca6364e-3a28-4bec-b7ad-2875e994cb9c" />


## Task 01: Pytorch Implementation

<img width="1000" height="300" alt="image" src="https://github.com/user-attachments/assets/9fdb6e42-8e46-4685-a01a-1d917b4c7821" />


## Task 02: Comparing Architectures (AlexNet, ResNet101, MobileNet)

All architectures agreed on the same class **bee** being important but ranked classes differently depending on their learned representations and capacity.
<br>
<br><img width="1000" height="650" alt="image" src="https://github.com/user-attachments/assets/1b156698-1320-47f7-8db0-7b1e1d3577ad" />


## Task 03: Transfer Learning on a Custom Dataset

The final task in the PDF was to use transfer learning on a dataset of my own. 
Instead of only predicting labels from ImageNet classes, transfer learning can be used to adapt a pretrained model to a new dataset by keeping the learned backbone and replacing the final classifier.
That is an important concept, it shows how pretrained models are useful not only for ready-made predictions, but also for practical custom classification problems.
<br>
<br><img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/4c5f944e-19e2-43d3-9586-1898bb60cda2" />

<img width="300" height="300" alt="stepper-motor-overheats" src="https://github.com/user-attachments/assets/b6be9ad2-a7a4-478c-9313-7ba87e498900" />  <img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/9259f47a-e657-4683-b1f5-58ceaf9081a4" />

Test Image of stepper motor vs Example of A50 Motor (prediction) from dataset

## Key Takeaways
- many tasks do not require training from scratch
- the input image must be resized and preprocessed correctly
- transfer learning can be extended to a custom dataset by fine-tuning or replacing the final layer
