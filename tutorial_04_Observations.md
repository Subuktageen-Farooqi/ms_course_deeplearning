# Tutorial 4 — Data Augmentation

## Overview

This tutorial explored  **data augmentation** as a practical way to increase dataset size and variability when collecting large numbers of diverse images is difficult. Deep learning models generally perform better with more diverse data, and augmentation helps create that diversity from existing images.

I implemented augmentation pipelines in both **Keras** and **PyTorch**. 
The original tutorial focused on augmenting a **single JPEG image** and saving **40 augmented versions** of it. 
My code also extends the idea further by including a **bulk PyTorch script** that applies augmentation to every valid image in a folder.

## What I Learned
- The concept of image augmentation
- Augmentation techniques used in practice
- Implementation in differnt frameworks
- Resulting synthetic image diversity

The project demonstrates how common augmentation operations can be combined into reusable pipelines that generate multiple transformed outputs from one or more source images.

## Why Data Augmentation Matters

Data augmentation creates new image variations from existing data while preserving the original class meaning. This is useful when:

- collecting more labeled images is expensive or impractical
- the dataset is too small
- the model needs to become more robust to viewpoint, framing, or lighting changes
- we want to reduce over-reliance on narrow visual patterns

The PDF presents augmentation as a way to increase both **dataset size** and **dataset variability**, which is the main concept of the tutorial.

## Augmentation Techniques Used

The tutorial and code use several common transformations.

### 1. Rotation

Rotation changes the image orientation within a chosen range. This helps simulate viewpoint changes and slight camera tilt.
**<figure><figcaption>&emsp;Original Image&emsp;vs.&emsp;Rotated Image</figcaption><br><img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/4f64b24a-b031-4c4e-9e8b-017f2177d2a1" />  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/e0e90c1d-05cf-433a-b569-1348d7247f12" /></figure>**

### 2. Shear / Affine Transformation

The tutorial includes **shear** as a geometric distortion in Keras. In the PyTorch version, this idea is extended with `RandomAffine`, which adds translation, scaling, and shear in one pipeline. This makes the PyTorch implementation slightly more flexible than the baseline tutorial code.

**<figure><figcaption>&emsp;Original Image&emsp;vs.&emsp;Sheared Image</figcaption><br><img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/37492b6d-31ef-4b85-aab3-09efc5c5d2c6" />  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/3e8252e2-b456-4475-9dde-c069c6e1cfe5" /></figure>**

### 3. Zoom / Resized Cropping

The PDF includes **zoom** as one of the standard augmentations. In the PyTorch scripts, this is handled through `RandomResizedCrop`, which changes crop region and scale while preserving the output size.

**<figure><figcaption>&emsp;Original Image&emsp;vs.&emsp;Zoomed Image</figcaption><br><img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/37492b6d-31ef-4b85-aab3-09efc5c5d2c6" />  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/ef69c324-644c-4435-b3a0-43bd516065df" /></figure>**

### 4. Horizontal Flip

Horizontal flipping mirrors the image and increases directional variation in the dataset. This is included in both the tutorial and the implementations. 
**<figure><figcaption>&emsp;Original Image&emsp;vs.&emsp;Flipped Image</figcaption><br><img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/37492b6d-31ef-4b85-aab3-09efc5c5d2c6" />  <img width="400" height="300" alt="image_2_08" src="https://github.com/user-attachments/assets/d6608537-5bf9-4da3-a661-3b6b95cddba0" /></figure>**

### 5. Brightness Adjustment

Brightness changes simulate different lighting conditions. The PDF explicitly includes brightness augmentation, and the Keras implementation uses `brightness_range` for this purpose.

**<figure><figcaption>&emsp;Original Image&emsp;vs.&emsp;Dimmed Image</figcaption><br><img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/37492b6d-31ef-4b85-aab3-09efc5c5d2c6" />  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/1d6f6405-00b9-49aa-a54a-f08578e75a8d" /></figure>**

### 6. Contrast and Saturation Adjustment

This goes beyond the original Keras tutorial implementation. The PyTorch scripts use `ColorJitter` to vary **contrast** and **saturation** in addition to brightness, which broadens appearance diversity and makes the augmentation pipeline more expressive.

**<figure><figcaption>&emsp;Original Image&emsp;vs.&emsp;High Contrast Image</figcaption><br><img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/37492b6d-31ef-4b85-aab3-09efc5c5d2c6" />  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/b6d0657e-d05c-439e-81f3-b9f013e98a16" /></figure>**

## Tutorial Workflow

The PDF walks through a simple augmentation workflow:

1. import necessary libraries  
2. define the output folder  
3. initialize the augmenter with chosen parameters  
4. load the image and convert it to an array  
5. reshape it for batch processing  
6. generate and save augmented images  
7. confirm the folder location where images were saved

The task in the PDF is to take **one JPEG image** from the computer and generate **40 augmented images** saved in a folder.

## Keras Implementation

The Keras script follows the tutorial very closely. It uses `ImageDataGenerator` with tutorial-style augmentation settings such as:

- rotation
- shear
- zoom
- horizontal flip
- brightness change

The script then:

- loads one input image
- converts it to an array
- reshapes it into batch form
- generates **40 augmented images**
- saves them into the `augmented_images` folder

This version is the most tutorial-aligned and is good for understanding the basic educational workflow.

**Stub: Insert screenshot of Keras output folder or saved images here**

## PyTorch Implementation

I also implemented the same idea in **PyTorch / torchvision**. The single-image PyTorch script creates a transform pipeline with:

- random rotation
- random affine transformation
- random resized crop
- horizontal flip
- color jitter for brightness, contrast, and saturation

It loads one image, applies the stochastic transform repeatedly, and saves **40 augmented outputs** to the output folder.{index=19}

Compared with Keras, this version gives finer control over how augmentations are composed.



## Bulk PyTorch Extension

Beyond the original tutorial, I created a **bulk augmentation** script in PyTorch. This version:

- takes a folder path from the user
- scans for valid image files
- safely ignores unreadable or unsupported files
- generates **40 augmented images per source image**
- saves everything into the output directory

This makes the project more useful for actual dataset expansion, since it scales the tutorial idea from one image to many.


## Key Takeaways

This tutorial helped me understand that data augmentation is not just a preprocessing trick, but an important part of building stronger deep learning pipelines. The main things I learned were:

- data augmentation helps when training data is limited
- a single image can produce many useful training variations
- common operations include rotation, shear, zoom, flip, and brightness adjustment
- Keras provides a very accessible augmentation workflow
- PyTorch provides more composable and flexible augmentation control
- extending augmentation from one image to a whole folder makes it much more practical

Overall, this tutorial gave me a clear understanding of both the **concept** of augmentation and its **practical implementation** across different frameworks.
