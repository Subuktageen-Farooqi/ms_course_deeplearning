# Tutorial 4 — Data Augmentation

## Overview

This tutorial introduced **data augmentation** as a practical way to increase dataset size and variability when collecting large numbers of images is difficult. 
Augmentation is useful in deep learning because models generally perform better with more diverse data, and augmentation helps create that diversity from existing images.

I implemented augmentation pipelines in both **Keras** and **PyTorch**. 
The original tutorial focused on augmenting a **single JPEG image** and saving **40 augmented versions** of it. 
My code also extends the idea further by including a **bulk PyTorch script** that applies augmentation to every valid image in a folder.

## What I Learned
- why data augmentation is important in deep learning
- how augmentation helps when datasets are small
- how to apply common augmentation operations to images
- how to use **Keras `ImageDataGenerator`** for single-image augmentation
- how to build a more flexible augmentation pipeline in **PyTorch / torchvision**
- how to save generated images into an output folder for inspection and reuse

## Why Data Augmentation Matters

Data augmentation creates new image variations from existing data while preserving the original class meaning. This is useful when:

- collecting more labeled images is expensive or impractical
- the dataset is too small
- the model needs to become more robust to viewpoint, framing, or lighting changes
- we want to reduce over-reliance on narrow visual patterns

The PDF presents augmentation as a way to increase both **dataset size** and **dataset variability**, which is the main concept of the tutorial. :contentReference[oaicite:3]{index=3}

## Augmentation Techniques Used

The tutorial and code use several common transformations.

### 1. Rotation

Rotation changes the image orientation within a chosen range. This helps simulate viewpoint changes and slight camera tilt. The PDF lists rotation as one of the main augmentation operations configured in `ImageDataGenerator`. :contentReference[oaicite:4]{index=4}

**Stub: Insert original vs rotated image here**

### 2. Shear / Affine Transformation

The tutorial includes **shear** as a geometric distortion in Keras. In the PyTorch version, this idea is extended with `RandomAffine`, which adds translation, scaling, and shear in one pipeline. This makes the PyTorch implementation slightly more flexible than the baseline tutorial code. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

**Stub: Insert original vs sheared / affine-transformed image here**

### 3. Zoom / Resized Cropping

The PDF includes **zoom** as one of the standard augmentations. In the PyTorch scripts, this is handled through `RandomResizedCrop`, which changes crop region and scale while preserving the output size. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

**Stub: Insert original vs zoomed image here**

### 4. Horizontal Flip

Horizontal flipping mirrors the image and increases directional variation in the dataset. This is included in both the tutorial and the implementations. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

**Stub: Insert original vs flipped image here**

### 5. Brightness Adjustment

Brightness changes simulate different lighting conditions. The PDF explicitly includes brightness augmentation, and the Keras implementation uses `brightness_range` for this purpose. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

**Stub: Insert original vs brightness-adjusted image here**

### 6. Contrast and Saturation Adjustment

This goes beyond the original Keras tutorial implementation. The PyTorch scripts use `ColorJitter` to vary **contrast** and **saturation** in addition to brightness, which broadens appearance diversity and makes the augmentation pipeline more expressive. :contentReference[oaicite:13]{index=13}

**Stub: Insert original vs contrast/saturation-adjusted image here**

## Tutorial Workflow

The PDF walks through a simple augmentation workflow:

1. import necessary libraries  
2. define the output folder  
3. initialize the augmenter with chosen parameters  
4. load the image and convert it to an array  
5. reshape it for batch processing  
6. generate and save augmented images  
7. confirm the folder location where images were saved :contentReference[oaicite:14]{index=14}

The final task in the PDF is to take **one JPEG image** from the computer and generate **40 augmented images** saved in a folder. :contentReference[oaicite:15]{index=15}

## Keras Implementation

The Keras script follows the tutorial very closely. It uses `ImageDataGenerator` with tutorial-style augmentation settings such as:

- rotation
- shear
- zoom
- horizontal flip
- brightness change :contentReference[oaicite:16]{index=16}

The script then:

- loads one input image
- converts it to an array
- reshapes it into batch form
- generates **40 augmented images**
- saves them into the `augmented_images` folder :contentReference[oaicite:17]{index=17}

This version is the most tutorial-aligned and is good for understanding the basic educational workflow.

**Stub: Insert screenshot of Keras output folder or saved images here**

## PyTorch Implementation

I also implemented the same idea in **PyTorch / torchvision**. The single-image PyTorch script creates a transform pipeline with:

- random rotation
- random affine transformation
- random resized crop
- horizontal flip
- color jitter for brightness, contrast, and saturation :contentReference[oaicite:18]{index=18}

It loads one image, applies the stochastic transform repeatedly, and saves **40 augmented outputs** to the output folder. :contentReference[oaicite:19]{index=19}

Compared with Keras, this version gives finer control over how augmentations are composed.

**Stub: Insert screenshot of PyTorch single-image outputs here**

## Bulk PyTorch Extension

Beyond the original tutorial, I created a **bulk augmentation** script in PyTorch. This version:

- takes a folder path from the user
- scans for valid image files
- safely ignores unreadable or unsupported files
- generates **40 augmented images per source image**
- saves everything into the output directory :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

This makes the project more useful for actual dataset expansion, since it scales the tutorial idea from one image to many.

## Results

The main result of this tutorial was the successful generation of multiple transformed versions of the original image. Instead of relying on only one source image, the augmentation pipeline created many visually varied images through rotation, zoom, flipping, cropping, brightness changes, and other transformations.

In the base task, the expected output was **40 augmented images from a single JPEG image**, which both the Keras and single-image PyTorch scripts achieve. :contentReference[oaicite:22]{index=22} :contentReference[oaicite:23]{index=23} :contentReference[oaicite:24]{index=24}

**Stub: Insert grid of sample augmented outputs here**

## Observations

Some useful observations from this tutorial and its implementations:

- the **Keras version** is simpler and follows the tutorial steps closely
- the **PyTorch version** offers more flexible augmentation design
- the **bulk PyTorch version** is the most practical for preparing a larger dataset
- augmentation can greatly increase dataset diversity without collecting new images
- the transformations should still preserve the semantic meaning of the image

## Key Takeaways

This tutorial helped me understand that data augmentation is not just a preprocessing trick, but an important part of building stronger deep learning pipelines. The main things I learned were:

- data augmentation helps when training data is limited
- a single image can produce many useful training variations
- common operations include rotation, shear, zoom, flip, and brightness adjustment
- Keras provides a very accessible augmentation workflow
- PyTorch provides more composable and flexible augmentation control
- extending augmentation from one image to a whole folder makes it much more practical

Overall, this tutorial gave me a clear understanding of both the **concept** of augmentation and its **practical implementation** across different frameworks.
