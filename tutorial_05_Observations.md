# Tutorial 5 — Convolutional Neural Network (CNN)

## Overview

This tutorial introduced **Convolutional Neural Networks (CNNs)** for image classification using the **CIFAR-10** dataset.
The tutorial covers the full workflow: loading and preprocessing image data, visualizing samples, building a CNN, compiling and training the model, evaluating performance, plotting learning curves, and making predictions on test images.

In my implementation, I followed the tutorial’s baseline CNN and then extended it with architecture experiments, an accuracy-improvement section, and an overfitting-reduction section to align with the tutorial tasks of comparing architectures, improving performance, and addressing overfitting or underfitting.
## What I Learned

From this tutorial, I learned:

- what a **CNN** is and why it is useful for image classification
- how to load and normalize the **CIFAR-10** dataset
- how convolution and max-pooling layers help extract image features
- how to train and evaluate a CNN using TensorFlow and Keras
- how to use training and validation curves to judge learning behavior
- how to make and visualize predictions on unseen test images
- how architecture changes, augmentation, and regularization affect performance

## Dataset and Preprocessing

The tutorial used the **CIFAR-10** dataset, which contains small color images belonging to 10 classes such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
The first preprocessing step was to normalize pixel values from the range `[0, 255]` to `[0, 1]`, which helps the model train more smoothly and converge faster.

This was an important lesson because even before changing the architecture, proper preprocessing already affects how well the model learns.

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/6d52fd67-c586-4d09-a15e-88146797f037" />


## CNN Concept

A CNN works by stacking **convolutional layers** and **max-pooling layers** before passing the extracted features to fully connected layers for classification.
Convolution layers learn spatial patterns such as edges, textures, and shapes, while pooling layers reduce spatial size and help retain important features.

Therefore, CNNs are better suited for images than simple fully connected networks because they preserve spatial structure and learn local visual features.

## Baseline Model

The baseline CNN in the tutorial consists of:

- a convolution layer with 32 filters
- max-pooling
- a convolution layer with 64 filters
- max-pooling
- another convolution layer with 64 filters
- flatten layer
- a dense hidden layer
- a softmax output layer for 10 classes

The model was compiled using:

- **Adam** optimizer
- **sparse categorical crossentropy** loss
- **accuracy** metric 

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/869d5f88-99c1-415d-b256-8d07e37aafb9" />


## Training and Evaluation

The model was trained for **10 epochs** using the training data and validated on the test data after each epoch. After training, the model was evaluated on the test set to measure how well it generalizes to unseen images.
This part showed me the standard deep learning workflow:

- train the model
- monitor validation performance
- evaluate final test accuracy
- inspect learning curves

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/93e43d42-467d-4c58-a826-712dd78c480d" />
<img width="1000" height="175" alt="image" src="https://github.com/user-attachments/assets/385ca973-e4cd-414f-a7ee-5d0cec2b4bf4" />


## Learning Curves

A major part of the tutorial was plotting:

- training accuracy
- validation accuracy
- training loss
- validation loss

they help detect whether the model is learning correctly, and they can reveal **overfitting** or **underfitting**. For example:

- if training accuracy keeps improving while validation accuracy stalls or drops, the model may be **overfitting**
- a persistent gap between training and validation curves may indicate **poor generalization**
- weak performance on both training and validation may indicate **underfitting**

<img width="1000" height="450" alt="image" src="https://github.com/user-attachments/assets/f34603b7-1773-45e9-858e-018815915639" />


## Predictions and Visualization

After training, the model was used to make predictions on test images. 
<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/b2b0c4b1-3312-42ee-86bc-39b2328d6ede" />

## Task 01: Architecture Experiments
- modifying filter size
- adding more convolution layers
- changing the number of filters
- adding or modifying fully connected layers
- comparing the modified model with the original model

In the notebook, I extended the baseline with several variants:

- **a stronger CNN with more convolution blocks and a larger dense layer**
<img width="1000" height="550" alt="image" src="https://github.com/user-attachments/assets/17d5172b-9a95-40e4-a175-7e6b43e79a1d" />

<img width="1000" height="55" alt="image" src="https://github.com/user-attachments/assets/5e2c800a-088e-4c58-b6d6-0660e2177c0c" />

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/9da41fba-9f38-4b26-bacf-4a94736cb590" />

- **a version with **Batch Normalization** for more stable learning**
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/d26cc9b5-ca48-47f8-8c9a-22247d2a160d" />
<img width="1000" height="55" alt="image" src="https://github.com/user-attachments/assets/1c748884-6e07-4042-8fe0-300a47271afd" />

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/b8dfeb63-dd49-4a5c-ab16-3444dc18b572" />


- **a version using **Global Average Pooling** instead of flattening to reduce parameter count**
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/8a216995-ff72-4ff2-8306-e25dc1a2a26c" />
<img width="1000" height="55" alt="image" src="https://github.com/user-attachments/assets/590c5033-86a1-45fc-a74b-0a554ad04c21" />
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/7dd0dc1f-e0b6-438a-8d5e-986a1ab806a2" />


These experiments helped show how architecture choices affect both accuracy and generalization.

## Task 02: Improving Accuracy

Another tutorial task was to improve the model’s accuracy.

In the implementation, this was addressed by adding:

- **data augmentation**
- a stronger optimizer configuration
- a more capable network structure

<img width="1000" height="1250" alt="image" src="https://github.com/user-attachments/assets/ba375edb-ffd7-4867-bc64-8a94a4cdf0b6" />
<img width="1000" height="55" alt="image" src="https://github.com/user-attachments/assets/5cf29829-2788-452f-a942-2e4b3631e46e" />
<img width="1000" height="450" alt="image" src="https://github.com/user-attachments/assets/9fd66ae0-a76a-4893-ba6f-5ce27885da0c" />
<img width="1573" height="550" alt="image" src="https://github.com/user-attachments/assets/d2df3656-4737-4db1-b2f7-6a2e4a5d07e2" />


This section taught me that model improvement is often not about one single trick, but about combining better data handling, better architecture, and better training setup.

## Task 03: Removing Overfitting / Underfitting

- **L2 regularization**
- **Dropout**
- stronger architecture control

Model-building is not just about making a network deeper — it is also about controlling generalization.<br>
<br><img width="1000" height="50" alt="image" src="https://github.com/user-attachments/assets/b314125e-8d24-4a3a-82f2-a8d7e2a2832b" />
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/b415ae6d-7594-4c8b-9a84-353cfe9acfb4" />



## Results and Reflection

The best results came from Variant B of Task 1 & Task 02.

## Key Takeaways

This tutorial helped me understand the practical workflow of CNN-based image classification. The main things I learned were:

- CNNs are well suited for image data because they learn spatial features
- normalization is an important preprocessing step
- training and validation curves are essential for diagnosing model behavior
- architecture changes can improve or worsen performance
- dropout and L2 regularization help reduce overfitting
- prediction visualizations make model behavior easier to interpret
