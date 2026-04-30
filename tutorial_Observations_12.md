# Tutorial 12 — Basic Autoencoders

## Overview

This tutorial focused on **basic autoencoders** using the MNIST handwritten digit dataset. The PDF introduced the core idea of an autoencoder: a neural network that learns to compress an input image into a smaller representation and then reconstruct the original image from that compressed representation.

For the notebook implementation, I kept the tutorial screenshot code separately as a direct TensorFlow implementation and then built a PyTorch equivalent of the same model. After that, I continued in PyTorch to address the task at the end of the PDF: improving the autoencoder model by experimenting with a different architecture.

The notebook was organized into three main parts:

* **Cell 1:** TensorFlow implementation copied from the tutorial screenshots
* **Cell 2:** PyTorch implementation of the same basic convolutional autoencoder
* **Cell 3 onward:** Improved PyTorch autoencoder for the tutorial task

## What I Learned

From this tutorial, I learned:

* what an **autoencoder** is
* how an encoder compresses image data into a smaller latent representation
* how a decoder reconstructs the image from that compressed representation
* how convolutional layers can be used for image reconstruction tasks
* how reconstruction loss is used instead of classification loss
* how to compare original and reconstructed images visually
* how to improve reconstruction quality by modifying the model architecture

## Autoencoder Concept

An autoencoder is a neural network trained to reproduce its input at the output. Unlike a classifier, it does not predict a class label. Instead, the target output is the same image that was given as input.

The model has two main parts:

1. **Encoder**

   The encoder compresses the input image into a smaller internal representation. In this tutorial, convolution and max-pooling layers were used to reduce the spatial size of the MNIST images.

2. **Decoder**

   The decoder reconstructs the image from the compressed representation. In the tutorial model, convolution and upsampling layers were used to bring the feature maps back to the original image size.

The goal is for the reconstructed image to be as close as possible to the original image.

## Dataset Used

The tutorial used the **MNIST** dataset. MNIST contains grayscale images of handwritten digits, where each image has a size of **28 × 28 pixels**.

Since the images are grayscale, each image has one channel. The images were normalized to the range **0 to 1** before training. This is important because the final layer of the autoencoder uses a sigmoid activation, which also outputs values between 0 and 1.

The dataset was used as follows:

* training input: `x_train`
* training target: `x_train`
* test input: `x_test`
* test target: `x_test`

This is correct for an autoencoder because the model learns to reconstruct the same image it receives.

## Cell 1 — TensorFlow Screenshot Code

The first notebook cell contains the TensorFlow implementation copied from the PDF screenshots. The code uses:

* `tensorflow`
* `tensorflow.keras.layers`
* `tensorflow.keras.models`
* `tensorflow.keras.datasets.mnist`
* `numpy`
* `matplotlib`

The TensorFlow model follows this structure:

### Encoder

The encoder uses:

* a `Conv2D` layer with 32 filters
* a `MaxPooling2D` layer
* a `Conv2D` layer with 64 filters
* another `MaxPooling2D` layer

This reduces the MNIST image into a smaller feature representation.

### Decoder

The decoder uses:

* a `Conv2D` layer with 64 filters
* an `UpSampling2D` layer
* a `Conv2D` layer with 32 filters
* another `UpSampling2D` layer
* a final `Conv2D` layer with 1 output channel and sigmoid activation

The final output has the same image shape as the input: **28 × 28 × 1**.

## Why Binary Cross-Entropy Was Used

The TensorFlow tutorial compiles the model using:

```python
loss='binary_crossentropy'
```

This is suitable here because the MNIST images are normalized between 0 and 1, and the model output also uses a sigmoid activation. The loss compares the reconstructed pixel values with the original pixel values.

For a basic MNIST reconstruction tutorial, binary cross-entropy is a reasonable reconstruction loss.

## Cell 2 — PyTorch Equivalent Implementation

The second part of the notebook contains a PyTorch version of the same basic autoencoder. The PyTorch version keeps the same overall idea as the TensorFlow model:

* input image shape: `1 × 28 × 28`
* encoder with convolution and max-pooling layers
* decoder with convolution and upsampling layers
* final sigmoid output
* reconstruction target equal to the input image

The PyTorch implementation uses:

* `torch`
* `torch.nn`
* `torch.optim`
* `torchvision.datasets.MNIST`
* `torchvision.transforms.ToTensor`
* `DataLoader`
* `matplotlib`

The training loop follows the standard PyTorch workflow:

1. set the model to training mode
2. send images to the selected device
3. run the forward pass
4. calculate reconstruction loss
5. backpropagate
6. update model weights

The target is not the digit label. The target is the input image itself.

## Guardrails Used in the Notebook

To avoid invalid results, I followed these guardrails:

* labels were not used for training because this is not a classification task
* training data was used only for training
* test data was kept separate for evaluation and visualization
* validation or test data was not mixed into the training loop
* model evaluation was done using `model.eval()`
* inference was done inside `torch.no_grad()`
* reconstruction targets were the original images, not labels
* outputs were kept in the same range as the normalized input images

These guardrails are important because an autoencoder can easily be implemented incorrectly if labels are accidentally used or if test images are included during training.

## Visualization of Results

After training, the notebook visualizes the reconstruction results by showing original and reconstructed images side by side.

The visualization uses two rows:

* top row: original MNIST images
* bottom row: reconstructed MNIST images

This makes it easy to judge whether the autoencoder has learned the general shape of each digit.

A good reconstruction should preserve:

* the digit shape
* the stroke position
* the overall structure
* the background as mostly black

A poor reconstruction would look blurry, distorted, or unrelated to the input image.

## Task — Improving the Model

The task at the end of the PDF asked to improve the autoencoder model by experimenting with different architectures to improve reconstruction quality.

To address this, I continued in PyTorch and created an improved convolutional autoencoder.

The improved model used:

* more convolutional layers
* more feature channels
* batch normalization
* a deeper encoder
* a deeper decoder

The purpose of the improved architecture was to give the model more capacity to learn useful image features before reconstructing the digit.

## Improved PyTorch Autoencoder

The improved model increased the encoder depth from two convolutional blocks to three convolutional blocks.

The encoder used:

* `Conv2d(1, 32)`
* `BatchNorm2d(32)`
* `ReLU`
* `MaxPool2d`
* `Conv2d(32, 64)`
* `BatchNorm2d(64)`
* `ReLU`
* `MaxPool2d`
* `Conv2d(64, 128)`
* `ReLU`
* `MaxPool2d`

The decoder then reconstructed the image using convolution and upsampling blocks.

This model is still simple enough for a learning tutorial, but it is more powerful than the basic model.

## Why the Improved Model Can Perform Better

The improved model can reconstruct better images because it has more feature extraction capacity. The additional convolutional layers allow the encoder to learn more complex local patterns in the digit images.

Batch normalization can also help stabilize training by normalizing intermediate feature activations. This can make training smoother and sometimes improve reconstruction quality.

However, the model was not made unnecessarily large because this is still an introductory MS deep learning tutorial. The goal was to improve the model without turning the assignment into an advanced research experiment.

## Basic Model vs Improved Model

The basic model is useful because it clearly demonstrates the autoencoder idea. It has a small encoder and decoder, making it easy to understand.

The improved model is useful because it shows how reconstruction quality can be improved by changing the architecture.

The comparison is:

| Model                | Purpose                          | Strength                                   |
| -------------------- | -------------------------------- | ------------------------------------------ |
| Basic Autoencoder    | Understand autoencoder structure | Simple and easy to follow                  |
| Improved Autoencoder | Address tutorial task            | Better feature learning and reconstruction |

## Expected Results

The basic autoencoder should reconstruct the general shape of MNIST digits, but some outputs may look blurry.

The improved autoencoder should produce cleaner reconstructions because it has more convolutional capacity.

The expected improvement is not that the model becomes perfect, but that the reconstructed digits should be more visually similar to the input digits.

## Key Takeaways

This tutorial helped me understand how autoencoders work in practice. The most important takeaways were:

* autoencoders learn reconstruction, not classification
* the input image is also the training target
* the encoder compresses the image into a smaller representation
* the decoder reconstructs the image from that representation
* convolutional autoencoders are suitable for image data
* reconstruction quality can be checked visually
* deeper architectures can improve reconstruction quality
* train and test data must remain separate even in reconstruction tasks

Overall, this tutorial was useful because it introduced the full autoencoder workflow: loading MNIST, building an encoder-decoder model, training with reconstruction loss, visualizing reconstructed images, and improving the model architecture.
