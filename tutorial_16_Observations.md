# Tutorial 16 — Generative Adversarial Networks (GANs)

## Overview

This tutorial focused on **Generative Adversarial Networks (GANs)** using the MNIST handwritten digit dataset.

The tutorial objectives were:

* understand the architecture of GANs
* implement a GAN
* visualize generated results

A GAN contains two neural networks:

* **Generator:** takes random noise as input and generates fake images
* **Discriminator:** receives real or fake images and predicts whether they are real

The two networks are trained adversarially. The discriminator learns to distinguish real MNIST images from generated images, while the generator learns to fool the discriminator.

## Dataset

The tutorial uses the **MNIST** dataset.

MNIST images are grayscale images of handwritten digits with shape `1 × 28 × 28`.

The images are normalized to the range `[-1, 1]` because the generator output uses `Tanh`.

## Notebook Structure

| Section | Description |
|---|---|
| Cell 1 | Code copied from the tutorial screenshots |
| Cell 2 | Clean PyTorch implementation of the same GAN |
| Task 1 | Change epochs, layers, and architecture depth |
| Task 2 | Replace fully connected layers with convolutional layers |
| Task 3 | Train a GAN on augmented MNIST images |
| Final Cell | Compare loss curves and generated samples |

## Cell 1 — Tutorial Screenshot Code

The first cell contains the code from the tutorial PDF screenshots.

It includes:

* importing PyTorch libraries
* loading MNIST
* defining the Generator
* defining the Discriminator
* defining BCE loss and Adam optimizers
* training the GAN
* visualizing generated images

## Cell 2 — PyTorch Implementation

The second cell implements the same GAN idea in PyTorch with cleaner organization.

The model remains a simple fully connected GAN:

`noise vector → Generator → generated 28×28 image`

and:

`real/fake image → Discriminator → probability of being real`

The training loop alternates between:

1. training the discriminator on real and fake images
2. training the generator to fool the discriminator

## Task 1 — Change Epochs, Layers, etc.

Task 1 uses a configurable MLP GAN.

The number of layers can be changed by editing:

```python
generator_hidden_dims = (256, 512, 1024)
discriminator_hidden_dims = (1024, 512, 256)
```

The number of epochs can be changed using:

```python
TASK1_EPOCHS = 10
```

This satisfies the task of changing epochs, layers, and related settings.

## Task 2 — Replace Fully Connected Layers with Convolutional Layers

Task 2 implements a DCGAN-style model.

The generator uses:

* `ConvTranspose2d`
* `BatchNorm2d`
* `ReLU`
* `Tanh`

The discriminator uses:

* `Conv2d`
* `BatchNorm2d`
* `LeakyReLU`
* `Sigmoid`

This replaces the fully connected image-generation architecture with a convolutional architecture.

## Task 3 — GAN for Augmented Images

Task 3 trains a DCGAN-style model on augmented MNIST images.

The augmentation includes:

* random rotation
* random translation
* random scaling

This addresses the task of developing the model for augmented images.

The augmented dataset still uses only the MNIST training split, so no test images are mixed into training.

## Guardrails

The notebook uses the following guardrails:

* MNIST training data is used for GAN training
* test data is not mixed into GAN training
* generated images are visualized after converting from `[-1, 1]` display range to `[0, 1]`
* generator output uses `Tanh`, matching normalized image data
* discriminator output uses `Sigmoid`, matching binary cross-entropy loss
* loss curves are shown but not treated as a complete image-quality metric
* visual inspection is used along with generator/discriminator losses

## Important Notes

GAN losses are not interpreted the same way as ordinary supervised learning losses.

A lower generator loss does not always mean better images. A lower discriminator loss can sometimes mean the discriminator is overpowering the generator.

Therefore, generated image samples should always be inspected together with the loss curves.

## Key Takeaways

* GANs contain a generator and a discriminator
* the generator learns to create fake images from random noise
* the discriminator learns to classify real versus fake images
* fully connected GANs can generate MNIST-like images
* convolutional GANs are more suitable for image generation
* data augmentation can change the target image distribution learned by the GAN
* GAN training is unstable and should be judged using both losses and image samples

## Final Result

The notebook implements the tutorial GAN and completes all PDF tasks:

| Task | Status |
|---|---|
| Change number of epochs/layers | Completed |
| Replace fully connected layers with convolutional layers | Completed |
| Develop model for augmented images | Completed |
