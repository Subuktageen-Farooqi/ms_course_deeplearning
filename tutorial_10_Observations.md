# Tutorial 10 - Image Segmentation using U-Net

## Overview
This tutorial introduced **image segmentation using U-Net**. We used a synthetic rectangle-mask example as the base TensorFlow/Keras pipeline, while the task section asked to use labeled data, test the model, change layers, and compare different learning rates with visualized results.

For the practical PyTorch implementation, I used the **Oxford-IIIT Pet** dataset from `torchvision`, which officially supports `target_types="segmentation"` and provides `trainval` and `test` splits. This made it a good small public labeled dataset for a segmentation tutorial.

## What I Learned
- what **image segmentation** is and how it differs from classification
- how the **U-Net** encoder-decoder structure works
- how skip connections help recover spatial detail
- how to evaluate segmentation results visually and numerically
- how changing the **learning rate** affects convergence and segmentation quality

## U-Net Concept

U-Net is a segmentation model built from two main parts:

- an **encoder** that gradually extracts deeper features
- a **decoder** that upsamples features back to the original resolution

Its most important feature is the use of **skip connections**, where feature maps from the encoder are concatenated with decoder features. This helps the model preserve fine spatial details while still learning higher-level semantic information.

In the TensorFlow pipeline, the model used two encoder blocks, a bottleneck, and two decoder stages before producing a single-channel segmentation mask with a sigmoid output.


## TensorFlow Pipeline from the PDF

The TensorFlow/Keras part of the tutorial used a simple **synthetic dataset** made from randomly generated rectangles and masks. I chagned that to use the oxford dataset as well. The workflow was:

1. generate sample images and binary masks  
2. define a U-Net  
3. split into training and validation sets  
4. train the model  
5. evaluate on validation data  
6. visualize predicted masks  

<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/8a46e570-760a-4644-bb52-343a0b88aa7f" />
<img width="1000" height="300" alt="image" src="https://github.com/user-attachments/assets/e5ed7f86-3165-4ad8-b2e4-1bd6b20fc3a4" />


## PyTorch Implementation on a Real Dataset

For the PyTorch version, I used the **Oxford-IIIT Pet** segmentation dataset instead of a synthetic dataset. This dataset contains pet images with segmentation annotations and is directly available through `torchvision`. It supports segmentation masks through `target_types="segmentation"` and provides official `trainval` and `test` splits.


## Model and Training Setup

The PyTorch model was a compact custom **U-Net-style** architecture:

- two encoder blocks
- one bottleneck block
- two decoder blocks
- a final `1×1` convolution for binary mask output

The training setup used:

- **Adam** optimizer
- **BCEWithLogitsLoss**
- **Dice score** as the main segmentation-quality metric
- validation after every epoch
- final testing on the held-out test split

This made the PyTorch pipeline conceptually very similar to the TensorFlow tutorial, but applied to real segmentation data.

## Learning Rate Experiments

One of the required tasks in the PDF was to use **different learning rates and visualize the results**.

To address this, I trained the PyTorch U-Net with multiple learning rates:

`1e-2`, `1e-3` ,`1e-4`

For each run, I recorded:

`training loss`, `validation loss`, `training Dice score`, `validation Dice score`, `final test Dice score`

Then I plotted:

- validation loss vs epoch
- validation Dice vs epoch

This made it possible to compare optimization behavior clearly.
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/1c7187a9-c1e5-4047-913b-afadaa49b8b0" />
## Results and Interpretation

The learning-rate comparison showed how strongly optimization settings affect segmentation training:

- a **high learning rate** can converge quickly but may be unstable
- a **moderate learning rate** often gives the best balance of speed and stability
- a **low learning rate** may train more slowly but can sometimes behave more smoothly

The final comparison was done using both metric values and visual inspection of predicted masks. Looking at the predicted masks was important, because segmentation quality is easier to judge visually than from a single scalar metric alone.

<img width="1000" height="100" alt="image" src="https://github.com/user-attachments/assets/8df13056-724f-4212-9558-da6bf8dd2c86" />

<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/3cc34dee-44e9-473f-9262-aaf57f273794" />

## Visualization

The tutorial emphasized evaluating and visualizing segmentation results, and this was one of the most useful parts of the work. For a few validation samples, I displayed:

- the input image
- the true segmentation mask
- the predicted segmentation mask

This makes it easy to see whether the model captured object shape correctly and whether the predicted boundaries were accurate.
<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/a5c5ab05-91a0-494f-99a9-4ef52d014d07" />
<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/586e4238-8988-4782-9583-445faec9ff41" />
<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/7c05f073-e073-46f5-84d9-583cb1489306" />
## Key Takeaways
- U-Net is effective because it combines deep features with spatial skip connections
- segmentation requires pixel-level labels, not only image-level labels
- synthetic examples are useful for understanding the pipeline, but real labeled datasets are needed for meaningful testing
- learning rate has a strong effect on convergence quality
- segmentation should be judged both numerically and visually
