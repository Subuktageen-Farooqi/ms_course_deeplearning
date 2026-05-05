# Tutorial 9 — Object Detection using SSD and YOLO

This tutorial focused on **object detection using SSD and YOLO**. The main building blocks of object detection models:

* a **backbone network** for feature extraction
* a **detection head** for class prediction and bounding-box prediction
* multi-scale feature maps for SSD
* grid-based prediction for YOLO
* dummy input testing to confirm model output shapes

## Notebook Review

The PyTorch cell is mostly equivalent of the TensorFlow code:

* the SSD backbone uses `Conv2d → ReLU → MaxPool2d` blocks matching the TensorFlow backbone
* the SSD head has separate class and box convolutions, matching the TensorFlow class and box heads
* the SSD model creates three feature scales and concatenates predictions
* the YOLO backbone follows the same five convolution/max-pooling stages as the TensorFlow YOLO backbone
* the YOLO head uses a single `1×1` convolution with `num_boxes × (5 + num_classes)` output channels

However, the PyTorch cell has some execution errors that must be corrected before using it:

* `transform` is used but not defined
* `ssd_loader` is used but not defined
* `yolo_loader` is used but not defined
* the same dataset object cannot directly serve both SSD and YOLO because SSD expects `300×300` inputs while YOLO expects `416×416` inputs
* the custom SSD/YOLO models only produce raw predictions; they do not include anchor matching, detection loss, NMS, or a full training pipeline

Because of this, the custom SSD and YOLO models should be tested by forwarding real object-detection images through them and checking prediction shapes. Full training is better handled using pretrained detection libraries/models in Task 3.

## Dataset

I used the **Penn-Fudan Pedestrian Dataset**. This is an appropriate public dataset because:

* it contains pedestrian object annotations
* it provides segmentation masks from which bounding boxes can be derived
* it can be used for object detection with one foreground class: `person`
* it is widely used in PyTorch object-detection tutorials

The class setup is:

`background: 0`
`person : 0`

</div>

For YOLO label files, the foreground class is written as class `0` because YOLO label files usually do not include a background class.

## TensorFlow SSD Model Summary

**Model:** `functional_4`

<div align="center">

| Layer              | Type         |           Output Shape | Param # | Connected To                             |
| ------------------ | ------------ | ---------------------: | ------: | ---------------------------------------- |
| `input_layer_4`    | InputLayer   |  `(None, 300, 300, 3)` |       0 | `-`                                      |
| `conv2d_36`        | Conv2D       | `(None, 300, 300, 32)` |     896 | `input_layer_4`                          |
| `max_pooling2d_14` | MaxPooling2D | `(None, 150, 150, 32)` |       0 | `conv2d_36`                              |
| `conv2d_37`        | Conv2D       | `(None, 150, 150, 64)` |  18,496 | `max_pooling2d_14`                       |
| `max_pooling2d_15` | MaxPooling2D |   `(None, 75, 75, 64)` |       0 | `conv2d_37`                              |
| `conv2d_38`        | Conv2D       |  `(None, 75, 75, 128)` |  73,856 | `max_pooling2d_15`                       |
| `conv2d_41`        | Conv2D       |  `(None, 38, 38, 128)` | 147,584 | `conv2d_38`                              |
| `conv2d_44`        | Conv2D       |  `(None, 19, 19, 128)` | 147,584 | `conv2d_41`                              |
| `conv2d_39`        | Conv2D       |  `(None, 75, 75, 126)` | 145,278 | `conv2d_38`                              |
| `conv2d_42`        | Conv2D       |  `(None, 38, 38, 126)` | 145,278 | `conv2d_41`                              |
| `conv2d_45`        | Conv2D       |  `(None, 19, 19, 126)` | 145,278 | `conv2d_44`                              |
| `conv2d_40`        | Conv2D       |   `(None, 75, 75, 24)` |  27,672 | `conv2d_38`                              |
| `conv2d_43`        | Conv2D       |   `(None, 38, 38, 24)` |  27,672 | `conv2d_41`                              |
| `conv2d_46`        | Conv2D       |   `(None, 19, 19, 24)` |  27,672 | `conv2d_44`                              |
| `reshape_12`       | Reshape      |    `(None, 33750, 21)` |       0 | `conv2d_39`                              |
| `reshape_14`       | Reshape      |     `(None, 8664, 21)` |       0 | `conv2d_42`                              |
| `reshape_16`       | Reshape      |     `(None, 2166, 21)` |       0 | `conv2d_45`                              |
| `reshape_13`       | Reshape      |     `(None, 33750, 4)` |       0 | `conv2d_40`                              |
| `reshape_15`       | Reshape      |      `(None, 8664, 4)` |       0 | `conv2d_43`                              |
| `reshape_17`       | Reshape      |      `(None, 2166, 4)` |       0 | `conv2d_46`                              |
| `concatenate_4`    | Concatenate  |    `(None, 44580, 21)` |       0 | `reshape_12`, `reshape_14`, `reshape_16` |
| `concatenate_5`    | Concatenate  |     `(None, 44580, 4)` |       0 | `reshape_13`, `reshape_15`, `reshape_17` |

<table>
<tr>
<td valign="top">

### SSD Parameter Count

| Parameter Type       |   Count |
| -------------------- | ------: |
| Total params         | 907,266 |
| Trainable params     | 907,266 |
| Non-trainable params |       0 |
| Model size           | 3.46 MB |

</td>
<td valign="top">

### SSD Output Shapes

| Output               | Shape            |
| -------------------- | ---------------- |
| SSD class prediction | `(1, 44580, 21)` |
| SSD box prediction   | `(1, 44580, 4)`  |

</td>
</tr>
</table>

</div>

## TensorFlow YOLO Model Summary

**Model:** `functional_5`

<div align="center">

| Layer              | Type         |            Output Shape |   Param # |
| ------------------ | ------------ | ----------------------: | --------: |
| `input_layer_5`    | InputLayer   |   `(None, 416, 416, 3)` |         0 |
| `conv2d_48`        | Conv2D       |  `(None, 416, 416, 32)` |       896 |
| `max_pooling2d_16` | MaxPooling2D |  `(None, 208, 208, 32)` |         0 |
| `conv2d_49`        | Conv2D       |  `(None, 208, 208, 64)` |    18,496 |
| `max_pooling2d_17` | MaxPooling2D |  `(None, 104, 104, 64)` |         0 |
| `conv2d_50`        | Conv2D       | `(None, 104, 104, 128)` |    73,856 |
| `max_pooling2d_18` | MaxPooling2D |   `(None, 52, 52, 128)` |         0 |
| `conv2d_51`        | Conv2D       |   `(None, 52, 52, 256)` |   295,168 |
| `max_pooling2d_19` | MaxPooling2D |   `(None, 26, 26, 256)` |         0 |
| `conv2d_52`        | Conv2D       |   `(None, 26, 26, 512)` | 1,180,160 |
| `max_pooling2d_20` | MaxPooling2D |   `(None, 13, 13, 512)` |         0 |
| `conv2d_53`        | Conv2D       |    `(None, 13, 13, 75)` |    38,475 |

<table>
<tr>
<td valign="top">

### YOLO Parameter Count

| Parameter Type       |     Count |
| -------------------- | --------: |
| Total params         | 1,607,051 |
| Trainable params     | 1,607,051 |
| Non-trainable params |         0 |
| Model size           |   6.13 MB |

</td>
<td valign="top">

### YOLO Output Shape

| Output          | Shape             |
| --------------- | ----------------- |
| YOLO prediction | `(1, 13, 13, 75)` |

</td>
</tr>
</table>

</div>

### Note

For the YOLO output, the final channel count is `75`, which comes from:

`3 × (5 + 20) = 75`

where `3` is the number of bounding boxes per grid cell, `5` represents bounding-box coordinates plus confidence, and `20` is the number of classes.


## Task 01 - PyTorch Implementation


<div align="center">

<table>
<tr>
<td valign="top">
  
### SSD Training Summary

| Epoch | Training Loss | Validation mAP50 |
| ----: | ------------: | ---------------: |
|  1/25 |        0.6905 |           0.0502 |
|   ... |           ... |              ... |
| 21/25 |        0.5795 |           0.1798 |
| 22/25 |        0.5808 |           0.2309 |
| 23/25 |        0.5803 |           0.2410 |
| 24/25 |        0.5810 |           0.2035 |
| 25/25 |        0.5804 |           0.1242 |

</td>
<td valign="top">

### YOLO Training Summary

| Epoch | Training Loss | Validation mAP50 |
| ----: | ------------: | ---------------: |
|  1/25 |        0.3168 |           0.0000 |
|   ... |           ... |              ... |
| 21/25 |        0.0546 |           0.0672 |
| 22/25 |        0.0512 |           0.0619 |
| 23/25 |        0.0478 |           0.0446 |
| 24/25 |        0.0491 |           0.0468 |
| 25/25 |        0.0445 |           0.0272 |

</td>
</tr>
</table>

### Test Metrics

| Model         | Test mAP50 | Test mAP50-95 |
| ------------- | ---------: | ------------: |
| Tutorial SSD  |   0.075757 |      0.016375 |
| Tutorial YOLO |   0.010197 |      0.001593 |

</div>



## Task 02 - Custom SSD/YOLO Model Testing

Reuse the existing custom SSD and YOLO PyTorch models from the notebook.

The purpose of this task is not full object-detection training. The models are architecture demonstrations, not complete detection training systems.

The Task checks whether real dataset images can pass through the custom models and produce valid output shapes.

Expected custom SSD output:

* class prediction tensor: `[batch, total_boxes, num_classes]`
* box prediction tensor: `[batch, total_boxes, 4]`

Expected custom YOLO output:

* prediction tensor: `[batch, num_boxes × (5 + num_classes), grid_h, grid_w]`

For the tutorial YOLO setup:

* `num_boxes = 3`
* `num_classes = 20`
* output channels = `3 × (5 + 20) = 75`

So a valid YOLO output should have 75 channels.

## Task 3 — Pretrained SSD and YOLO

Task 3 uses one pretrained SSD-style model and one pretrained YOLO model.

For SSD, the code uses TorchVision's `ssdlite320_mobilenet_v3_large` with a pretrained MobileNetV3 backbone and adapts it for two classes:

* background
* person

For YOLO, the code uses Ultralytics YOLO, starting from `yolov8n.pt`, and converts the Penn-Fudan annotations into YOLO text-label format.

This is more appropriate than trying to manually train the custom SSD/YOLO architectures from scratch because pretrained detection models already include the practical components needed for training and inference.



## Expected Results

The custom SSD and YOLO models should successfully produce prediction tensors on real images.

The pretrained SSD model should train for a small number of epochs and produce detections for pedestrians.

The pretrained YOLO model should train on the converted YOLO-format dataset and report validation/test metrics through the Ultralytics training pipeline.

Because this is a small learning dataset and the training epochs are intentionally limited, the results should be interpreted as a tutorial demonstration rather than a final benchmark.

## Key Takeaways

* SSD uses multiple feature-map scales for detection
* SSD predicts class scores and box coordinates separately
* YOLO predicts bounding boxes, confidence, and class probabilities on a grid
* custom detection architectures can be shape-tested with dummy or real images
* full detection training requires more than just the model forward pass
* pretrained detection models are more practical for training on small datasets
* dataset formatting is different for TorchVision SSD-style models and YOLO models
* object detection evaluation must keep training, validation, and test data separate

Overall, the tutorial is useful because it introduces the internal structure of SSD and YOLO while also showing why pretrained detection models are usually preferred for practical training.
