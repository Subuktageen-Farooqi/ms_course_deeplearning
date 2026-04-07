# Tutorial 3 — Building a Neural Network for MNIST Handwritten Digit Classification

## Overview

This tutorial focused on building and training an **Artificial Neural Network (ANN)** for **MNIST handwritten digit classification**.
The main objectives were to train and evaluate a neural network, preprocess the dataset, visualize the data, and understand how to detect and handle **overfitting** and **underfitting**.

## What I Learned

- how to load and preprocess the **MNIST** dataset
- why **normalization** is important for stable training
- why **one-hot encoding** is needed for multi-class classification
- how to build a neural network using the **Sequential API**
- how to compile, train, and evaluate a model
- how to visualize training and validation accuracy/loss
- how to recognize **overfitting**, **underfitting**, and a **well-fitted** model
- how architecture, optimizer, epochs, and regularization affect performance


## Dataset and Preprocessing

The dataset used in this tutorial was **MNIST**, which contains grayscale images of handwritten digits from **0 to 9**. Each image has shape **28 × 28**. 
The preprocessing pipeline had three key parts:

- loading the training and test sets
- normalizing pixel values from `0–255` to `0–1`
- converting labels to **one-hot encoded** vectors for multiclass classification

<img width="900" height="300" alt="image" src="https://github.com/user-attachments/assets/2334b7b7-3d92-4d60-a3b3-9b84f1532272" />


## Model Architecture

The tutorial built the network using the **Sequential API**. The baseline model consisted of:

- a **Flatten** layer to convert `28 × 28` images into a vector of 784 values
- a **Dense(128, ReLU)** hidden layer
- a **Dense(64, ReLU)** hidden layer
- a **Dense(10, Softmax)** output layer for digit classes `0–9`

Each layer has a clear role: flattening the image, learning intermediate features, and outputting class probabilities.

<img width="900" height="450" alt="image" src="https://github.com/user-attachments/assets/77dc3ab7-660a-45f2-b4c0-ff6f105aab6b" />


## Compiling and Training the Model

The model was compiled with:

- **Adam** optimizer
- **categorical cross-entropy** loss
- **accuracy** as the main metric
<img width="900" height="50" alt="image" src="https://github.com/user-attachments/assets/27272dfc-c443-4e9c-adc5-dafeaa182b1d" />

It was then trained using:

- **10 epochs**
- **batch size = 32**
- **validation split = 0.2** (allows the model’s generalization behavior to be monitored during training instead of only after training ended)
<img width="900" height="550" alt="image" src="https://github.com/user-attachments/assets/524cb85a-4d43-4f3c-bb46-52bc3bc43135" />


## Evaluation and Visualization

After training, the model was evaluated on the test set to measure how well it generalized to unseen images. Then the tutorial plotted:

- training accuracy
- validation accuracy
- training loss
- validation loss

<img width="863" height="150" alt="image" src="https://github.com/user-attachments/assets/49322483-4c65-44ec-96e9-b16472418d05" />


The tutorial then used the trained model to make predictions on the test set and visualized some sample digits along with their true and predicted labels. This made the results much easier to interpret.

<img width="900" height="525" alt="image" src="https://github.com/user-attachments/assets/b0e4d0f2-f88d-4319-89a0-4277e14a1020" />

<img width="900" height="50" alt="image" src="https://github.com/user-attachments/assets/6468c076-5b42-4212-8883-620d042db957" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/8cfa2427-173a-46f0-b0ab-25f88ed8d5b9" />


## Tasks


### Task 01: Re-implement the full tutorial in PyTorch


<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/1ba8bb3f-a0ca-4325-899c-f181c5b6a20a" />
<img width="900" height="50" alt="image" src="https://github.com/user-attachments/assets/690c11dc-f36e-4695-a399-b9585fa71fde" />
<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/83919193-6610-4d40-b557-081022ad5e7e" />
<img width="900" height="1000" alt="image" src="https://github.com/user-attachments/assets/b6246925-7d5b-448d-89dd-91ef03c8a70f" /><br>


|Experiment_no|section |name                  |architecture       |activation|optimizer|learning_rate|epochs_requested|epochs_run|dropout|l2_lambda|early_stopping_patience|final_train_acc|final_val_acc|test_acc|final_train_loss|final_val_loss|test_loss|fit_comment|
|-------------|--------|----------------------|-------------------|----------|---------|-------------|----------------|----------|-------|---------|-----------------------|---------------|-------------|--------|----------------|--------------|---------|-----------|
|1            |Baseline|PyTorch baseline (128, 64) + ReLU + Adam|(128, 64)          |relu      |adam     |0.001        |10              |10        |0.0    |0.0      |None                   |	0.9814        |0.9724       |0.9737  |0.0574          |0.0981        |0.0869   |Reasonably well-fitted: train and validation c...|


### Task 02: Experiment with different architectures
   - add more layers
   - change the number of neurons
   - try different activations like **tanh** or **sigmoid**
   - compare with the original model
   <br><br>So, I ran few experiments
   - Experiment 2: Smaller network (64)
   - Experiment 3: Original size (128, 64)
   - Experiment 4: Wider network (256, 128)
   - Experiment 5: Deeper network (256, 128, 64)
   - Experiment 6: Deeper network + tanh
   <img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/a14b35a7-3809-4584-a137-5d282349f070" />
   <img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/2cdd6a2c-c71c-498b-88d2-0cf4ac9ee952" />


### Task 03: Compare different optimizers
  - **SGD**
  - **RMSprop**
  - **Adam**
  - **compare speed and final accuracy**

Experiments:
   - Experiment 7: SGD baseline architecture
   - Experiment 8: RMSprop baseline architecture
   - Experiment 9: Adam baseline architecture

<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/e0392f6f-5301-4cc3-91a1-cfc6dba4e1b1" />
<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/cdd45b5c-ead8-4f29-9c3f-b4174dd7eca8" />

**Interpretation notes**
- Faster improvement in validation accuracy suggests quicker convergence
- Lower final validation loss with strong test accuracy usually indicates the more effective optimizer
- If one optimizer is unstable, its loss curve may oscillate more


### Task 04: Interpret the training/validation curves
   - identify overfitting, underfitting, or good fit
   - study the effect of changing epochs
   - implement **Early Stopping**
   - experiment with **Regularization Techniques**

Epoch Experiments:
   - Experiment 10: Too few epochs (3)
   - Experiment 11: Baseline epochs (10)
   - Experiment 12: More epochs (20)
Epoch Experiments:
   - Experiment 13: EarlyStopping only
   - Experiment 14: Dropout 0.3
   - Experiment 15: L2 regularization

<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/85263ef3-76e8-42d9-9abf-3e9c2510d10e" />
<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/88ea1577-d2ca-4206-abf8-37ea9b4ceadf" />


## Compiled Experiments' Results
|experiment_no|Experiment_Variable|Exp_Var_Value                            |architecture                            |activation    |optimizer|learning_rate|epochs_requested|epochs_run       |dropout|l2_lambda|early_stopping_patience|final_train_acc|final_val_acc|test_acc                                          |final_train_loss|final_val_loss       |test_loss                                         |fit_comment                                       |FIELD20                                           |FIELD21                                           |FIELD22                                           |
|-------------|-------------------|-----------------------------------------|-----------------|--------------|---------|-------------|----------------|-----------------|-------|---------|-----------------------|---------------|-------------|--------------------------------------------------|----------------|---------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
|	4           |	Architecture      |	Wider network (256, 128)                | (256, 128)    |relu         |adam            |0.001            |10     |10       |0.0                    |0.0000         |NaN          |0.9846                                            |0.9733          |0.9780               |0.0457                                            |0.0931                                            |0.0785                                            |Reasonably well-fitted: train and validation c... |                                                  |
|	12          |	Epochs            |	More epochs (20)                        |	(128, 64)     |	relu    |	adam        |	0.001          |	20              |	20    |	0.0     |	0.0000                |	NaN           |	0.9912      |	0.9711                                           |	0.9754         |	0.0264              |	0.1255                                           |	0.1013                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |
|	7           |	Optimizer         |	SGD baseline architecture               |	(128, 64)     |	relu    |	sgd         |	0.010          |	10              |	10    |	0.0     |	0.0000                |	NaN           |	0.9841      |	0.9697	0.9744                                    |	0.0496         |	0.1015              |	0.0895                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |                                                  |
|	1           |	Baseline          |	PyTorch baseline (128, 64) + ReLU + Adam|	(128, 64)     |	relu        |	adam           |	0.001           |	10	10 |	0.0     |	0.0000                |	NaN	0.9814    |	0.9724      |	0.9737                                           |	0.0574         |	0.0981              |	0.0869                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |                                                  |
|	3           |	Architecture      |	"Original size (128| 64)"               |	(128, 64)     |	relu        |	adam           |	0.001           |	10    |	10      |	0.0                   |	0.0000        |	NaN         |	0.9809                                           |	0.9714         |	0.9736              |	0.0592	0.1021	0.0931                             |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |                                                  |
|	15          |	Regularization    |	L2 regularization                       |	(128, 64)     |	relu    |	adam        |	0.001          |	20              |	20    |	0.0     |	0.0001                |	NaN           |	0.9873      |	0.9707                                           |	0.9732         |	0.0382	0.1058	0.0985|	Reasonably well-fitted: train and validation c...|                                                  |                                                  |                                                  |                                                  |
|	5           |	Architecture      |	Deeper network (256, 128, 64)           | (256, 128, 64)| relu        |	adam            |0.001  |	10      |	10                    |	0.0           |	0.0000      |	NaN                                              |	0.9834         |	0.9698              |	0.9720                                           |	0.0517                                           |	0.1103                                           |	0.0908                                           |	Reasonably well-fitted: train and validation c...|
|	9           |	Optimizer         |	Adam                                    | baseline      | architecture |	"(128   | 64)"        |	relu           |	adam            |	0.001 |	10      |	10                    |	0.0           |	0.0000      |	NaN                                              |	0.9829         |	0.9701              |	0.9718                                           |	0.0544                                           |	0.1097	0.1012                                    |	Reasonably well-fitted: train and validation c...|                                                  |
|	13          |	Regularization    |	EarlyStopping only                      |	(128 , 64)"         |	relu    |	adam        |	0.001          |	20              |	12    |	0.0     |	0.0000                |	3.0           |	0.9834      |	0.9698                                           |	0.9710         |	0.0492              |	0.1174                                           |	0.0975                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |
|	8           |	Optimizer         |	RMSprop      | baseline                               | architecture |	"(128   | 64)"        |	relu           |	rmsprop	0.001	10|	10    |	0.0     |	0.0000                |	NaN           |	0.9795      |	0.9674                                           |	0.9699         |	0.0620              |	0.1156	0.1037                                    |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |                                                  |
|	11          |	Epochs            |	Baseline     | epochs                                 | (10)	(128    | 64)	relu|	adam        |	0.001          |	10              |	10    |	0.0     |	0.0000                |	NaN           |	0.9816      |	0.9687                                           |	0.9697         |	0.0582              |	0.1170                                           |	0.0996                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |
|	14          |	Regularization    |	Dropout      | 0.3	(128                               | 64)	relu     |	adam    |	0.001       |	20             |	20              |	0.3   |	0.0000  |	NaN                   |	0.9523        |	0.9649      |	0.9677                                           |	0.1547         |	0.1271              |	0.1123                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |                                                  |
|	6           |	Architecture      |	Deeper network + tanh|	"(256                                  | 128          | 64)"    |	tanh        |	adam           |	0.001           |	10    |	10      |	0.0                   |	0.0000        |	NaN         |	0.9739                                           |	0.9613         |	0.9620              |	0.0828                                           |	0.1350                                           |	0.1241                                           |	Reasonably well-fitted: train and validation c...|                                                  |
|	2           |	Architecture      |	Smaller network (64)|	(64                                    |)	relu	adam	0.001	10	10|	0.0     |	0.0000      |	NaN            |	0.9700          |	0.9547|	0.9602  |	0.0981                |	0.1540        |	0.1376      |	Reasonably well-fitted: train and validation c...|                |                     |                                                  |                                                  |                                                  |                                                  |                                                  |
|	10          |	Epochs            |	Too few epochs (3)|	"(128                                  | 64)"         |	relu    |	adam        |	0.001          |	3               |	3     |	0.0     |	0.0000                |	NaN           |	0.9538      |	0.9566                                           |	0.9601         |	0.1517              |	0.1462                                           |	0.1321                                           |	Reasonably well-fitted: train and validation c...|                                                  |                                                  |

## Overfitting and Underfitting

A major lesson in this tutorial was how to read training curves:

- **Overfitting:** training accuracy becomes much higher than validation accuracy, while validation loss stagnates or increases
- **Underfitting:** both training and validation accuracy remain low, and training loss stays high

This was very useful because it gave me a concrete way to judge whether a model is learning well or just memorizing the training data. After improvement, the model curves should look smoother and better balanced.


## Key Takeaways
- preprocessing steps like normalization and one-hot encoding are essential
- an ANN for MNIST can be built with simple dense layers and still perform well
- training and validation curves are critical for diagnosing model behavior
- optimizer choice, architecture, and number of epochs all affect convergence
- early stopping and regularization are practical tools to improve generalization
- understanding cross-entropy makes the training objective much clearer
