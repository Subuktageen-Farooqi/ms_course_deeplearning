# Tutorial 1 — Simple Perceptron

## Overview

This tutorial introduced the **Perceptron** as the precursor to Neural Networks. I implemented and trained the perceptron, and observed how it makes decisions using a weighted sum and an activation function.

## What I Learned

- how a perceptron uses **weights** and **bias**
- how to compute a **weighted sum**
- how the **activation function** converts that sum into a prediction
- how training works through repeated **weight updates**
- how to train and test a basic classifier on a real dataset
- how changing the activation function and label encoding changes model behavior

## Perceptron Concept
The perceptron is a **linear classifier** that separates data using a decision boundary. It takes input features, multiplies them by weights, adds a bias, and then applies an activation function to produce an output. 
The initial activation function was the **step function**, which outputs one class if the value is above a threshold and the other class otherwise.
Later we used **Sigmoid** activation function which returns a continous value between 0 & 1. Which can be used for classification via thresholding.

Mathematically, the perceptron computes:

$$
z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
$$

Then the step function is applied:

$$
\hat{y} =
\begin{cases}
1, & z \ge 0 \\
-1, & z < 0
\end{cases}
$$

or sigmoid:

$$
\hat{y} = \begin{cases}
1, & 1/(1 + e^{-z}) \ge 0.5 \\
0, & 1/(1 + e^{-z}) < 0.5
\end{cases}
$$
## Training Process

The most important idea in this tutorial was how the perceptron learns. During training, the model:

1. takes an input sample
2. computes a prediction
3. compares it with the true label
4. updates the weights if the prediction is wrong

The update rule is:

$$
\Delta w_i = \eta (y - \hat{y}) x_i
$$

where:

- $$\eta\$$ is the learning rate
- $$y\$$ is the true label
- $$\hat{y}\$$ is the predicted label
- $$x_i\$$ is the input feature

If the model makes a mistake, it adjusts its weights to reduce the chance of making the same mistake again.

## Dataset and Workflow

The tutorial used the **Iris dataset** which contains 150 datapoints, 4 features `Sepal Length` `Sepal Width` `Petal Length` `Petal Width` & six labels `Iris-virginica` `Iris-setosa` `Iris-versicolor`.

The workflow followed these steps:

- load the dataset
<img width="830" height="197" alt="image" style="border-radius: 30%;" src="https://github.com/user-attachments/assets/8d95f0d9-f220-4902-94bf-ba68d85a0577" />

- separate input features and target labels
<img width="833" height="118" alt="image" src="https://github.com/user-attachments/assets/1908e76c-7107-4341-9b50-19c252d94e89" />

- split the data into training and testing sets
<img width="831" height="157" alt="image" src="https://github.com/user-attachments/assets/f4bb7333-bb44-42ce-8c53-ee4d97d8f12b" />

- train the perceptron on the training data
<img width="824" height="65" alt="image" src="https://github.com/user-attachments/assets/4e45e928-b825-413b-a661-aacc6e7938d5" />

- generate predictions on the test data
<img width="830" height="55" alt="image" src="https://github.com/user-attachments/assets/2aa7483c-fa1e-40a2-beeb-6ed2971d0b5f" />

- evaluate the results using accuracy
<img width="830" height="101" alt="image" src="https://github.com/user-attachments/assets/b8a3e864-bb42-4ee6-abe4-4e3c58e4111a" />


## Results and Evaluation

<img width="878" height="40" alt="image" src="https://github.com/user-attachments/assets/262dbde0-4a26-4230-a7bc-aba7c0d5d4d4" /><br>
The model was evaluated using **accuracy** on the test set. This was a suitable metric for this tutorial because the task was simple binary classification and the main goal was to understand the learning process rather than optimize a complex system.
<img width="880" height="19" alt="image" src="https://github.com/user-attachments/assets/2e6fbccb-237f-46c2-bd19-72553b8a0eb0" />


## Tasks

### 1. Manual Input Prediction

Entering flower feature values manually to using the trained perceptron to predict unseen data.
<img width="832" height="212" alt="image" src="https://github.com/user-attachments/assets/cb630042-80eb-4604-9653-522647263a76" />

<img width="830" height="119" alt="image" src="https://github.com/user-attachments/assets/5174cfa1-6c32-4d86-9f97-cd4ca08d4325" />

### 2. Replacing Step Function with Sigmoid

a smoother nonlinear activation is more common in modern neural network models
<img width="830" height="366" alt="image" src="https://github.com/user-attachments/assets/1a9f930d-ce2a-4902-8fc8-24524ef08843" />

### 3. Changing Labels from `1, -1` to `1, 0`

`1, 0` is a better choice when using sigmoid-based binary classification, especially when outputs are interpreted relative to a threshold such as `0.5`.
<img width="834" height="298" alt="image" src="https://github.com/user-attachments/assets/3f3821ff-1eb7-4e9f-a3da-83f3ac723ecc" />

### 4. sklearn based implementation
using `LogisticRegression` from  `sklearn.linear_model` to build Preceptron
<img width="830" height="479" alt="image" src="https://github.com/user-attachments/assets/4cdd0f21-73b2-490f-a6e4-d6ace92d9c72" />

## Key Takeaways

- a perceptron is a simple **linear binary classifier**
- predictions come from a **weighted sum + activation**
- learning happens through **error-based weight updates**
- a simple model pipeline:
  - data preparation
  - training
  - prediction
  - evaluation
