# Tutorial 2 — Multi-Layer Perceptron (MLP) Classifier


## Overview
This tutorial introduced the **Multi-Layer Perceptron (MLP)** classifier, a deeper model with hidden layers, nonlinear activations, and a more realistic training process for multiclass classification using the **Iris dataset**. 


## What I Learned
- what a **Multi-Layer Perceptron** is
- why **data scaling** is important for neural networks
- how to train an `MLPClassifier` on structured data
- how to evaluate multiclass predictions using **accuracy** and a **classification report**
- how to inspect the trained model’s structure
- how to visualize and interpret the **training loss curve**
- how changing hidden layers, neurons, and learning rate affects convergence and performance :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}


## MLP Concept
An MLP is a feedforward neural network made up of:

- an **input layer**
- one or more **hidden layers**
- an **output layer**

The input nodes feed into hidden neurons and then into output neurons, this is how information flows through the network. 
Unlike a simple perceptron, an MLP can learn more complex patterns because it has hidden layers and nonlinear activations.


## Dataset and Preprocessing
We used the **Iris dataset**, which has:
- Three classes: `Setosa` `Versicolor` `Virginica`
- Four features: `Sepal Length` `Sepal Width` `Petal Length` `Petal Width`
The preprocessing workflow followed these steps:
- load & split the dataset into training and testing
<img width="906" height="151" alt="image" src="https://github.com/user-attachments/assets/1ddf81aa-1ff2-435c-9ae4-4a1a9bc05f0a" />
- standardize the features using **StandardScaler** to mean 0 and standard deviation 1
Scaling ensures that all features contribute more equally to the learning process as neural networks are much more sensitive to feature scale than many simpler models.
<img width="906" height="106" alt="image" src="https://github.com/user-attachments/assets/2f1a7dfc-3792-4c5a-8db1-006518e4efba" />


## Training the MLP Classifier
Then we created an `MLPClassifier` with **two hidden layers of 10 neurons each**, trained on the scaled training data. `random_state=42` is used for reproducibility.
I followed this base setup and then extended it by trying multiple hidden-layer configurations. This matched the tutorial task of modifying the number of layers and neurons to see how performance changes. 
<img width="906" height="104" alt="image" src="https://github.com/user-attachments/assets/292c7b0c-2bd2-42aa-9338-bddb19148737" />


## Model Evaluation
After training, the model was used to predict the test set, and its performance was evaluated using:

- **Accuracy:** the overall percentage of predictions the model got correct.
- **Precision:** out of the samples predicted as a class, how many were actually that class.
- **Recall:** out of the samples that truly belong to a class, how many the model correctly found.
- **F1-score:** the balanced combined score of precision and recall.
- **Support:** the number of true samples belonging to a class in the dataset.

**Classification report:** a summary table showing precision, recall, F1-score, and support for each class.
<img width="906" height="342" alt="image" src="https://github.com/user-attachments/assets/9d6daced-cb20-40f1-a836-5de70b37cb93" />

The PDF also explains the meanings of **support**, **macro average**, and **weighted average**, which helped me better understand how multiclass evaluation is reported. It showed that macro average treats all classes equally, while weighted average gives more importance to classes with more samples.


## Inspecting the Model Structure

Printing information about the trained MLP structure, including:

- number of layers
- number of outputs
- activation function
- output activation function
- number of epochs 

This made the training process feel less like a black box. Instead of only seeing the final predictions, I could also inspect the architecture and how long training took.
<img width="906" height="174" alt="image" src="https://github.com/user-attachments/assets/c1836da0-d729-45a1-8a97-968cd62a26d5" />


## Learning Curve Visualization

We then plotted the **loss curve** across epochs. This is one of the most useful parts because it shows whether the model is steadily converging during training. 
Smooth decreasing curves indicate stable learning. I could better understand the training process instead of only looking at final accuracy.

<img width="906" height="758" alt="image" src="https://github.com/user-attachments/assets/92c9ae83-562d-4512-a155-88cb60a2883e" />


## Tasks
I developed a experiment running function that prompts the user for 
- Number of experiments; then for each experiment:
  - Layers & neurons
  - Learning Rate
  - Iterations
I then used it for both tasks

Task 01: modify hidden layers and neuron counts, then observe the effect on accuracy and learning curve
<img width="906" height="591" alt="image" src="https://github.com/user-attachments/assets/e83ea3f7-a9bd-4d33-be6f-ec15b210d981" />
These results show that increasing number of layers improves convergence, increasing number of neurons also helps to a lesser extent. decreasing neurons in successive layers (horizontal funnel) has little to no benefit.
Also almost no effect on the accuracy was observed.

Task 02: change the learning rate and observe the effect on convergence, loss curve, and number of epochs 
<img width="906" height="574" alt="image" src="https://github.com/user-attachments/assets/b0e180f3-8341-493e-b874-e7481c814510" /><br>
This chart was spoiled due to one experiment having a large learning rate of 0.1 which caused "exploding gradient" problem.<br>
Repeating set of experiments without unstable learning rate
<img width="906" height="562" alt="image" src="https://github.com/user-attachments/assets/c6584b3d-8efd-48a4-a266-3da7f0e47abd" />
These experiments showed that a learning rate of 1 percent drastically improved `number of epochs` required for convergence. in the case of [100, 50, 25] with learning rate of `0.001`, `357` epochs were required to convergence. 
When learning rate was increased to `0.01`, the accuracy dropped by less than 3% to 97.7% & epochs dropped to 101.

## Key Takeaways

This tutorial helped me move from a basic perceptron to a more practical neural network classifier. The most important things I learned were:

- MLPs use **hidden layers** to model more complex patterns
- **feature scaling** is important before training neural networks
- evaluation should include both **accuracy** and detailed class-wise metrics
- the **loss curve** provides insight into convergence
- architecture size and learning rate both affect model behavior
- understanding the underlying math makes the library implementation easier to interpret

Overall, this tutorial gave me a much clearer understanding of how a small neural network is trained, evaluated, and analyzed in practice.
