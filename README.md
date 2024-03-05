# Hand Written Digit Recognition using Keras

![Prediction](https://github.com/omrusman/Hand-Written-Digit-Recognition/assets/87089227/bf379846-7895-4c9c-b256-6413b60da8ea)

## Overview
The project is a Python-based machine learning project that uses the Keras library to classify handwritten digits from the MNIST dataset.
Here's an overview of the steps involved:
- Imports: The script begins by importing necessary libraries such as numpy, matplotlib, keras, sklearn, and seaborn.
- Importing Dataset: The MNIST dataset is loaded using the Keras datasets module. The dataset is split into training and testing sets.
- Visualization: The script visualizes the first instance of each class (0-9) in the training set. It also prints the first 10 labels of the training and testing sets.
- Data Preprocessing: The labels are converted to categorical format using one-hot encoding. The pixel values of the images are normalized to the range [0,1] by dividing by 255. The images are reshaped from 28x28 to 784x1 to be fed into the neural network.
- Model Creation: A Sequential model is created with two Dense layers (each with 128 units and ReLU activation), a Dropout layer for regularization, and a final Dense layer with 10 units (for the 10 classes) and softmax activation. The model is compiled with the 'adam' optimizer and 'categorical_crossentropy' loss function.
- Training: The model is trained on the training data for 10 epochs with a batch size of 512.
- Evaluation: The trained model is evaluated on the test data, and the loss and accuracy are printed.
- Prediction: The model makes predictions on the test data. The predicted classes are extracted from the output probabilities. A random sample from the test set is chosen, and its true and predicted labels are printed.
- Finally, a confusion matrix is generated to provide a detailed evaluation of the model's performance.

<p align="center">
  <img width="746" height="555" src=https://github.com/omrusman/Hand-Written-Digit-Recognition/assets/87089227/4109fcf0-d7aa-415e-a8e4-6c048c14481d>
</p>


## Model Architecture
```
model = Sequential()

model.add(Dense(units=128, input_shape=(784,), activation = 'relu'))
model.add(Dense(units=128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
```

## Libraries
- Keras - library that provides a Python interface for artificial neural networks.
- sklearn - library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
- seaborn - library for data visualization
- matplotlib - library for creating static, animated, and interactive visualizations in Python
- numpy - Library used to perform a wide variety of mathematical operations on arrays.
