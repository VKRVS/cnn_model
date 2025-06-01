# Jupyter Notebook: MNIST Handwritten Digit Recognition with CNN

This Jupyter Notebook demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras. It also includes steps for data augmentation, k-fold cross-validation, visualizing training progress, and predicting on a custom image.

## 📍 Overview

The notebook covers the end-to-end workflow of an image classification task:

1.  **Data Loading and Preprocessing:**
    *   Loading the MNIST dataset.
    *   Normalizing pixel values to the range [0, 1].
    *   Reshaping images to include the channel dimension (28x28x1).
    *   One-hot encoding the labels.
    *   Setting up an `ImageDataGenerator` for data augmentation (used in K-Fold cross-validation).
2.  **CNN Model Architecture:**
    *   Defining a sequential CNN model with Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, and Dropout layers.
3.  **Model Compilation:**
    *   Compiling the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
4.  **Model Training:**
    *   Initial training using a standard train-test split with `EarlyStopping` to prevent overfitting.
    *   Advanced training using Stratified K-Fold Cross-Validation with data augmentation.
5.  **Evaluation:**
    *   Evaluating the model's performance on the test set.
    *   Calculating and displaying the average cross-validation accuracy.
6.  **Visualization:**
    *   Plotting training and validation accuracy and loss curves.
7.  **Prediction on Custom Image:**
    *   Functions to load, preprocess (including color inversion for typical digit images), and predict a digit from a custom PNG image.
8.  **Model Saving:**
    *   Saving the trained model to a `.keras` file.

## ⚙️ Prerequisites

*   Python 3.x
*   Jupyter Notebook or a compatible environment (e.g., VS Code with Jupyter extension).
*   The following Python libraries:
    *   `tensorflow` (which includes Keras)
    *   `numpy`
    *   `matplotlib`
    *   `scikit-learn` (specifically for `StratifiedKFold`)
    *   `Pillow` (PIL - Python Imaging Library, often a dependency for image processing in Keras)

You can typically install these libraries using pip:
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow
