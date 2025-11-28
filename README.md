# 👕 Fashion-MNIST Neural Classification 🤖

## 💡 Project Overview

This is a machine learning project that implements a **simple feed-forward Neural Network (NN)** to classify images from the **Fashion-MNIST** dataset. The goal is to build, train, and evaluate a neural network capable of correctly categorizing grayscale images of clothing items.

---

## 📊 Dataset

The project uses the widely available Fashion-MNIST dataset, a standard benchmark in machine learning.

* **Source:** Loaded directly from `tf.keras.datasets.fashion_mnist`.
* **Contents:** The dataset consists of 70,000 grayscale images (60,000 for training and 10,000 for testing).
* **Image Dimensions:** Each image is $28 \times 28$ pixels.
* **10 Classes (Labels):**
    1.  T-shirt/top
    2.  Trouser
    3.  Pullover
    4.  Dress
    5.  Coat
    6.  Sandal
    7.  Shirt
    8.  Sneaker
    9.  Bag
    10. Ankle boot

---

## 🧠 Model Architecture

The classification model is built using the Keras Sequential API and is designed for speed and simplicity in image classification.

The architecture is defined as follows:

| Layer (Type) | Output Shape | Parameters | Role |
| :--- | :--- | :--- | :--- |
| `flatten` (Flatten) | (None, 784) | 0 | Flattens the $28 \times 28$ input into a single $784$ element vector. |
| `dense_2` (Dense) | (None, 128) | 100,480 | Fully connected hidden layer. |
| `dense_3` (Dense) | (None, 10) | 1,290 | Output layer corresponding to the 10 classes. |

* **Total Trainable Parameters:** 101,770.

---

## ⚙️ Training Details

The model was compiled and trained using industry-standard settings for multi-class classification:

* **Epochs:** 20.
* **Optimizer:** `Adam`.
* **Loss Function:** `SparseCategoricalCrossentropy(from_logits=True)` (Chosen because the labels are integer-encoded, not one-hot encoded).
* **Metrics:** `Accuracy`.

---

## ✅ Key Results

After 20 epochs of training, the model achieved the following performance on the training data:

| Metric | Value |
| :--- | :--- |
| Final Loss | 0.1784 |
| Final Training Accuracy | 0.9332 (93.32%) |

---

## 🛠️ Installation and Setup

To run this project, you need a Python environment with the following libraries installed.

### Prerequisites

* Python 3.x
* Jupyter Notebook or JupyterLab

### Required Libraries 📚

The following dependencies are used for data handling, visualization, and model building:

* `tensorflow` (including `keras`)
* `numpy`
* `pandas`
* `matplotlib`

You can install all necessary libraries using **pip**:


pip install tensorflow numpy pandas matplotlib 🚀

### 🚀 Usage 

To get started with this project, follow these simple steps:

1.  **Clone / Download:** Clone this repository or download the `nural_clasification.ipynb` file directly. 💾
2.  **Navigate:** Navigate to the project directory in your terminal. 📁
3.  **Launch Jupyter:** Launch a Jupyter environment using the following command:

    ```bash
    jupyter notebook
    ```

4.  **Run:** Open the `nural_clasification.ipynb` notebook and run the cells sequentially (from top to bottom) to load the data, preprocess it, define the model, train it, and view the results. 💡
