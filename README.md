# DeepFake-Detection
This repository aims to train a deep learning-based deepfake detection model from scratch using Python, Keras and TensorFlow. The proposed deepfake detector is based on the EfficientNet structure with some customizations on the network layers, and the sample models provided were trained against a massive and comprehensive set of deepfake datasets.

# Dataset:
The original data used in this project is from a public Kaggle dataset called ["Deepfake Detection Challenge"](https://www.kaggle.com/competitions/deepfake-detection-challenge/data).
(Due to computational limit, only 5 zip files were used for training).

# Getting Started:
## Installation:

```pip install -r requirements.txt```

## Modeling:
### Step 0: Converting video frames to individual images.
Run ```python 00-video_to_image.py```
### Step 1: Extracting faces from the deepfake images with MTCNN.
Run ```python 01-crop_faces.py```
### Step 2: Balancing and splitting datasets into various folders.
Run ```python 02-fake_real_dataset.py```
### Step 3: Model training.
Run ```python 03-train_model.py```

In this code sample, we have adapted the EfficientNet B0 model in several ways: The top input layer is replaced by an input size of 128x128 with a depth of 3, and the last convolutional output from B0 is fed to a global max pooling layer. In addition, 2 additional fully connected layers have been introduced with ReLU activations, followed by a final output layer with Sigmoid activation to serve as a binary classifier.

# Experimental Results:
Performing extensive training and hyperparameter tuning, such as comparing different EfficientNet models, number of convolution layers, weights, data augmentations, dropout rates, and regularizers. In the end, the following settings give us the best results:

* Input Size: 128 x 128
* Batch Size: 32
* Optimizer: Adam
* Learning Rate: 0.0001
* Dropout Rate: 0.5
* Regularization: L2 with 0.001 rate
* Trainig Accuracy: 93.05%
* Val Accuracy: 90.42%
