# Final Report

## Group Members

Yang Haocheng 1004883 

Lu Mingrui 1005354



## 1. Problem Statement

This project aims to develop a deep learning model that can accurately classify CT scans as either COVID-19 positive or negative, using a dataset of labeled CT scans of COVID-19 and non-COVID-19 patients.it can aid healthcare professionals in accurately and quickly diagnosing COVID-19 from CT scans.



## 2. Dataset and Collection

We collected a dataset of COVID-19 chest CT scans from various sources, including publicly available datasets and hospitals. The dataset consists of 500 CT scans from COVID-19 positive patients, and 500 CT scans from healthy patients. The CT scans were collected from different hospitals across the world, using different CT scanner models and settings. The dataset was annotated by radiologists to indicate the presence or absence of COVID-19 infection. 
the source：https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed



## 3. Data Pre-processing

This code is used to split a dataset consisting of CT scans of patients into a training set and a test set. The dataset contains two classes: COVID-19 scans and non-COVID-19 scans. The code first creates two directories to store the COVID and non-COVID images separately. It then loops through each image in the original dataset and copies it to the appropriate directory while also appending its file path to the data_target list and its label to the data_label list.

Next, the code uses the train_test_split function from the sklearn library with test_size=0.3 to randomly split the data_target and data_label lists into a training set and a test set. The training set and test set are each split into two subdirectories, one for COVID images and one for non-COVID images. The images in each set are then copied to their respective subdirectories.

## 4. Algorithm/Model:

### Data Augmentation

In this project we utilize `torchvision.datasets.ImageFolder`  module to read image data, decode jpeg images into RGB pixel networks.

And we also performed data augmentation to make our model more robust.

```
train_transforms = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

RandomRotation: This applies a random rotation to the image, with a degree range of 0 to 40. This helps the model to become more robust to variations in image orientation.

RandomResizedCrop: This crops a random area of the image and resizes it to the specified size (150x150 pixels in this case), with a scale factor randomly chosen between 0.8 and 1.0. This helps the model to become more invariant to changes in image scale.

RandomHorizontalFlip: This flips the image horizontally with a probability of 0.5. This helps to increase the amount of data available to the model and makes it more robust to left-right image orientation.

Normalize: This normalizes the image pixel values using the mean and standard deviation values specified in the two lists ([0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] in this case). This helps to ensure that the input data has a similar scale and distribution, which can improve the model's performance.

### Our Model

Here we decided to use Convolutional Nerual Network to capture spatial dependency and utilize monophily of pixels. The model is structures inside `cnn.py`.

The total number of parameters to be trained is: 3453121, the size of the first layer convolution kernel is: 33, the number of convolution kernels is 32, and the activation function is RELU; the second layer is the pooling layer, pooling The size is 22, the third layer convolution kernel size is: 33, the number of convolution kernels is 64, and the activation function is RELU; the fourth layer is a pooling layer, and the pooling size is 2 2. The size of the fifth layer convolution kernel is: 33, the number of convolution kernels is 128, and the activation function is RELU; the sixth layer is the pooling layer, the pooling size is 22, and the seventh layer The size of the convolution kernel is 33, the number of convolution kernels is 128, and the activation function is RELU; the eighth layer is the pooling layer, and the pooling size is 22, and the ninth layer uses the size of the eighth layer Perform tiling, then add the DROUPOUT layer, randomly drop 50% of the convolution kernel, add a dropout layer to further improve the recognition rate, then add a fully connected layer with a length of 512, the activation function is RELU, and finally add a classification layer , the classification function is SIGMOID. 

<img src="https://p.ipic.vip/5v7mtc.png" alt="截屏2023-04-16 下午8.58.10" style="zoom:50%;" />



### Hyperparameters tuning

First we have decided to Evaluate performance of model using **Accuracy**, so that we can decide the best combination of hyperparameters through choosing the combination with best accuracy on validation set.

#### Learning rate, num of epochs, optimizer, dropout rate

For these we are using grid search to determine the best combination

For learning rate we have tried 1e-2, 1e-3, 1e-4, 1e-5;

For num of epochs we have tried 100, 200, 500, 1000;

For dropout rate of dropout layer we have tried 0.3, 0.5, 0.8;

Finally we decided to use RMSprop with learning_rate=1e-4 and train for 500 epochs with batchsize 20 and a dropout rate of 0.5.

#### Structure of model

We have combined the method of grid search and random search when deciding the structure of model.

Which is use random search initially to explore the hyperparameter space to determine the size of filters, number of layers, etc. Then use grid search to fine-tune the optimal values.

These hyperparameters were chosen based on empirical experimentation and prior knowledge about the task the model is supposed to perform. The kernel size of the convolutional layers is typically set to an odd number to ensure that the output feature maps have the same spatial dimensions as the input. The padding is used to ensure that the spatial dimensions of the feature maps are preserved after convolution. The pool layer reduces the spatial dimensions of the feature maps and helps to reduce overfitting. The number of output channels in the convolutional layers gradually increases, while the spatial dimensions of the feature maps gradually decrease, allowing the model to learn increasingly complex features as it progresses through the layers. The number of output features in the fully connected layers was chosen to be 512 based on the size of the input features and the desired complexity of the model. The dropout layer helps to prevent overfitting by randomly dropping out units during training.



## 5. Evaluation Methodology:

### Loss function: Binary Cross-entropy

The Binary cross-entropy loss function is often used in binary classification problems.

### Evaluation metric of model: Accuracy on test set

The dataset is split into training set and a test set with test_size=0.3 randomly. And the effectiveness of model is reflected via the accuracy on test set.

Accuracy is the easiest to understand. It refers to the samples that are correctly classified in all samples, including all sample categories.



## 6. Results

Here we use 500 epochs, and the number of CT images for each training iteration(batch size) is 50, and the number of test images is also 50 for one validation. 

The training error at the beginning is relatively large. As the number of epochs gradually increases, training The error gradually decreases and tends to be stable, and the validation accuracy gradually increases.

The results of the training set and validation set on the accuracy index are as follows:

<img src="https://p.ipic.vip/n87jt6.png" alt="accuracy" style="zoom:50%;" />

 Finally, the accuracy of our model on the test set is: **0.8293**.



## 7. Compare our model with existing ones

In this project we also use transfer learning  where we use a pre-trained model as a starting point for a new task and compared its performance with our final model which we build and train from scratch

Firstly, we loaded a pre-trained model ResNet-18 with its weights which is a pre-trained convolutional neural network architecture that has shown remarkable performance on the ImageNet dataset. So it possibly has better ability to extract meaningful features from images than our final model.

Then we extracts the pre-trained model's feature extractor layers, which exclude the final classification layer. The original last layer of ResNet-18 is a fully connected layer that outputs a tensor of shape [batch_size, num_classes], where num_classes is the number of classes in the ImageNet dataset. Since our new task is binary classification task (COVID-19 vs Non-COVID-19), we need to remove this last layer and replace it with our own classification layer.

Thus we need to flatten the output of the feature extractor layers to a 1D tensor first. This is necessary because the output of the ResNet-18 feature extractor layers has a shape of [batch_size, num_channels, height, width], where num_channels, height, and width depend on the size of the input image. We need to flatten this tensor to a 1D tensor so that we can pass it through the fully-connected classification layer.
Finally, we add a full-connected layer with 512 input features (which is the output size of the last convolutional layer in ResNet-18) and 2 output features (which is the number of classes in our new binary classification task). This layer is the classification layer that maps the output of the feature extractor layer to the binary classification output.

Overall, in this transfer learning we created a new model that reuses the pre-trained ResNet-18 architecture for feature extraction and replaces the final classification layer with a new binary classification. 

The accuracy on test set of the model with transfer learning is **0.82666**

<img src="https://p.ipic.vip/e0702u.jpg" alt="IMAGE 2023-04-16 20:53:13" style="zoom:50%;" />

And when we compare the performance of this pre-trained model with our final model, the result turns out that our own model has the better performance. We think the reason is because transfer learning models require a large amount of data to learn meaningful representations. but our dataset is relatively small so ResNet-18 may not have enough data to learn good representations, and another reason we think is because ResNet-18 was pre-trained on the ImageNet dataset, which is a very large dataset of diverse natural images but our own dataset contains only COVID-19 chest CT scans which is a very different images distribution from the ImageNet dataset, hence ResNet-18 may not be able to learn meaningful representations for our specific task. So finally we still choosed our own CNN model which we build and train from scratch as our final model.



## Steps to re-train the model from scratch

### required libraries

`tensorflow 2.12.0`,`matplotlib`,`scikit-learn 1.2.2`,`PIL`, `shutil`

### Step1

The original project folder is structured in this way:

```
.
├── dataset/
│   ├── CT_COVID
│   └── CT_NonCOVID
├── cnn.py
├── inference.py
├── data_preprocess.py
└── main.py
```

Run `data_preprocess.py` to preprocess the `dataset` and generate `train_set` and `test_set`

Detail of the code for `data_preprocess.py`  can refer to '3. Data Pre-processing'

After running the file the folder will look like this:

```
.
├── dataset/
│   ├── CT_COVID
│   └── CT_NonCOVID
├── covid
├── non_covid
├── train_set/
│   ├── covid
│   └── non_covid
├── test_set/
│   ├── covid
│   └── non_covid
├── cnn.py
├── inference.py
├── data_preprocess.py
└── main.py
```

### Step2

Run `main.py`

This will trigger training on `train_set` and validation on `test_set`

After the training is done the model will be saved to `savedmodel/trained_model.pth`

The figure for Training and validation acc/loss will also be saved as `accuracy.png` and `loss.png`

After running the file the folder will look like this:

```
.
├── dataset/
│   ├── CT_COVID
│   └── CT_NonCOVID
├── covid
├── non_covid
├── train_set/
│   ├── covid
│   └── non_covid
├── test_set/
│   ├── covid
│   └── non_covid
├── cnn.py
├── inference.py
├── data_preprocess.py
├── main.py
├── savedmodel/
│   └── trained_model.pth
├── accuracy.png
└── loss.png
```



## How to make prediction using our trained model

This can be done by changing the path of Input CT scan inside `inference.py` and run `inference.py`

The result will be printed.



## How to recreate the exact trained model

The trained model is saved in `savedmodel/trained_model.pth`

This can be done by first import model from cnn.py `from cnn import CNNModel`

then loading the trained model:

```
state_dict = torch.load("savedmodel/trained_model.pth")
reconstructed_model = CNNModel()
reconstructed_model.load_state_dict(state_dict)
```

