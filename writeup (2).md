# **Traffic Sign Recognition** 
**TensorFlow-based LeNet and OpenCV are used in this project to classify traffic signs. Dataset used: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This dataset has more than 50,000 images of 43 classes.**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Pipeline architecture:
- **Load The Data.**
- **Dataset Summary & Exploration**
- **Data Preprocessing**.
    - Grayscaling.
    - Normalization.
- **Design a Model Architecture.**
    - LeNet-5.
- **Model Training and Evaluation.**
- **Testing the Model Using the Test Set.**
- **Testing the Model on New Images.**

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"


## Step 0: Load The Data
`pickle` library is used to load the dataset. The figure size is 32x32. They are already seperated with three `.p` files as training set, testing set and validation set. The labels are coded with numbers and stored in [.csv file](data/signnames.csv).

## Step 1 : Dataset Summary & Exploration

Here the data set is explored. First, show some general numbers about it:

- Number of training examples = 34799
- Number of testing examples = 12630
- Number of validation examples = 4410
- Image data shape = (32, 32, 3)
- Number of classes = 43

Some examples of the traffic signs image are shown:

![alt text][image1]

The distibution of the labels are shownL

![alt text][image2]


## Step 2: Design and Test a Model architecture

### Pre-processing

First, the original figures are converted to gray scale. Then, the figures are normalized and zero meaned. Neural networks work better if the input(feature) distributions have mean zero.

![alt text][image3]

### Model architecture

The starting model was [LeNet](http://yann.lecun.com/exdb/lenet/) provided by [Udacity](https://github.com/udacity/CarND-LeNet-Lab). This model was proved to work well in the recognition hand and print written character. It could be a good fit for the traffic sign classification.The model is described as follows:

|Layer | Description|Output|
|------|------------|------|
|Input | Gray image| 32x32x1|
|Convolutional Layer 1 | 1x1 strides, valid padding | 28x28x32|
|RELU| | |
|Max Pool| 2x2 | 14x14x32|
|Convolutional Layer 2 | 1x1 strides, valid padding | 10x10x64|
|RELU| | |
|Max Pool | 2x2 | 5x5x64|
|Fatten| To connect to fully-connected layers |
|Fully-connected Layer 1| | 1600|
|RELU| | |
|Dropout| 0.7 keep probability ||
|Fully-connected Layer 2| | 120
|RELU| | |
|Dropout| 0.7 keep probability||
|Fully-connected Layer 3| | 43

The hyperparameters: Leanrning rate=0.001; Dropout=0.7; Initialization mu=0, sigma=0.1; Epochs=15; Batch size=128.

The model is stored in Lenet.ckpt. Training accuracy is 0.998, validation accuracy is 0.971 and test accuracy is 0.947.

## Step 3: Test a Model on New images

In this step, five new images found on the Web are classified.
First, the images are loaded and presented:

![alt text][image4]

The pre-processing is applied to the figure:

![alt text][image5]

Four out of the five images were classified correctly. That make the network 80% accurate on this images:

![alt text][image6]

Here are the top five softmax probabilities for them and their name values:

- Image: test_figs/stop_sign.jpg
  - It was classified correctly.
  - Probabilities:
    - **0.990601 : 34 - Turn left ahead**
    - 0.007298 : 14 - Stop
    - 0.002091 : 38 - Keep right
    - 0.000004 : 35 - Ahead only
    - 0.000002 : 13 - Speed limit (70km/h)


- Image: test_figs/left_turn.jpg
  - Probabilities:
    - **0.990601 : 34 - [34 'Turn left ahead']**
    - 0.007298 : 14 - [14 'Stop']
    - 0.002091 : 38 - [38 'Keep right']
    - 0.000004 : 35 - [35 'Ahead only']
    - 0.000002 : 13 - [13 'Yield']

- Image: test_figs/60_kmh.jpg
  - Probabilities:
    - **0.496766 : 3 - [3 'Speed limit (60km/h)']**
    - 0.230563 : 9 - [9 'No passing']
    - 0.194319 : 35 - [35 'Ahead only']
    - 0.056915 : 10 - [10 'No passing for vehicles over 3.5 metric tons']
    - 0.004492 : 23 - [23 'Slippery road']

- Image: test_figs/road_work.jpg
  - **Wrong prediction**
  - Probabilities:
    - 0.999916 : 18 - [18 'General caution']
    - 0.000068 : 40 - [40 'Roundabout mandatory']
    - 0.000016 : 26 - [26 'Traffic signals']
    - 0.000001 : 4 - [4 'Speed limit (70km/h)']
    - 0.000000 : 1 - [1 'Speed limit (30km/h)']

- Image: test_figs/yield_sign.jpg
  - Probabilities:
    - **1.000000 : 13 - [13 'Yield']**
    - 0.000000 : 34 - [34 'Turn left ahead']
    - 0.000000 : 12 - [12 'Priority road']
    - 0.000000 : 38 - [38 'Keep right']
    - 0.000000 : 35 - [35 'Ahead only']

- Image: test_figs/stop_sign.jpg
  - Probabilities:
    - **0.983788 : 14 - [14 'Stop']**
    - 0.014333 : 1 - [1 'Speed limit (30km/h)']
    - 0.000722 : 32 - [32 'End of all speed and passing limits']
    - 0.000426 : 3 - [3 'Speed limit (60km/h)']
    - 0.000373 : 0 - [0 'Speed limit (20km/h)']

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

