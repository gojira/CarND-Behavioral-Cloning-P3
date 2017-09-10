# **Behavioral Cloning Project** 

---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_bgr]: ./images/image1_bgr.png
[image]: ./images/image1.png
[image_cropped]: ./images/image1_cropped.png
[image_normalized]: ./images/image1_normalized.png
[image_normalized_cropped]: ./images/image1_normalized_cropped.png
[image_flipped]: ./images/image1_flipped.png

### Rubric Points
Below I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the code to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 for showing successful result of driving at 15mph
* video-25mph.mp4 for showing result of driving at 25mph
* writeup_report.md for summarizing the results, with associated images

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is the Nvidia end to end driving model convolutional neural network.

My model is described in more detail in the next major section.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.  

There are dropout layers after the 5x5 convolution layers, the 3x3 convolution layers, and after each fully connected layer (except the output layer). 

The model also uses L2 regularization in each fully connected layer.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Empirically, I found that generalization (dropout and regularization) was not needed 
to enable the car to drive around track one successfully.  Generalization did help to 
improve the driving by smoothing out the steering from 
frame to frame. There is a significant difference observing the smoothness of
the driving with and without generalization.

#### 3. Model parameter tuning

The model used an adam optimizer with default settings.

#### 4. Appropriate training data

I used the Udacity training data with augmentation as described in the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was as follows.

First, I experimented with the single fully-connected layer setup as described in
the video lecture. That was an interesting baseline to compare against, but did
not perform very well.

After that, I went directly to the Nvidia architecture rather than trying LeNet
or any other architecture for two reasons. First of all, I have known about the
Nvidia work for a few months and wanted to try it out in general.  Secondly,
people in the forum suggested that the Nvidia model was a good model.

In the end I found the Nvidia model worked very well with only minor adjustments.


#### 2. Final Model Architecture

The final model architecture is the Nvidia end to end driving convolutional neural
network model.

The only changes to the Nvidia model are input image shape, and the addition of dropout layers to increase generalization in the model.  Otherwise it is exactly as described in
the [blog post](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 
and [paper](https://arxiv.org/pdf/1604.07316v1.pdf).

The model is constructed in the `make_model` function.

It has the architecture below.

| Layer				| Output Shape	| Description |
|:-----------------:|:--------------:|:-----------:|
| Input				| 75x320x3		| Input RGB image | 
| 5x5 Convolution	| 36x158x24		| 24 filters, stride=2, relu activation|
| 5x5 Convolution	| 16x77x36		| 36 filters, stride=2, relu activation|
| 5x5 Convolution	| 6x37x48			| 48 filters, stride=2, relu activation|
| Dropout				| 6x37x48			| Keep = 0.25 |
| 3x3 Convolution	| 4x35x64			| 64 filters, stride=2, relu activation|
| 3x3 Convolution	| 4x35x64			| 64 filters, stride=2, relu activation|
| Dropout				| 2x3x64			| Keep = 0.25 |
| Flatten				| 4224				| |
| Fully connected	| 100				| 100 hidden units |
| Dropout				| 100				| Keep = 0.5 |
| Fully connected	| 50				| 50 hidden units |
| Dropout				| 50				| Keep = 0.5 |
| Fully connected	| 10				| 10 hidden units |
| Dropout				| 10				| Keep = 0.5 |
| Fully connected	| 1					| 1 hidden unit |



#### 3. Creation of the Training Set & Training Process

I used the Udacity driving data.  I had a difficult time driving the
simulator with keyboard.  I eventually found that connecting an Xbox
controller worked well. For the submission I used the Udacity data.

An example image from the dataset is here:

![alt text][image]

cv2 reads an image in BGR format, which looks like the below. To be
consistent with `drive.py` which reads in RGB format, the image needs
to be converted to RGB format first before processing (which results 
in the image looking like above).

![alt text][image_bgr]

For the training, I experimented with the following.

1. Activation functions
2. Image preprocessing
3. Dataset filtering
4. Dataset augmentation
5. Dropout
6. Regularization


##### Evaluating model performance
For this project, driving performance is the ultimate goal, and therefore
I focused on driving as the criteria for evaluating model architecture,
hyperparameters, and training data/approach.

Generally speaking, with this project, I found that loss and accuracy were
not completely correlated with driving performance.  First of all, the
accuracy did not change very much, during training, and 
whether I changed model architecture, training 
epochs, or training images.  The accuracy was also not very different between
trained models that stayed on course and those that went off course.

The validation loss was a good guide for understanding how many epochs were
needed to stop training.  In general, 3-5 epochs were sufficient for training.
I trained up to 20 epochs but more epochs did not result in better driving.

##### Training / Validation / Test

I used standard approach of splitting data into training and validation
sets.  As described above, with this project, accuracy was not a good
indicator of driving performance, which is the actual objective criterion
for success. The test "set" is actually running the model
in the simulator using drive.py, and judging whether the car is able to
drive around the track without going off track, as well as how fast it can go.

I shuffled the data set and used 20% validation split.

##### Activation functions
I tried using Relu for convolutional layers and for the fully connected
layers. The model worked best with Relu for the covolutional layers.
For the fully connected layers, the model performed best without non-linearity
(i.e., no special activation function).

##### Image preprocessing
For image preprocessing, I experimented with the following.

1. Normalization
2. Cropping
3. Resizing

Since the model is used in the `drive.py` file, I modified `drive.py`
to apply the same image augmentation in the driving phase as training.

All image preprocessing is done in the `process_image` function which is 
used in both `model.py` and `drive.py`.

Normalization is the standard linear transformation of the (0,255) pixel range
into (-0.5,+0.5).

For reference, a normalized version of the image from before looks like the following:

![alt text][image_normalized]

For cropping, I set the crop boundaries as training parameters that could
be modified in the training setup.  I tried to crop as much of the upper
'sky' portion of the image as possible, and the car hood as well.  Ultimately
I settled on 60 pixels at top and 25 at bottom out of the 320x160 image.

The normalized, cropped image looks like this:

![alt text][image_normalized_cropped]

I experimented with resizing the normalized and cropped image, for example
to 64x64, as suggested in the forums.  However, I found that driving 
performance degraded when doing this so my final submission omits
image resizing.

##### Dataset filtering
I filtered training examples based on how close to straight the steering
angle is.  I discarded training examples (both image and steering angle)
if the steering angle was less than a threshold, with a certain probability.

This is done when loading the training data in the `read_training_data`
function.  The steering threshold, and drop probability are both 
hyperparameters that I experimented with.

In the end, I chose `0.15` as the steering threshold and `0.8` as
the drop probability.  i.e., I dropped a sample with 80% probability
if the steering angle was less than 0.15.

I also applied steering correction for the left and right images.  This
was also a hyperparameter.  I chose `0.25` as the correction in the end
and it worked well in practice.

This dataset filtering helped a lot with reducing overfitting of the model.
The key here was to throw out most straight angle samples, but not all.
I initially threw out all straight angle samples, and that did not work
well in driving performance.  When I started to keep some straight samples,
the driving immediately improved.

##### Dataset augmentation
I found that dataset augmentation, in addition to dataset filtering
and image cropping, had the highest impact to driving performance.

For augmentation, I added, for every single image, a flipped version 
of the image, with correspondingly flipped steering.

A flipped version of the normalized cropped image from before
looks like this:

![alt text][image_flipped]

Before dataset augmentation, I was able to load the whole dataset
into memory for training.  With image flipping doubling the
dataset size, I ran out of memory and switched to using a
generator for training and validation data.


##### Dropout and regularization.
I did most of my initial experiements with just one dropout layer
after the convolutional layers.

Once the model was able to drive all the way around the track, I added
more dropout layers in the fully connected layers, and added L2
regularization to the fully connected layers.

This worked immediately to smooth the driving performance so that it was
less jerky.  I did not experiment with the hyperparameters such as 
the regularization constant as the initial setup sufficiently improved
driving performance.


### Videos
I experimented with driving speed with different models, from the default
9mph, to 15mph, 20mph, 25mph, up to 30mph.  

The file `video.mp4` captures the submitted model at 15mph.

The same model drives correctly at 25mph, staying on the road the whole time. 
However, it does get close to the right edge toward the end of the lap and
it may not be fully apparent from the video that it stays on the road.
The file `video-25mph.mp4` shows the driving run at 25mph.
