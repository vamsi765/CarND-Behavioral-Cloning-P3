# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image6]: ./straight.jpg "Normal Image"
[image7]: ./flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run_working
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use nvidia convolution neural network model with 3x3 filter sizes and depths between 3, 6, 9 and 12 (model.py lines 95-114) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 98)[I had to remove this optimization as the model was created using python 3.5 but I was running it on python 3.6 on my local system]. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 104 and 107). 

The model was trained and validated on different data(flipping random images) sets to ensure that the model was not overfitting (code line 35-37). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

I used the sample training data provided. I removed the large concentration of 0 parameters to have close distribution of data so as to remove any bias. Also, the images are randomly flipped so as to remove left directional bias.


For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have used nvidia model for image processing.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

I had the issue where all the time 1 was being predicted from the model, this was an issue with softmax function at the final model. After solving this, I found that the model was always giving the same predicted value, then I realised that during training the data wasn't randomized. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 40, 160, 3)        84        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 80, 6)         456       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 40, 9)         1359      
_________________________________________________________________
dropout_1 (Dropout)          (None, 10, 40, 9)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 40, 12)        984       
_________________________________________________________________
dropout_2 (Dropout)          (None, 10, 40, 12)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4800)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               480100    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 488,554
Trainable params: 488,554
Non-trainable params: 0
_________________________________________________________________


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the sample data provided by Udacity team to save time

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image3]


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 107 as evidenced by final simulation I used an adam optimizer so that manually training the learning rate wasn't necessary.
