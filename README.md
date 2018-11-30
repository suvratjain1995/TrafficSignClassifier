# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./GraphImages/Uneven.png "Visualization"
[image2]: ./GraphImages/Grayscale.png "Grayscaling"
[image3]: ./GraphImages/Even.png "Random Noise"
[image4]: ./GraphImages/LossCurve.png "Traffic Sign 1"
[image5]: ./TestImages/Image1.jpg "Traffic Sign 2"
[image6]: ./TestImages/Image2.jpg "Traffic Sign 3"
[image7]: ./TestImages/Image3.jpg "Traffic Sign 4"
[image8]: ./TestImages/Image4.jpg "Traffic Sign 5"
[image9]: ./TestImages/Image5.jpg "Traffic Sign 6"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to check number of data I have for training,testing and validataion:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32X32**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

This chart below shows us the number of traing example for each class. It can be easily seen that the data is unbalanced.I aim to make data even using image augmentation such that each class have mean number of the test data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the data.

**Step1**:- I decided to grayscale the image using opencv as given below
```python
cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

**Step2**:- I decided to normalize the data so that the value of each pixel is between 0.1-0.9.The reason to chose such range of value was to make sure that our ativation function (RELU)doesnt get killed or exhausted because of the value being less that 0
Here the command that I used to normalize the pixel of the image.

```python
### Here x is the pixel value###
x = x/255*0.8+0.1
```

As we have observed in the previous graph showing us that number of data for each class is unbalanced, So I am to add certain number of image to each class that has number of test data less than the mean of the entire data. 

For Image augmentation  used **Keras** predefined library named 
**ImageDataGenerator**
Following is the code that I used to augment the images to the training data.

```python
datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range = 0.2,
        fill_mode='nearest')
datagen.flow(X_train[temp],y_train[temp],batch_size = 1)
```
What this snippet of does is that it keeps adding images indefinetly with randomly shearing the image to 20% or rotating the image to 40degrees.


Here is an example of an original image and an augmented image:

![alt text][image2]

We can see that this image have been sheared to 20% of the actual image.

After image augmentation we were able to balance out the number of test data for each class
Here a visualization of the graph after image augmentation

![alt text][image3]



#### 2. Architecture of the Neural Network Model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| AVG pooling 2x2	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU		|         									|
| AVG pooling 2x2			|  2x2 stride,  outputs 5x5x16      									|
|	Fully Connected Layer Input 400					|Weights 400x200 Bias 200												|
|		RELU				|												|
|Fully Connected Layer Input 200	|Weights 200x100 Bias 100|
|RELU||
|Fully Connected Layer Input 100|Weights 100x43 Bias 43|



#### 3.Training Model.
The Steps I took to train my Model was

1.Computed the softmax of the output of the model vrs that actual output and calculated the cross-entropy of the model .Returns Probability of the error between the actual output and the computed output.Following command was used to compute the function.
```python
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
```

2.Computed the Mean loss of the entire model using the following command.
```python
loss_operation = tf.reduce_mean(cross_entropy)

```
3.We used AdamOptimizer to minimize the error we have for the model.The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.Following code was used to minimize the error in the model.
```python
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```
The Learning Rate was set to **0.005**

The Epochs was set to **25**

The Batch size was set to **150**

#### 4. Approach and Accuracy of the model .

1st Iteration with original LeNet Arch 
Parameters :-
* Batch-128.
* Learning Rate:-0.005.
* Epochs:-25.
* Models validation Accuray:-93.9002513885498%.

2nd Itration:-

* LeNet Architecture was chosen.
* LeNet Architecture initially suggested for result for hand written numbers . I increased the number of hidden neuron on the fully connected network as suggested by a book that the number of hidden neuron must be between the number of input neuron and number of output neuron.I took number of hidden neurons to be the mean of the input & output neuron. I increased the batch size to 150 for better training.

My final model results were:
* training set accuracy of **98.2%**
* validation set accuracy of **94%**
* test set accuracy of **92%**

### Test a Model on New Images

#### 1.Testing on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5] 

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

### Images Characterstics thats different than the test data ###
* Image has Background likes tress around it and hence makes it difficult for the model to understand the object actually in it,but yet model was able to predict correctly those images.
* Image of No vehicle has car and bike in it and model was not able to detect it making it difficult to detect . It instead understood that its a no entry sign but with less probability that its a case.
#### 2.Models Prediction on the New Images
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield     		| Yield   									| 
| Turn Right     			| Turn right 										|
|Road Work				| Road Work									|
| Pedestrian      		| Beaware of ice and snow			 				|
| No vehicles			| Stop      							|

#### 3. Models Performance on the new Data ####
The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of **60%**.

#### 4. Models Certainity

For Image  1 Probability's are

    Yield 				 0.995927
    Turn left ahead 				 0.0040727
    No vehicles 				 1.10008e-08
    Keep right 				 5.16268e-09
    Ahead only 				 2.13038e-09



For Image  2 Probability's are

    Turn right ahead 				 0.99911
    Stop 				 0.000889522
    Yield 				 1.12821e-07
    Keep right 				 8.23998e-08
    No entry 				 3.56418e-08



For Image  3 Probability's are

    Road work 				 0.999922
    Wild animals crossing 				 6.98046e-05
    Road narrows on the right 				 7.54203e-06
    Bicycles crossing 				 3.58383e-07
    Double curve 				 1.17357e-07



For Image  4 Probability's are

    Beware of ice/snow 				 0.996524
    Bicycles crossing 				 0.00339599
    Road work 				 3.2852e-05
    Road narrows on the right 				 2.63804e-05
    Slippery road 				 1.73779e-05



For Image  5 Probability's are

    Stop 				 0.994628
    Turn left ahead 				 0.00512475
    No entry 				 0.000216603
    Keep right 				 1.88807e-05
    Turn right ahead 				 1.01744e-05




