
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/plt1.png
[image2]: ./examples/plt2.png
[image3]: ./examples/plt3.png
[image4]: ./examples/all.png




### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a visualization of the training data set as example. It is a bar chart showing the quantity and distribution of the total training data in each classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because the shape informations of the traffic signs are enough to do the classification task.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then, I normalized the image data to let the pixel values from [0,255] to [-1,1] so as to let the follwing training more robust, otherwise, it is too sensitve to the outliers.

Here is an example of an original image and an augmented image, actually, that is not apparently different. 

![alt text][image3]

At the very end, i reshape the image sets, which adds a dimension representing the number of color channels and shuffle the image in each of image dataset respectively.

For example : shape of training set is changed from (34799, 32, 32) to (34799, 32, 32, 1).
 
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution_1 5x5     | 1x1 stride, valid padding, outputs 28x28x6 |
| Batch_Normalization_1 | Data normalized | 
| Relu_1				| Activation Function |
| Max pooling_1	      	| 2x2 stride,  outputs 14x14x6 |
| Convolution_2 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| Batch_Normalization_2 | Data normalized | 
| Relu_2				| Activation Function |
| Convolution_3 5x5	    | 1x1 stride, valid padding, outputs 6x6x32 |
| Batch_Normalization_3 | Data normalized | 
| Relu_3				| Activation Function |
| Dropout_3             | Generalization |
| Flatten      	        | change to 1-D Array Length is 1152 |
| Fully connected_1		| 1152 to 256 |
| Relu_f1				| Activation Function |
| Dropout_f1            | Generalization |
| Fully connected_2		| 256 to 64 |
| Relu_f2				| Activation Function |
| Dropout_f2            | Generalization |
| Fully connected_3		| 64 to 43 |
| logits                | output result |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The total neural network model is based on tensorflow, and some important parameters used in the training are listed below:

* optimizer: AdamOptimizer 
* learning rate: 0.001
* batch size: 128
* number of epochs: 50
* keep_prob in dropout : 50%



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy is : 99.6%
* validation set accuracy is : 95.4%
* test set accuracy is : 95.1%

I adjust the original LeNet and create my own 'Model' Architecture. 

At first, i train with original LeNet Network, and the validatzionaccuracy reaches at about 89%, which is below what we need.

Then, I add batch normalization for each conv layers to avoid the model from not being convergent. The accuracy of validation then reaches at about 92&, which has a great improvement in comparision with the former one.

Then, i add convolution layer 3 and change the correponding input and ouput of each layer, which adds extra parameters of the whole network. Although the calculation speed for each epoch will decrease, but the accuracy increased once more. 

After that, I delete the max pooling for conv2 layer and just use the max pool once so as to aviod losing too much information.

I also try to change the batch_size and epochs and finally i choose the batch size 128 and epoch 50, because the runtime is quite acceptable and the result is quite good

After all above, the validation accuracy reaches at 95.4%, which meets the demand.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4] 

All images i choose are very clear and are not difficult to identify. The only difficulty might be the main surfaces of traffic signs like the last three are not parallel to the image plane.   

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry  									| 
| Priority road    		| Priority road  								|
| Slippery road			| Slippery road									|
| Traffic signals	    | Bumpy Road					 				|
| Speed limit (30km/h)	| Speed limit (50km/h)     						|
| Speed limit (70km/h)	| Vehicles over 3.5 metric tons prohibited		|
| Stop               	| Speed limit (60km/h)      					|

the accuracy on these extra image sets are just 57.1%. Compared to test set accuracy 95.1%, it is too low. But what we have to notice is that, only 7 images here are tested, and because the dataset is so tiny, this 57.1% accuracy do not have any statistical significance comnpared to the former test accuray 95.1%.

+ The Prediction on the last three images are all wrong, i would like to answer why will that happen, theoretically the images are so clear and these wrong classifications shall not happen.

#### 3. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 


The top five soft max probabilities for the extra images are listed:

Image 1

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%        			| No entry 								| 
| 0%    				| Turn left ahead	|
| 0%					| No passing								|
| 0%	      			| Speed limit (60km/h)					 		|
| 0%				    | Go straight or right     						|

Image 2

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%         			| Priority road  								| 
| 0%    				| No passing									|
| 0%					| End of no passing								|
| 0%	      			| Speed limit (100km/h)				 		    |
| 0%				    | Ahead only    							    |

Image 3

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%        			| Slippery road 								| 
| 0%    				| No passing for vehicles over 3.5 metric tons	|
| 0%					| Turn left ahead								|
| 0%	      			| Speed limit (60km/h)					 		|
| 0%				    | Beware of ice/snow     						|

Image 4

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.99%         		| Traffic signals   							| 
| 0.01%     			| Bicycles crossing								|
| 0%					| General caution								|
| 0%	      			| Children crossing					 			|
| 0%				    | Speed limit (120km/h)     					|

Image 5

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.74%         		| Speed limit (50km/h)   						| 
| 0.26%    				| Speed limit (30km/h) 							|
| 0%					| Speed limit (80km/h)							|
| 0%	      			| Speed limit (60km/h)			 				|
| 0%				    | No vehicles      		     				    |

Image 6

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 72.33%         			| Vehicles over 3.5 metric tons prohibited   									| 
| 26.36%     				| Dangerous curve to the right 										|
| 0.58%					| General caution										|
| 0.49%	      			| No passing for vehicles over 3.5 metric tons					 				|
| 0.12%				    | Roundabout mandatory      							|

Image 7

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 96.36%         			| Speed limit (60km/h)   									| 
| 3.63%     				| Road work 										|
| 0.01%					| End of all speed and passing limits										|
| 0%	      			| End of speed limit (80km/h)					 				|
| 0%				    | Go straight or right      							|


Questions and Improvment:

1. i would like some suggestions of better solutions which are specially for modifing the neural network architecture.

2. i would like to konw the why the result is very bad for extra testing images at the very end, although these images are very clear.

3. Improvement Possibility : using methods like Data Augmentation to enlarge the dataset.


```python

```
