CV ---  stanford

[toc]

## lecture 2: K-neariest neighbor

image classifiction (March 7—lecture 2)

* n-neariest neighbor

* linear classification

1. **n-neariset neighbor**

distance —— L1 / L2 distance between each picture based on each piexl

Process:

Train section—just memorize all of train data, use those labled data for following prediction for each input image

predcition section— computing the dictance between input image and all of memorized labeled datas, for the `K` neariest neighbor, and vote for the prediction of input image

use cross validation to figure out the optimal value of k

> 1. drawback——the distance metric is not good to represence the difference between different image (L1/L2)

> 2. drawback——use more time to prediction than trianing

> 3. drackback—— the curse of dimensionality — need so many images to construct the high dimenson space to represent image space

&nbsp;

2. **lieaner classification**

[X] x[W] + [b]= [Prediction]

1. Drawback: can not handle some situations of non-linear separable cases

\---

\----

## Lecture 3 | Loss Functions and Optimization

#### Loss function

1. **Multi-Class SVM loss -- Hinge loss**

$$Loss_i = \sum_{j \neq y_i}^n \max{(0, s_j - s_{y_{i}}+1)}$$

---$n=$ the number of classes

&nbsp;

*this loss function means that predictive ground true label ($s_{y_{i}}$) must be larger than **$s_j+ 1$**, and then the loss of this sample will be 0; otherwise, the loss will be **$s_j - s_{y_{i}}+1$**.*

**in there, the safe margin is 1,which can be modified according to the task --- what is you requirement for the ground true prediction to be larger than the rest of class prediction.**

e.g.:

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406221834641.png" alt="image-20220406221834641" style="zoom:50%;" />

**the range of this Loss:**

$$0 <= Loss_{hinge} <= infinity$$

**Debugging trick** :

if the initial prediction for each class in each sample point, is 1 or say squal; then the initial loss will be $n-1$(in here, $n$ denotes the number of classes)

The final Loss among the dataset, can be the average of the loss of all of data points

**Square Loss and Linear Loss**:

square loss emphasizes the loss deriving from poorer predictions among all of wrong predictions, and linear one just equally estimates all of wrong predictions.

&nbsp;

***********

2. **Softmax classifier(Loss)--Multinomial Logistic Regression**

advantage: the **probability representation** of prediction:

$$P(Y=k | X = x_i )= \frac{e^{s_k}}{\sum_{j}e^{s_j}}$$

what we want is the prediction of ground true label as close to 1 as possible. therefore, use the **ground true label** to measure the preformance of the predictio in the specific data point.

therefor, for one data sample:

$$Loss_i = -log(P(Y=k | X = x_i ))=-log(\frac{e^{s_k}}{\sum_{j}e^{s_j}})$$

$$Loss = \sum_i^m Loss_i$$

-----$m$ denotes the number of data points



**Debugging trick:**

initially, the predictions among different classes are the same, let's say, 1/10 among 10 classes.

then the initial Loss will be $-\log \frac{1}{C} = \log C = \log 10$

##### Optimization

Loss derives from *n* data points in the dataset, if the dataset is large, which means the computation of loss among the entire dataset will be computional expensive; and it will be really slow;

due to the gradient descent is a linear operation, we can iterate it several times to obtian final outcome; meanwhile, if this process iterate the *n* data points(the whole data set) several time, and then will really slow the process.

**So in practice**, do not use the entire dataset to train each time; just use the subset of data to train; due to the linear property of Gradient Descent, the combination of several gradients among subset can also guarantee the performance

&nbsp;

***Stochastic Gradient Descent (SGD)***

each iteration, just sample (randomly) minibatch(32/64/128) of examples, and use those batch data to get Loss and estimate the gradient, and update weight.

##### image feature:

for CV, model need feature to represent data;

there could be several feature represences. the common measure is **_concatenate_** (flatten) all of features and then feed this concatecated features into model.

image Feature:

* color

* edge, direction

* visual word - bag of words

&nbsp;

Tradictional idea in CV,just extract image features, and use those feature to feed model,update weights in the model (the feature of image is fixed).

Now, CNN, which extract feature from image automatically, furthermore, the process of extraction, and weight during extracting will all be trained as well, same as the rest weights in the neural network does.





## Lecture 4 | Introduction to Neural Networks

***back propagation---chain rule***

**sigmoid function**

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

derivative

$$\frac{\partial\sigma(x)}{\partial x} = \frac{e^{-x}}{(1+e^{-x})^2} = (1-\sigma(x))\sigma(x)$$

&nbsp;

pattern in back propagation flow:

* **add gate: gradient distribution**

----distribution gradient from the upper flow to all nodes

* **max gate: gradient router**

---only the one node which has be chosen for following compution get the gradient, the rest of nodes get 0.

* **mult(*) gate: gradient switcher**

---- switch the values between nodes, and get the new gradient combining the gradient flowing from upper layer and the other value from other node in the current layer.

each element in gradient matrix represent the contribution of current factor for the final output



## Lecture 5 | Convolutional Neural Network

#### convolutional layer:

**the whole process:** image --> **convolutional layer**-->full connected layers--> prediction

origianl image--

**--> filters --(dot prodcut)--> features layers---> activation layer(RELU) ----> pooling

--> filters --(dot prodcut)--> features layers---> activation layer(RELU) ----> pooling

--> filters --(dot prodcut)--> features layers---> activation layer(RELU) ----> pooling

-->.... --**

--> full connected layers--> predcitions

&nbsp;

**output size of feature layer:**

$N$ -- dimension of original images--7

$F$ -- dimension of filter --3

the size of generated feature image

$$(N-F)/stride + 1 = size-Of-Feature Image$$

$$stride:1--> (7-3)/1+1 = 5--> 5*5$$

$$stride:2--> (7-3)/2+1 = 3 --> 3*3$$

$$stride:3 --> (7-3)/3+1 = 2.33 :/$$



in practice, use **`zero pads the border`**, to generate targeted size of filter output;

for example, use zero padding to **`maintain the original image size`** after the filtering processing;

without zero padding, the generated images will **shrink quickly**; shrink too fast, and loss much information

***for example:***

with padding 1, (stride 1) , filter works 3 times:

$$(7+(1*2)-3)/1 +1 =7->7*7-> (7+(1*2)-3)/1 +1 = 7*7....$$

without padding 1, (stride 1), filter works 3 times:

$(7-3)/1+1 = 5-> 5*5->(5-3)/1+1 = 3-> 3*3..->1*1$



**The number of parametes in convolutional netowork:**

input->32x32x3, filter 10 5x5:

**So the total number of parameters is:**

$$5 \times 5 \times 3 + 1(bais) = 76 \left(per-filter\right)$$

$$76 \times 10 = 760 \left(for-the-whole-network\right)$$

&nbsp;

#### pooling layer

**pooling layer makes the image smaller and more manageable.**-- just reduce the 2-d size, do not affect the depth of image;

\* max pooling-- filter size and stride are hyperparameters

commonly, there is not overlap while pooling processing



## Lecture 6 | Training Neural Networks I

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222400996.png" alt="image-20220406222400996" style="zoom:80%;" />

#### **activation function:**

* 0<=sigmoid<=1 ----> no zero-mean function(problem)

* -1<=tanh<=1

* ReLU

* Leaky ReLU= max(0.01x , x)

* parametric ReLU--PReLU = max(ax, x)

* Exponential Linear units--ELU

* Maxout = $max(w_1^T x+b_1,w_2^T x+b_2 )$

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222440989.png" alt="image-20220406222440989" style="zoom:80%;" />



#### data preprocess

zero-mean data distribution good for optimization.

normalizing data make the gradient descent more efficient



#### initializtion

> in tha small network, initialize the weight using gausian distrubtion is fine.

> however, in deep network, due to the value range of weight is centrialize around 0, then the output of each layer will become smaller and smaller. Therefore, in back propagation, the gradient of each layer will be smaller and smaller from upper stream to lower strain, and finally, the gradient of first initial layer may vanish.

> same problem when the initial weight is too large, the output will become larger and larger. and after activation function will generate saturated values(-1,1 for sigmoid), then the gradient will be `0` in back propagation

> both of those cases is devastating for training



#### Batch Normalization

**guarantee the output data distribution, to make sure the performance of back propagation.**

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222549657.png" alt="image-20220406222549657" style="zoom:80%;" />

**batch normalization works in each neuron. Due to each operation in one neuron process one mini-batch data one time , so can use those data samples in the batch to do normalization in each dimension among `n` data**

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222611597.png" alt="image-20220406222611597" style="zoom:80%;" />

**$\beta$ and $\gamma$ are learned parameter**, after normalization, the output is a gaussian distribution. while which does not gaurantee that it is the optimal data distribution. therefore introduce $\beta$ and $\gamma$ to increase flexibility. Of course, the final $y_i$ can be identical with normilized $x_i$. but again, this process just increase the flexibility, and let the model makes the best decision(use the optimal data distribution )

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222633595.png" alt="image-20220406222633595" style="zoom:80%;" />

!<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222652135.png" alt="image-20220406222652135" style="zoom:50%;" />



#### whole process of cv

1. normalized data

2. choose architecture

3. forward one thorogh time--make sure the loss is reasonable

4. train--first train on the small data , make sue the capability of overfitting

5. figure out the optimal `learing rate`, with small regularization

6. use Cross valication to find good hyperparameters



## Lecture 7 | Training Neural Networks II

* Fancier optimization

* regularization

* transfer learning



Problem of SGD:

1. zigzagging to converge(slow / inefficient)

2. local optimum/ saddle point----saddle point it more common in high dimension

3. noise in `S` of SGD, the fault gradient descent of the object funtiuon

**improvement:**

SGD + Monmentum

use `velocity` replace gradient

the initialization of velocity is `0`

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222813529.png" alt="image-20220406222813529" style="zoom:60%;" />

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222839687.png" alt="image-20220406222839687" style="zoom:50%;" />

--------

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222901958.png" alt="image-20220406222901958" style="zoom:50%;" />

--------

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406222937797.png" alt="image-20220406222937797" style="zoom:67%;" />



#### learning rate

decay learning rate (the second tuning parameters --- non-decay version should be tried first)

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406223011624.png" alt="image-20220406223011624" style="zoom:60%;" />

usually, too large for deep neural network

so -- quasi-Newton(BGFS)

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406223054014.png" alt="image-20220406223054014" style="zoom:67%;" />



#### regularization

1. L1

2. L2

3. Elastic Net

4. **Dropout**

5. **batch normalization**

6. **Data augmentation**





## Lecture 8 | Deep Learning Software

GPU is good at paralle computation  and matrix compution 



pytorch is similar with numpy 

the difference is the numpy cannot run in GPU

the frequently transition between CPU and GPU is expensive 



tensor in pytorch is a nparray running in GPU



## Lecture 9 | CNN Architectures 

* AlexNet ---8 layers

* VGG----19 layers---small filter,deeper networks
  * Deeper filter : more non-linearities

* GoogleNet --- 22 layers
  * inception module
    * desigen a good local network topology(network within a network) and then stack these modules on the top of each other
    * <img src="/Users/liu/Library/Application Support/typora-user-images/image-20220406184659932.png" alt="image-20220406184659932" style="zoom:25%;" />

* ResNet ---- 152 layers
  * deeper network actually do not gaurantee the better performance even in the training dataset
  * Hypothesis: the problem is a optimization problem, deeper models are hard to optimize 
  * a solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping 





## Lecture 10 | Recurrent Neural Networks



## Lecture 13 ｜ Generative Model

**unsupervised learning---- learn the underlying data features**

1. strength : cheap for training 
2. Weakness: how to inteprate the underlying data distribution 



### Generative Model 

Data --> model --> distribution -->new samples 

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220503150511713.png" alt="image-20220503150511713" style="zoom:35%;" />





### PixelRNN/CNN

1. PixelRNN (2016)

   1. <img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 3.48.53 PM.png" alt="Screen Shot 2022-05-03 at 3.48.53 PM" style="zoom:33%;" />
   2. Drawback: slow

2.  PixelCNN (2016)

   1. <img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 3.50.04 PM.png" alt="Screen Shot 2022-05-03 at 3.50.04 PM" style="zoom:33%;" />

   2. training is much faster than PixelRNN (can paralleize convolutions since context region values known from training images)

   3. Generation must still proceed sequentially --> still slow

      

### Variational Autoencoders (VAE)

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 3.57.50 PM.png" alt="Screen Shot 2022-05-03 at 3.57.50 PM" style="zoom:33%;" />



1. Autoencoder:
   * Input --Encoderer---> latent feature layer --Decoder--> output layer  ( L2 loss : input and output)

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.00.50 PM.png" alt="Screen Shot 2022-05-03 at 4.00.50 PM" style="zoom:33%;" />

​			after train, link to the downflow task to do some supervised learning task.

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.01.57 PM.png" alt="Screen Shot 2022-05-03 at 4.01.57 PM" style="zoom:33%;" />



* Variational autoencoder

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.03.56 PM.png" alt="Screen Shot 2022-05-03 at 4.03.56 PM" style="zoom:33%;" />

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.09.07 PM.png" alt="Screen Shot 2022-05-03 at 4.09.07 PM" style="zoom:30%;" />



<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.12.22 PM.png" alt="Screen Shot 2022-05-03 at 4.12.22 PM" style="zoom:35%;" />

<img src="/Users/liu/Library/Application Support/typora-user-images/image-20220503161458601.png" alt="image-20220503161458601" style="zoom:33%;" />



overall picture

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.18.48 PM.png" alt="Screen Shot 2022-05-03 at 4.18.48 PM" style="zoom:33%;" />



After training -- generate data

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.21.59 PM.png" alt="Screen Shot 2022-05-03 at 4.21.59 PM" style="zoom:33%;" />

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.24.34 PM.png" alt="Screen Shot 2022-05-03 at 4.24.34 PM" style="zoom:33%;" />



### GAN

### 	-- Generative Adversarial Network

<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-03 at 4.34.56 PM.png" alt="Screen Shot 2022-05-03 at 4.34.56 PM" style="zoom:25%;" />





<img src="/Users/liu/Library/Application Support/typora-user-images/Screen Shot 2022-05-04 at 10.37.47 AM.png" alt="Screen Shot 2022-05-04 at 10.37.47 AM" style="zoom:50%;" />



after training, use generator network to generate new image

[GAN trick](https://github.com/soumith/ganhacks)



### recap

![image-20220504112233061](/Users/liu/Library/Application Support/typora-user-images/image-20220504112233061.png)





## Reinforcement Learning 

