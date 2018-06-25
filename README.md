# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Architecture
Using the default VGG-16 pre-train model and convert to fully convolutional neural network to perform semantic segmentation.

```
# 1x1 convolution of vgg layer 7
  layer7a_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                 padding= 'same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
  # upsample
  layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out, num_classes, 4,
                                           strides= (2, 2),
                                           padding= 'same',
                                           kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
  # make sure the shapes are the same!
  # 1x1 convolution of vgg layer 4
  layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                 padding= 'same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
  # skip connection (element-wise addition)
  layer4a_out = tf.add(layer4a_in1, layer4a_in2)
  # upsample
  layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, num_classes, 4,
                                           strides= (2, 2),
                                           padding= 'same',
                                           kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
  # 1x1 convolution of vgg layer 3
  layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                 padding= 'same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
  # skip connection (element-wise addition)
  layer3a_out = tf.add(layer3a_in1, layer3a_in2)
  # upsample
  nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16,
                                             strides= (8, 8),
                                             padding= 'same',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
```


### Optimizer
Loss function: Cross-entropy Optimizer: Adam optimizer

### Training Hyperparameters

|  Input          |    Value |
|  -----          |  ------- |
|  keep_prob      |  0.5     |
|  learning_rate  |  0.0009  |
|  epochs         |  100     |
|  batch_size     |  10      |

### Results

![](images/images1.png)

![](images/images2.png)

![](images/images3.png)

![](images/images4.png)

![](images/images5.png)

![](images/images6.png)

![](images/images7.png)

![](images/images8.png)

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
