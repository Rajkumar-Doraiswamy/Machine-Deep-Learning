# Last amended: 24th June, 2019
# Ref:  https://github.com/keras-team/keras/issues/4301
#       https://keras.io/applications/#vgg16
#       https://keras.io/applications/
#
# Objectives:
#	     i)  Experimenting with Very Deep ConvNets: VGG16
#        ii) Peeping into layers and plotting extracted-features
#
#  Make tensorflow as backend
#  =========================

#    cp /home/ashok/.keras/keras_tensorflow.json  /home/ashok/.keras/keras.json
#    cat /home/ashok/.keras/keras.json
#    source activate tensorflow
#    ipython
#    OR, on Windows
#    > conda activate tensorflow_env
#    > atom

## Expt 1
# ==========

# 1.0 Import libraries
%reset -f

# 1.0 Import VGG16 function from vgg16 module. Other modules are:
#     resnet50, inception_resnet_v2, inception_v3, vgg16, vgg19, xception

from keras.applications.vgg16 import VGG16

# 1.1 With every deep-learning architecture, keras has a function
#     to automatically process image to required dimensions

from keras.applications.vgg16 import preprocess_input

# 1.2 Keras image preprocessing. Import image module
#      https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
from keras.preprocessing import image

# 1.3
import numpy as np
import pylab

# 1.4 PIL: Python Image library
#     Like jpg, PIL also has another image format

from PIL import Image as im

###################### AA. Image processing ##########################

# 3.1 Where is my image?
#img_path = "C:\\Users\\ashok\\Desktop\\chmod\\Documents\\deeplearning\\data\\cat.jpg"
img_path = '/home/ashok/Images/cat/cat.png'

# 3.2 Read image in Python Image Library (PIL) format
#     using keras image module

img = image.load_img(img_path)

# 3.3 Its has a PIL format
type(img)                  # PIL.Image.Image
                           # It is one compressed format for images
						   #  and not an array of pixel intensities

# 3.4
img       # Maybe it calls show() method automatically

# 3.4 Some examples of image manipulation using PILLOW library
#     Ref: http://pillow.readthedocs.io/en/3.1.x/handbook/tutorial.html
img.size
img.show()
img.save("/home/ashok/abc.png")
img.rotate(45).show()
img.transpose(im.FLIP_LEFT_RIGHT).show()
img.transpose(im.FLIP_TOP_BOTTOM).show()


# 3.4 Transform PIL image to numpy array
#     Use image module
x = image.img_to_array(img)
x.shape                              # 320 X 400 X 3
                                     # Last index of 3 is depth

# 4. For processing an image in VGG16, shape of image should be:
#         [samples, height, width, no_of_channels ]
#    So we need to transfrom the img-dimensions. We can use
#    np.newaxis() as follows:
#    https://stackoverflow.com/a/25755697

x[np.newaxis, :, :, :].shape
x = x[np.newaxis, :, :, :]


# 4.1.1 OR do it this way
#       reshape to array rank 4
#  x = x.reshape((1,) + x.shape)


# 4.2 About preprocess_input, pl see
#     https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
#     Some models use images with values ranging from 0 to 1.
#     Others from -1 to +1. Others use the "caffe" style,
#     that is not normalized, but is centered.
#     The preprocess_input function is meant to adjust your image
#     to the format the model requires.

x
x[0,80,90,1]                # 136 (pixel intensity)
x = preprocess_input(x, mode = 'tf')    # 'tf' : tensorflow
x[0,80,90,1]                # 0.06666 (normalized pixel intensity)
x[0, :2, :2 , :2]           # Have a look at few data-points
x.shape                     # (1, 320, 400,3) shape remains same

###################### BB. Model Building ##########################
# 5.0 Create VGG16 model
#     Use the same weights as in 'imagenet' experiment
#     include_top: F means do not include the 3 fully-connected layers
#      at the top of the network.
#     Model weights are in folder ~/.keras/models
#     OR on Windows on: C:\Users\ashok\.keras\models\
model = VGG16(
	          weights='imagenet',
	          include_top=False
	          )

# 5.1 Get features output from all filters
#      We make predictions considering all layers, ie till model end
#      We have jumped 'model.compile' and 'model.fit' steps
#      Why? Read below.
"""
Why no model compilation and fitting?
=====================================
	A model is compiled for setting its configuration such
	as type of optimizer to use and loss-function to use.
	Given these, it is 'fitted' or 'trained' on the dataset
	to learn weights.
	But, if weights are already fully learnt, as is the case in
	this example then there is no need to compile and 'fit'
	the model.
	We straightaway move to 'prediction' using our data so to say
	as 'test' data.
"""
features = model.predict(x)

# 5.2 So how many features or tiles at the end?
features.shape                   # (1,10,12,512) Tile size: (10,12)
                                 # 1       :no of samples
								 # (10,12) :feature ht and width;
								 # 512     :Depth/number of channels
								 # See below, model summary

# 4.3 Number '512' matches with the last layer of model (block5_pool):
model.summary()

				 #  1      =    One batch input,
				 # (10,12) =    filter size
				 #  512    =    No of filters


###################### CC. Display a feature ##########################

# 5 Display output of a specific feature.  Try 10, 115, 150, 500
pic=features[0,:,:,500]         # (1,10,12,512) => initial index can only be 0
pylab.imshow(pic)               # Image of 10 X 12 total no of squares/pixels
pylab.gray()                    # Gray image
pylab.show()


######################  ################################################  ##########################
#           *************  Features at some intermediate layer *****************
######################  ################################################  ##########################

## Expt 2
#==========
# 	Objective:
#             Extract and display features from any arbitrary
#             intermediate convolution layer in VGG16
#
# For layer names of VGG16, pl see:
#     https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py

# 1. Call libraries
%reset -f
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# 1.1 For image processing
from keras.preprocessing import image
import matplotlib.pyplot as plt

# 1.2 Help create a model uptil some intermediate layer
#     Import 'Model' class
from keras.models import Model

# 2. Create base model
#    include_top = False, abandons the last three FC layers

base_model = VGG16(
	               weights='imagenet',
	               include_top=False    # Exclude FC layers
	               )


# 2.1 What is VGG16 architecture?
base_model.summary()

# 2.2 To see complete model, including FC layers, try following:
#     WARNING: Be warned, there will be fresh download of model
#              weights.
#             Generally FC layers have a very large number of weights
#             Do it only you have sufficient RAM allocated to your VM
# full_model = VGG16(weights='imagenet', include_top=True)
# full_model.summary()

# 2.2 How many layers in VGG16
len(base_model.layers)      # Total layers: 19

# 2.3 Type of 'layers' object
type(base_model.layers)    # List of layers



# 3.  Access first node ie input layer node

inp = base_model.layers[0]

# 3.1 What is this layer's name

inp.name              # NAME MAY BE DIFFERENT IN EACH CASE.

# 4.  Access first block-convolution layer node: 'block1_conv1'
#     based on its name. An instance of layer is returned
#
by = base_model.get_layer('block1_conv1')

# 4.2 This returns input to node
by.input                   # Input tensor to this layer
                          # <tf.Tensor 'input_4:0' shape=(?, ?, ?, 3) dtype=float32>
# 4.3 Output of node
by.output 				  # by.output is equivalent, internally, to a series
						  #  of nested functions, such as: g(f(h(a.input)))
						  #   where each one of the functions is a layer-function
						  #    In short by.out contains information as to how it has
						  #     been arrived at.
                          # <tf.Tensor 'block1_conv1_3/Relu:0' shape=(?, ?, ?, 64) dtype=float32>
						  # shape=(?, ?, ?, 64) =  (samples, img_ht, img_width, channels)


# 5    Instantiate model uptil required node/layer
#      Model extends from 'input' of one node  to 'output'
#      of another node

model = Model(inputs=inp.input , outputs= by.output)

# 5.1 Have a look at this intermediate model
model.summary()

# 4.0 Image processing
img_path = '/home/ashok/Images/cat/cat.png'
# 4.1
img = image.load_img(img_path)
# 4.2
x = image.img_to_array(img)
x.shape                                # (320, 400, 3)
# 4.3
x = x[np.newaxis,:,:,:]               # (1, 320, 400, 3)
# 4.4
x = preprocess_input(x, mode = 'tf')
# 5 Feed 'x' to input and predict 'output'
#   This is an intermediate (first) layer of vgg16
# 5.1
block1_conv1_features = model.predict(x)
block1_conv1_features.shape           #  (1, 320, 400, 64)


# 5.2 See nine features in various filters
for i in range(9):
	plt.subplot(3,3, 1 + i)
	im = block1_conv1_features [0,:,:,i+20]    # Layers from index 20 (ie 21st) to 28 (ie 29th)
	plt.imshow(im, cmap=plt.get_cmap('gray'))
plt.show()
##########################
