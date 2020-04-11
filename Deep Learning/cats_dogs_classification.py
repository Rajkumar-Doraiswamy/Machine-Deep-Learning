# Last amended: 24th June, 2019
# My folder: E:/cats_and_dogs/data
# Data folder:  '/home/ashok/Images/cats_dogs/
# VM: lubuntu_deeplearning
# Objectives:
#           i)  Building powerful image classification models using
#               very little data
#           ii) Predicting cats and dogs--Kaggle
#               https://www.kaggle.com/c/dogs-vs-cats
#          iii) Calcualting model weights, stage-by-stage
#
# ************************************
# USE THEANO NOT TENSORFLOW
# ************************************
# Ref:
#  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


'''

A. Arrange your data first
==========================

    Download data from: https://www.kaggle.com/c/dogs-vs-cats/data .
    Unzip train.zip and arrange its files as follows:
        data/
            train/
                dogs/                      1000
                    dog001.jpg
                    dog002.jpg
                    ...
                    dog1000.jpg
                cats/                      1000
                    cat001.jpg
                    cat002.jpg
                    ...
                    cat1000.jpg
           validation/
               dogs/                        400
                   dog1001.jpg
                   dog1002.jpg
                   ...
                   dog1400.jpg
              cats/                         400
                  cat1001.jpg
                  cat1002.jpg
                  ...
                  cat1004.jpg

    So we are picking up only 1000 files of each category for training.
    Arrangement of (only) 'training' files in this fashion has an advantage that
    keras automatically knows which images are of cats and which images are
    of dogs. It does automatic labeling of images; we do not have to specify
    explicitly in the code for building training model. This automatic labelling
    is done by ImageGenerator.

B. Training steps are as follows:
=================================

    1. Arrange training files as above. (Our total samples: nb_train_samples = 2000)
    2. Arrange validation files as above (Valid Samples: nb_validation_samples = 800)
    3. Specify location of all train folder and vaidation folder
    4. Depending upon your backend (tensorflow/theano) decide
       the shape/format of your input image arrays. This is needed in CNN modeling.
    5. Build the CNN model & compile it

    6. Use ImageDataGenerator to augment train images. This is in two steps:
        i)  Create an object with configuration of possible changes in any image
        ii) Use the object to create an iterator with following further configuration:
                a)  Create iterator using flow(), '.flow_from_directory()' method
            In '.flow_from_directory()', specify:
                b)   Where is the directory of your images?
            In ''.flow()' specify X_train, y_train
            Further in both cases:
                b)  Do you also want to resize images, if so, specify these
                c) What batch-size to augment and model at a time;depends upon RAM
                d)  Is classification binary or categorical?

    7. Use ImageDataGenerator to augment validation images. Again the two steps
       as above. But we only resize validation images.
    8. Begin extracting images or training using the iterator fit_generator():
        CNN fit_generator() takes these arguments:
        i)   train-data-iterator (batch-wise source of train images)
        ii)  validation-data generator (batch-wise source of validation images)
        iii) no of epochs


   9. After training has finished, save model weights to a '.h5' file and also
      save model configuration to a json file.

   ------------
   Later, maybe, after some time
   10.Unzip test data file in a folder (within another folder. This is impt.).
   11.Configure test Image Data Generator
   12.Use above configuration and test-folder address, to create a test generator
   13.Load saved cnn model and load network weights in this model from saved h5 file
   14.Use predict_generator() to make predictions on test_generator.
   15.Evaluate predictions

C. About keras backend:
=======================

    The default keras configuration file is in folder:
        C:\Users\ashokharnal\.keras.  It looks like as below.
        The configurtion is as per the installed backend on your machine:
        tensorflow, theano or CNTK

            {
                    "image_data_format": "channels_last",
                    "epsilon": 1e-07,
                    "floatx": "float32",
                    "backend": "tensorflow"
                    }

            "epsilon" is used instead of zero when division is by zero. 'floatx'
            specifies the datatype that keras will process.
            For 2D data (e.g. image), "channels_last" assumes (rows,cols,channels)
            while "channels_first" assumes (channels,rows,cols)
            (channels stand for RGB colour channels)

D. Prerequisites:
=================
    Before attempting this problem, pl study Image Augmentation in Moodle at
    http://203.122.28.230/moodle/course/view.php?id=11&sectionid=166#section-9

E. Note
=======
    This is a full code from building model to making predictions for test data.
    AUC is very less as no. of training epochs are just 5. The training consumes
    time but very less memory (around 50%) on an 8GB machine. Vary batch size
    to control memory usage.


F. Make theano as backend
=========================

#    cp /home/ashok/.keras/keras_theano.json  /home/ashok/.keras/keras.json
#    cat /home/ashok/.keras/keras.json
#    source deactivate tensorflow
#    source activate theano
#    ipython
	In file:  ~/.keras/keras.json, set:

	"backend": "theano",    # instead of "tensorflow"

         Also the following environment variable needs to be set in bashrc:

	 export "MKL_THREADING_LAYER=GNU"

'''

#%%                                A. Call libraries

#        $ source activate theano
#        $ ipython
# OR in Windows
#       > conda activate tensorflow_env
#       > atom

# 0. Release memory
%reset -f

# 1.0 Data manipulation library
import pandas as pd

# 1.1 Call libraries for image processing
#     Another preprocessing option is text and sequence
from keras.preprocessing.image import ImageDataGenerator

# 1.2, Libraries for building CNN model
#      A model is composed of sequence of layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# 1.3.Keras has three backend implementations available: the TensorFlow,
#    the Theano, and CNTK backend.
from keras import backend as K

# 1.4 Save CNN model configuration
from keras.models import model_from_json

# 1.5 OS related
import os

# 1.6 For ROC plotting
import matplotlib.pyplot as plt

# 1.7
import numpy as np
from sklearn import metrics
import time
from skimage import exposure           # Not used
from PIL import Image                  # Needed in Windows


#%%                            B. Define constants

# 2. Our constants
# 2.1 Dimensions to which our images will be adjusted
img_width, img_height = 150, 150

# 2.2 Data folder containing all training images, maybe in folders: cats and dogs
train_data_dir = '/home/ashok/Images/cats_dogs/train'
#train_data_dir ="C:\\Users\\ashok\\Desktop\\chmod\\Documents\\deeplearning\\data\\train"

# 2.3 What is the total number of training images
#      that should be generated (not what are available)
nb_train_samples = 2000   # Actual: 1000 + 1000 =    2000


# 2.4 Data folder containing all validation images

validation_data_dir = '/home/ashok/Images/cats_dogs/validation'
#validation_data_dir = "C:\\Users\\ashok\\Desktop\\chmod\\Documents\\deeplearning\\data\\validation"

# 2.5 What is the total no of validation samples that should
#     be generated?
nb_validation_samples = 800   # Actual: 400 + 400 =  800


# Some hyperparameters

# 2.6 Batch size to train at one go:
batch_size = 16             # No of batches = 4000/125 = 32
                             # So per epoch we have 32 batches

# 2.7 How many epochs of training?
epochs = 5            # For lack of time, let us make it just 5.

# 2.8 No of test samples
test_generator_samples = 300

# 2.9 For test data, what should be batch size
test_batch_size = 25    # This is different from training batch size


# 3. About keras backend
# 3.1 Can get backend configuration values, as:
K.image_data_format()          # Read .keras conf file to findout
K.backend()


# 3.2 What is our backend and input_shape? Decide data shape as per that.
#     Depth goes last in TensorFlow back-end, first in Theano
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:                                           # So, Tensorflow!
    input_shape = (img_width, img_height, 3)


#%%                         C. Define CNN Model


# 4. Create convnet model
#    con->relu->pool->con->relu->pool->con->relu->pool->flatten->fc->fc

# 4.1   Call model constructor and then pass on a list of layers
model = Sequential()
# 4.2 2-D convolution layers with 32 filters and kernel-size 3 X 3
#         Default strides is (1, 1)
#         help(Conv2D)
#

model.add(Conv2D(
	             filters=32,                       # For every filter there is set of weights
	                                               # For each filter, one bias. So total bias = 32
	             kernel_size=(3, 3),               # For each filter there are 3*3=9 kernel_weights
	             strides = (1,1),                  # So output shape will be 148 X 148 (W-F+1).
	                                               # Default strides is 1 only
	             input_shape=input_shape,          # (150,150,3)
	             use_bias=True,                     # Default value is True
	             padding='valid',                   # 'va;id' => No padding. This is default.
	             name="Ist_conv_layer"
	             )
         )

# 4.3 So what have we done? Can you explain?
#     Total weights = (kernel_weights) * RGB_channel * (filters)  + ToalNoBias
model.summary()


# 4.4 For each neuron in the convolved network, assign an activation function
model.add(Activation('relu'))

# 4.5
model.summary()


# 4.6 pool_size:  max pooling window size: (2,2)
#     Default stride for pool-layer is same as pool_size
#     Here: 2 across and 2 down ie (2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4.7
model.summary()


# 4.8 Input shape is inferred. Default strides is 1.
#     Note: Activation is specified here only
#     input_shape from top = 74 X 74 X 32
model.add(Conv2D(32,
                (3, 3),
                activation = 'relu',
                name = "IInd_con_layer"))

# 4.9 So how many parameters now?
#     Total weights = (kernel_weights) * (filters_from_earlier_conv) * (filters)  + ToalNoBias
model.summary()


# model.add(Activation('relu'))
# 4.10 Add another pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4.11 Summary?
model.summary()


# 4.12 Add another conv layer but with 64 filters
#      Total weights = (kernel_weights) * (filters_from_earlier_conv) * (filters)  + ToalNoBias

model.add(Conv2D(64, (3, 3), name = "IIIrd_conv_layer"))

# 4.13
model.summary()

# 4.14
model.add(Activation('relu'))

# 4.15
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4.16 Flattens the input. Does not affect the batch size.
#      It merely flattens the earlier layer without adding any weight
#     See summary() next
model.add(Flatten(name = "FlattenedLayer"))

# 4.17
model.summary()



# 4.18 Dense layer having 64 units
#      dimensionality of the output space.
#      Total weights = hidden_neurons * input_size + bias_foreach_hidden_neurons
#      64 * 18496 + 64
#      Most number of weights come from this layer

model.add(Dense(64))

# 4.19
model.summary()


# 4.20
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 4.21
model.summary()


# 4.22 Dense layer having 1 unit
#      dimensionality of the output space.
#      Weights = No of input layers + bias (64+1)
model.add(Dense(1))

# 4.23
model.summary()

# 4.24
model.add(Activation('sigmoid'))    # tanh vs sigmoid? See Stackoverflow

# 4.25 Compile model
model.compile(loss='binary_crossentropy',  # Metrics to be adopted by convergence-routine
              optimizer='rmsprop',         # Strategy for convergence?
              metrics=['accuracy'])        # Metrics, I am interested in



#%%                            D. Create Data generators


## 5. Image augmentation
# 5.1 Define a preprocessing function
def preprocess(img):
	# Histogram equalization
	# WITHOUT IT RESULTS ARE BETTER
	# http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    # img_eq = exposure.equalize_hist(img)
    # Do something more with image
    return img


# 5.2 Config1: Augmentation configuration for training samples
#     Instantiate ImageDataGenerator object with requisite configuration

tr_dtgen = ImageDataGenerator(
                              rescale=1. / 255,      # Normalize colour intensities in 0-1 range
                              shear_range=0.2,       # Shear varies from 0-0.2
                              zoom_range=0.2,
                              horizontal_flip=True,
                              preprocessing_function=preprocess
                              )


# 5.3 Config2: Create iterator from 'train_datagen'.
#     We use flow() or flow_from_directory() methods to further
#     configure and return an iterator object.
#     See at the end of code: Differences between flow() and flow_from_directory

train_generator = tr_dtgen.flow_from_directory(
                                               train_data_dir,       # Data folder of cats & dogs
                                               target_size=(img_width, img_height),  # Resize images
                                               batch_size=batch_size,  # Return images in batches
                                               class_mode='binary'   # Output labels will be 1D binary labels
                                                                     # [1,0,0,1]
                                                                     # If 'categorical' output labels will be
                                                                     # 2D OneHotEncoded: [[1,0],[0,1],[0,1],[1,0]]
                                                                     # If 'binary' use 'sigmoid' at output
                                                                     # If 'categorical' use softmax at output

                                                )

# 5.4 Augmentation configuration we will use
#     for validation. Only rescaling of pixels

val_dtgen = ImageDataGenerator(rescale=1. / 255)



# 5.4.2 validation data

validation_generator = val_dtgen.flow_from_directory(
                                                     validation_data_dir,
                                                     target_size=(img_width, img_height),   # Resize images
                                                     batch_size=batch_size,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     )



#%%                           E. Fit model & save CNN network weights


## 6. Model fitting

# 6.1 Manual process of fitting:

start = time.time()
for e in range(epochs):
    print('Epoch', e)
    steps = 0
    for x_batch, y_batch in train_generator:
        model.fit(x_batch, y_batch)
        steps += 1
        if steps >= 2:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

end = time.time()
(end - start)/60

# 6.2 fit_generator() directly pulls data from iterators
start = time.time()
history = model.fit_generator(
                              generator = train_generator,          # First argument is always training data generator
                              steps_per_epoch=nb_train_samples // batch_size, # How many batches per epoch?
                                                                              # Can be any number as generator loops indefinitely
                              epochs=epochs,                        # No of epochs
                              validation_data=validation_generator, # Get validation data from validation generator
                              verbose = 1,                          # Do not be silent
                              validation_steps=nb_validation_samples // batch_size
                              )

end = time.time()
(end - start)/60

# 7.0 Model evaluation

# 7.1 USing generator
model.evaluate_generator(validation_generator,
                         steps = 2
                         )



# 7.2 Manually, per batch
steps = 0
result = []
start = time.time()
for x_batch, y_batch in validation_generator:
        result.append(model.evaluate(x_batch, y_batch))
        steps += 1
        if steps >= 2:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

end = time.time()
(end - start)/60

# 7.2.1
result


# 8.0 Make predictions

# 8.1 Using generator
pred = model.predict_generator(validation_generator, steps = 2)

# 8.2 Manually
pred = []
steps = 0
start = time.time()
for x_batch, y_batch in validation_generator:
        pred.append(model.predict(x_batch))
        steps += 1
        if steps >= 2:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

end = time.time()
(end - start)/60

# 8.2.1
pred

################ I am done ##########################
####### End of class #####



# 7. Verify your results like here
#    Though accracy is very less
im = test_generator    # Get iterator
images = next(im)      # Get images
images.shape         # (25, 150, 150, 3)
# 7.1 Make predictions
results = model.predict(images)
results               # Probability values

# 7.2 Plot the images and check with
#     results
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
plt.figure(figsize= (10,10))
for i in range(results.shape[0]):
    plt.subplot(5,5,i+1)
    imshow(images[i])

plt.show()


####### End in class #####

## 7. Model saving
# 7.1 Install h5py using Anaconda
# 7.2 Save CNN model weights to a file
#     The h5py package is a Pythonic interface to HDF5 binary data format.
#     It lets you store huge amounts of numerical data, and easily manipulate
#     that data from NumPy. For example, you can slice into multi-terabyte
#     datasets stored on disk, as if they were real NumPy arrays.
#     Thousands of datasets can be stored in a single file, categorized and
#     tagged however you want.

os.system('rm -rf  /home/ashok/useless')
os.mkdir("/home/ashok/useless")
model.save_weights('/home/ashok/useless/first_try_theano.h5')

#os.system('rm -r c:\\users\\ashok\\useless')
#os.mkdir("c:\\users\\ashok\\useless")
#model.save_weights("c:\\users\\ashok\\useless\\first_try.h5")


os.getcwd()   # Where are these saved: 'C:\\Users\\ashokharnal'


# 7.3 Save your CNN model structure to a file, cnn_model.json
#  Get your model in json format
cnn_model = model.to_json()
cnn_model


# 7.4 Now save this json formatted data to a file on hard-disk
#     File name: cnn_model.json. File path: check with setwd()
# 7.4.1. Open/create file in write mode

json_file = open("/home/ashok/useless/cnn_model.json", "w")
#json_file = open("c:\\users\\ashok\\useless\\cnn_model.json", "w")


# 7.4.2 Write to file
json_file.write(cnn_model)
# 7.4.3 Close file
json_file.close()



#%%                          F. Load model and model weights


## 8. Later

# 8.1 Open saved model file in read only mode
#     Just
# os.chdir("C:\\Users\\ashokharnal")

json_file = open('/home/ashok/useless/cnn_model.json', 'r')
#json_file = open("c:\\users\\ashok\\useless\\cnn_model.json", "r")

loaded_model_json = json_file.read()
loaded_model_json            # Model structure in file: loded_model_json
json_file.close()

# 8.2 Create CNN model from the file: loaded_model_json
cnn_model = model_from_json(loaded_model_json)


# 8.3 load saved weights into new model
cnn_model.load_weights("/home/ashok/useless/first_try_theano.h5")
#cnn_model.load_weights("c:\\users\\ashok\\useless\\first_try.h5")


# 8.4 Compile the model. Same way as was done earlier
cnn_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


#%%                         G. Make predictions on test data

# 9 Where is the directory which contains ANOTHER directory
#   containing your test images
#test_data_dir = validation_data_dir = 'E:/cats_and_dogs/test'

test_data_dir =   '/home/ashok/Images/cats_dogs/test'
#test_data_dir = "C:\\Users\\ashok\\Desktop\\chmod\\Documents\\deeplearning\\data\\test"



# 9.1 Augmentation configuration for test dataset.
#     Only rescaling as we did for validation data
test_datagen = ImageDataGenerator(rescale=1. / 255)


# 9.2 Create test data generator
test_generator = test_datagen.flow_from_directory(
        test_data_dir,                         # Which folder has test data
        target_size=(img_width, img_height),   # Resize images
        batch_size = test_batch_size,            # batch size to augment at a time
        class_mode=None)                       # Data has binary classes



# 10. Make predictions from loaded model. Takes few seconds.
start = time.time()
predictions = cnn_model.predict_generator(
        test_generator,
        steps= int(test_generator_samples/float(test_batch_size)), # all samples once
        verbose =1
        )
end  = time.time()
(end-start)/60

# 10.1 OR predictions directly from created model
start = time.time()
predictions1 = model.predict_generator(
        test_generator,
        steps=int(test_generator_samples/float(test_batch_size)), # all samples once
        verbose =1
        )

end  = time.time()
(end-start)/60


# 10.1 See arrays of predictions
predictions
predictions[0:10]

#%%                         H. Make submissions on Kaggle


# 11. Unfortunately Kaggle is not allowing submissions.
#     I have manually compiled a file looking at 300 images out of 12500
#     images in the test folder.
actual=pd.read_csv("E:/cats_and_dogs/actual_result300.csv", header = 0)
actual.head()

# 11.1 Add predictions column to this data frame
actual['new'] = predictions[0:300]

# 12. Evaluate accuracy
fpr, tpr, _ = metrics.roc_curve(actual['label'], actual['new'])

# 12.1 AUC
metrics.roc_auc_score(actual['label'], actual['new'])

# 12.2 ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.show()

############ END ##############################################

"""
Difference between flow() and flow_from_directory:
--------------------------------------------------

    flow_from_directory(directory) takes the path to a
    directory, and generates batches of augmented/normalized
    data. flow() expects data in RAM.
    In the flow(), we provide labels explicitly but in
    flow_from_directory, labels are inferred from dir structure
    Also in flow() images canNOT be resized. But in
    flow_from_directory these can be.

"""


"""
