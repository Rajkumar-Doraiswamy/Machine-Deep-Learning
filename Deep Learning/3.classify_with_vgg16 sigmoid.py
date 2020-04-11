'''
Last amended: 17th June, 2019
Ref: Page 143 (Section 5.3) of Book 'Deep Learning with python'
     by Francois Chollet

Objective:
         Transfer Learning: Building powerful image classification
                            models using very little data using
                            pre-trained applications

							THIS PROBLEM IS ATTEMPTED WITH SIGMOID LAYER AT OUTPUT

Steps:
	1. Create higher level abstract features from train data
       and save these to file
	2. Used saved features as input to a FC model to make predictions
    3. Save FC model
    4. Use the complete model to make predictions

Data from Kaggle: https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a folder: Images/cats_dogs/ folder
- created train/ and validation/ subfolders inside cats_dogs/
- created cats/ and dogs/ subfolders inside train/ and validation/

In summary, this is our directory structure:

Images/
	data/
	    train/
        	dogs/
        	    dog001.jpg
        	    dog002.jpg
        	    ...
        	cats/
        	    cat001.jpg
        	    cat002.jpg
        	    ...
	    validation/
        	dogs/
        	    dog001.jpg
        	    dog002.jpg
        	    ...
        	cats/
        	    cat001.jpg
        	    cat002.jpg
        	    ...

	$ source activate tensorflow
	$ ipython

'''

####### BEGIN #########

####********************************************************************************
#### *****************   PART I: Tranform train data to abstract features and save**
####********************************************************************************

%reset -f
## 1. Call libraries
import numpy as np

# 1.1 Classes for creating models
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# 1.2 Class for accessing pre-built models
from keras import applications

# 1.3 Class for generating infinite images
from keras.preprocessing.image import ImageDataGenerator

# 1.4 Miscelleneous
import matplotlib.pyplot as plt
import time, os

############################# AA. Constants & Hyperparameters ###################3
## 2. Constants/hyperparameters

# 2.1 Where are cats and dogs?
train_data_dir      =  '/home/ashok/Images/cats_dogs/train'
validation_data_dir =  '/home/ashok/Images/cats_dogs/validation'


# 2.2 Constrain dimensions of our images
#     during image generation
img_width, img_height = 75,75       # Small size images for modeling speed; (150, 150)


# 2.3 How many of them?
nb_train_samples, nb_validation_samples = 2000, 800


# 2.4 Predict in batches that fit RAM
#     and also sample-size is fully divisible by it
batch_size = 50         # Maybe for 4GB machine, batch-size of 32 will be OK


# 2.5 File to which transformed bottleneck features for train data wil be stored
bf_filename = '/home/ashok/.keras/models/bottleneck_features_train.npy'

# 2.6 File to which transformed bottleneck features for validation data wil be stored
val_filename = '/home/ashok/.keras/models/bottleneck_features_validation.npy'


############################# BB. Data Generation ###################3

## 3. Data augmentation (sort of)

# 3.1 Instanstiate an image data generator: Needd to feed into the model
#     Only normalization & nothing else like flipping, rotation etc; just to
#     keep things simple.

datagen_train = ImageDataGenerator(rescale=1. / 255)

# 3.2 Configure datagen_train further
#     Datagenerator is configured twice. First configuration
#    is about image manipulation features
#    IInd configuration is regarding data source, data classes, batchsize  etc

generator_tr = datagen_train.flow_from_directory(
              directory = train_data_dir,		      # Path to target train directory.
              target_size=(img_width, img_height),    # Dimensions to which all images will be resized.
              batch_size=batch_size,                  # At a time so many images will be output
              class_mode=None,                        # Return NO labels along with image data
              shuffle=False                           # Default shuffle = True
                                                      # Now images are picked up first from
                                                      #  one folder then from another; no shuffling
                                                      #   We will be using images NOT for
                                                      #    learning any model but only for prediction
                                                      #      so shuffle = False is OK as we now know
                                                      #       that Ist 1000 images are of one kind
                                                      #        and next 1000 images of another kind
                                                      # See: https://github.com/keras-team/keras/issues/3296
              )



"""
# 3.2 If data was not arranged in the directory, then iterator would be:
generator_t = datagen.flow(X_train,  # Should have rank 4. In case of grayscale data,
                                     #   the channels axis should have value 1, and in
                                     #    case of RGB data, it should have value 3.
                           y_train,  #  X_train labels
                           shuffle=False,
                           batch_size=batch_size
                        )

  There is, however, no 'target_size' parameter here
"""


# 3.3. Generator for validation data.
#      Initialize ImageDataGenerator object once more
datagen_val = ImageDataGenerator(rescale=1. / 255)
generator_val = datagen_val.flow_from_directory(
                                          validation_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False   # Default shuffle = True
                                                      # Now images are picked up first from
                                                      #  one folder then from another; no shuffling
                                                      #   We will be using images NOT for
                                                      #    learning any model but only for prediction
                                                      #      so shuffle = False is OK as we now know
                                                      #       that Ist 1000 images are of one kind
                                                      #        and next 1000 images of another kind
                                                      # See: https://github.com/keras-team/keras/issues/3296
                                          )



############################# CC. Modeling & Feature creation #####################
############################For both train and validation data ####################
####################Created features become our fresh train/validation data########

# 4. Buld VGG16 network model with 'imagenet' weights
#     Do not include the top FC layer of VGG16 model
#      Weights will be downloaded, if absent
model = applications.VGG16(
	                       include_top=False,
	                       weights='imagenet',
	                       input_shape=(img_width, img_height,3)
	                       )
model.summary()


# 4.1 Feed images through VGG16 model in batches (steps)
#     And make 'predictions'.
#     Following takes time 7 +3 = 10 minutes
#     Note that there is no need for 'fit' method as weights are
#     already learnt

start = time.time()
# 4.1 By feeding the input samples from generator, create vgg16 output/predictions
#     uptil the last layer. We call it 'bottleneck features' as it is not the desired
#     end result
#     steps:  How many batches of images to output
bottleneck_features_train = model.predict_generator(
                                                    generator = generator_tr,
                                                    steps = nb_train_samples // batch_size,
                                                    verbose = 1
                                                    )
end = time.time()
print("Time taken: ",(end - start)/60, "minutes")



# 4.2   Similarly, make predictions for validation data and extract features
#     Takes 12 minutes

start = time.time()
bottleneck_features_validation = model.predict_generator(
                                                         generator = generator_val,
                                                         steps = nb_validation_samples // batch_size,
                                                         verbose = 1
                                                         )

end = time.time()
print("Time taken: ",(end - start)/60, "minutes")


############################# DD. Saving features ###################


# 5. Save the train features
# 5.1 First delete the file to whcih we will save

if os.path.exists(bf_filename):
    os.system('rm ' + bf_filename)

# 5.2 Next save the train-features
np.save(open(bf_filename, 'wb'), bottleneck_features_train)


# 5.3 Save validation features from model
if os.path.exists(val_filename):
    os.system('rm ' + val_filename)

np.save(open(val_filename, 'wb'), bottleneck_features_validation)

# 6. Quit python so that complete memory is reset
#     Maybe reboot your lubuntu (NOT WINDOWS)

################### ########### ##################### #######
################### PART-II BEGIN AGAIN #####################
################### ########### ##################### #######

## Part II: Load saved abstract features and proceed
#           with modeling and prediction
# Start ipython #

# 1.0 Call libraries
%reset -f
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,Softmax
from keras import applications
import time, os


# 2. Hyperparameters/Constants
# 2.1 Dimensions of our images.
img_width, img_height = 75,75  # 150, 150
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 64
num_classes = 2
# Bottleneck features for train data
bf_filename = '/home/ashok/.keras/models/bottleneck_features_train.npy'
# Validation-bottleneck features filename
val_filename = '/home/ashok/.keras/models/bottleneck_features_validation.npy'


# 2.7 File to which FC model weights could be stored
top_model_weights_path = '/home/ashok/.keras/models/bottleneck_fc_model.h5'



# 3. Load first train features
train_data_features = np.load(open(bf_filename,'rb'))

# 3.1
train_data_features.shape

# 3.2 Train lables. First half are of one kind and next half of other
#     Remember we had put 'shuffle = False' in data generators
#     1000 labels of one kind. Another 1000 labels of another kind
train_labels = np.array([0] * 1000 + [1] * 1000)   # Try [0] * 3 + [1] * 5

# 4. Validation features
validation_data_features = np.load(open(val_filename,'rb'))

# 4.1
validation_data_features.shape

# 4.2 Validation labels: half-half
validation_labels = np.array([0] * 400 + [1] * 400)


# 5. Plan model with FC layers only
#    We use transformed features as input to FC model
#    instead of actual train data
model = Sequential()
model.add(Flatten(input_shape=train_data_features.shape[1:]))     # (2, 2, 512)
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# 5.1
model.compile(
              optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

# 5.2 Fit model and make predictions on validation dataset
#     Takes 2 minutes
#     Watch Validation loss and Validation accuracy (around 81%)
start = time.time()
history = model.fit(train_data_features, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data_features, validation_labels),
                    verbose =1
                   )
end = time.time()
print("Time taken: ",(end - start)/60, "minutes")


# 6.0
plot_history()


# 7. Finally save model weights for later use
model.save_weights(top_model_weights_path)
#######################################################

#     How accuracy changes as epochs increase
#     We will use this function agai and again
#     in subsequent examples

def plot_history():
    val_acc = history.history['val_acc']
    tr_acc=history.history['acc']
    epochs = range(1, len(val_acc) +1)
    plt.plot(epochs,val_acc, 'b', label = "Validation accu")
    plt.plot(epochs, tr_acc, 'r', label = "Training accu")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()
