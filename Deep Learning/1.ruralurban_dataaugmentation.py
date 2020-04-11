# Last amended: 28th June, 2019
# My folder:  /home/ashok/Documents/4. transfer_learning

#
# Ref https://www.kaggle.com/dansbecker/exercise-data-augmentation
# https://www.kaggle.com/learn/deep-learning
# How the images were collected:
# Images were collected using facility available in Moodle at
#    http://203.122.28.230/moodle/mod/url/view.php?id=1768
# Or see:
#    https://addons.mozilla.org/en-US/firefox/addon/google-images-downloader/

# ResNet architecture is here:  http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006

# Objective:
#           Predict Rural-urban images with data augmentation using ResNet50

"""
Steps:
     i) Call libraries
    ii) Define necessary constants
   iii) Create ResNet model and classifier
   iv)  Create nested model
    v)  Make certain layers of ResNet50 trainable
        THIS STEP REDUCES ACCURACY
   vi)  Compile model
  vii)  Train-Data Augmentation:
             i) Define operations to be done on images--Configuration 1
            ii) Define from where to read images from,
                batch-size,no of classes, class-model  --Configuration 2
	ix)) Validation data augmentation
	x) Start training--Model fitting

"""



#    cp /home/ashok/.keras/keras_tensorflow.json  /home/ashok/.keras/keras.json
#    cat /home/ashok/.keras/keras.json
#    source activate tensorflow

######################### Call libraries
# 1. Call libraries
%reset -f

# 1.1 Application (ResNet50) library
# https://keras.io/applications/#resnet50
from keras.applications import ResNet50

# 1.2 Keras models and layers
from keras.models import Sequential

from keras.layers import Dense,  GlobalAveragePooling2D

# 1.3 Image generator and preprocessing
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# 1.4 To save model-weights with least val_acc
#      Refer Page 250 Chollet

from keras.callbacks import ModelCheckpoint


# 1.4 Misc
import matplotlib.pyplot as plt
import time,os


######################### Define necessary constants

# 2. Constants & Modeling
num_classes = 2
image_size = 224      # Restrict image-sizes to uniform: 224 X 224
                      # Decreasing increases speed but reduces accuracy

# 2.1 Image data folders
train_dir= '/home/ashok/Images/ruralurban/train'
val_dir = '/home/ashok/Images/ruralurban/val'


######################### Create ResNet model and classifier


# 2.2 Create ResNet59 model and import weights
base_model = ResNet50(
                          include_top=False,        # No last softmax layer
                          pooling='avg',            # GlobalAveragePooling2D to flatten last convolution layer
                                                    #  See: https://www.quora.com/What-is-global-average-pooling
                          weights='imagenet'
                          )

"""
Meaning of arguments
=====================
include_top = False
       whether to include the fully-connected layer at the top of the network.
pooling = 'avg'
       means that global average pooling will be
       applied to the output of the last convolutional layer,
       and thus the output of the model will be a 2D tensor.
       (2D includes batch_size also else it is just 1D)
weights:
       one of None (random initialization) or 'imagenet' (pre-training on ImageNet).

"""


# 2.3 Look at the model
base_model.summary()

# 2.4 Total number of layers are:

len(base_model.layers)      # 175

# 2.5 Look at layer names again
for i,layer in enumerate(base_model.layers):
    print((i,layer.name))


# 2.6 Initialise: Freeze all layers from training
for layer in base_model.layers:
    layer.trainable = False


# 2.7 Make layers 160 onwards available for training
#     Increasing training layers may reduce speed
#     as also accuracy

for layer in base_model.layers[160:]:
    layer.trainable = True


# 2.8 Quick Check
for layer in base_model.layers:
    print(layer.trainable)


# 3 Start nested model building
my_new_model = Sequential()


# 3.1 Nest base model within it
my_new_model.add(base_model)


# 3.2
my_new_model.summary()



# 3.3 Last output softmax layer
my_new_model.add(Dense(
                       num_classes,
                       activation='softmax'
                       )
                )

# 3.4
my_new_model.summary()

# 3.5 Can access nested layers as here:
for layer in my_new_model.layers[0].layers:
    print(layer.trainable)


# 4.0 Compile model
my_new_model.compile(
                     optimizer='sgd',
                     loss='categorical_crossentropy',
                     metrics=['accuracy']
                     )

######################### Train-Data Augmentation

# 4.1 Image processing and image generation
#     train data Image generator object
data_generator_with_aug = ImageDataGenerator(
                                             preprocessing_function=preprocess_input,      # keras image preprocessing function
                                             horizontal_flip=True,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2
                                            )

# 4.2 No fit() needed
#     Create image data generator interator for train data
train_generator = data_generator_with_aug.flow_from_directory(
                                                              train_dir,
                                                              target_size=(image_size, image_size),
                                                              batch_size=16 ,    # Increasing it increases
                                                                                 # processing time & may decrease accu
                                                              class_mode='categorical'
                                                              )


######################### Validation-Data Augmentation


# 4.3 validation data generator object
#     We will manipulate even Validation data also
#     Just to see if predictions are still made correctly
#     'data_generator_no_aug' is learner + ImageDataGenerator object
data_generator_no_aug = ImageDataGenerator(
                                           preprocessing_function=preprocess_input,
                                           rotation_range=90,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           )


# 4.4 validation data image iterator
validation_generator = data_generator_no_aug.flow_from_directory(
                                                                 val_dir,
                                                                 target_size=(image_size, image_size),
                                                                 batch_size= 16,  # Increasing it increases
										  # processing time & may decrease accu
                                                                 class_mode='categorical'
                                                                 )


######################### Model fitting


# 5. Prepare a list of callback functions
#    We will only have one. It will look at the val_loss
#    at the end of each epoch. Only if val_loss, for current
#    epoch is less than the previous, model-weights will be
#    saved, else not.
#    Refer page 250 Chollet

mycallbacks_list = [ModelCheckpoint(filepath='/home/ashok/Documents/4.transfer_learning/my_model.h5',
                                    monitor = 'val_loss',
                                    save_best_only = True
                                    )
                    ]


# 5.1 Model fitting. Takes 21 minutes for epochs = 5
#     When inner layers of ResNet50 are trained, accuracy
#     is less.


start = time.time()
history = my_new_model.fit_generator(
                           train_generator,
                           steps_per_epoch=4,     #  Total number of batches of samples to
                                                  #   yield from generator before declaring one epoch
                                                  #     finished and starting the next epoch.
                                                  #   Increase it to train the model better
                           epochs=5,             # All steps when finished, constitute one epoch
						                         # Increase it to get better training
                           validation_data=validation_generator,
                           validation_steps=2,    #  Total number of steps (batches of samples)
                                                  #   to yield from validation_data generator per epoch
                                                  #    Increase it to get a better idea of validation accu
                           workers = 2,           # Maximum number of processes to spin up
                           callbacks=mycallbacks_list, # What callbacks to act upon after each epoch
                           verbose = 1            # Show progress

                           )

end = time.time()
print("Time taken: ", (end - start)/60, "minutes")

# 5.2 Model weights
#     Have a look if model-weights have actually
#     been saved.

os.listdir('/home/ashok/Documents/4.transfer_learning/')

# 5.3 Plot training accuracy and validation accuracy
plot_history()


# 5.4
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


###########################################################################



"""
 What does GlobalAveragePooling2D do?
 And what are its adavantages over normal FC layer?
  It takes overall average of every filter. So for a convolution layer
    with 32 filters, we will have a 1D layer with 32 neurons
     GlobalAveragePooling2D can, therefore, be used to flatten the last
      convolution layer. See also below at the end of this code.
      See: https://www.quora.com/What-is-global-average-pooling
	 : https://stats.stackexchange.com/a/308218

Effect of GlobalAveragePoolimg is that the last resnet50 layer is a flat layer
with 2048 neurons. See this link: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
The next layer is a dense layer with 2 neurons. Thus total number of weights are: 4098


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
resnet50 (Model)             (None, 2048)              23587712
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 4098
=================================================================
Total params: 23,591,810
Trainable params: 4,098
Non-trainable params: 23,587,712


"""
