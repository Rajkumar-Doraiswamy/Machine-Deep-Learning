# -*- coding: utf-8 -*-
"""
Last amended: 14th Feb, 2018
My folder: C:\Users\ashokharnal\OneDrive\Documents\kaggle_invasive
	  /home/ashok/Documents/7.BasicImageManipulation
Data folder: /home/ashok/Images/grey_seal

Moodle location: 'Image Processing, classification & deep learning' 


Ref: 
  User guide:  http://scikit-image.org/docs/dev/user_guide
  API:         http://scikit-image.org/docs/dev/api/api.html

Objectives:
    A.   Image manipulation using skimage
	    i.  Reading image to numpy array
	    ii. Displaying a png/jpg image
	    iii.  Displaying an image in numpy array form
	    iv.   Image resizing to a fixed height and width
	    v.  Image downsampling by a factor
	    vi. Adding noise to images
	    vii.  Image manipulation using numpy array functions
    B.   Image manipulation using Opencv
	    Ref: 
            http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html#display-image

Change to opencv virtualEnv

	$ source activate opencv
	$ ipython

"""

#%%                             Using skimage
%reset -f

#1
import os
import numpy as np
import pandas as pd

# 1.1 Install as: conda install scikit-image
#      scikit-image (or skimage) is a collection of algorithms
#       for image processing and computer vision.
from skimage import io, transform, util



#%%                               Images & Manipulation


# 2. This folder contains some images
imagepath = "/home/ashok/Images/grey_seal"
os.chdir(imagepath)

# 2.1 What images are here?
files = os.listdir(imagepath)

# 2.2 What is the total no of images?
len(files)


# 3. Read image into a numpy array
x1 = io.imread(files[0])
x1   

# 3.1 Retrieving the geometry of the image and the number of pixels:
#      First dimension (x1.shape[0]) corresponds to rows, while the
#       second (x1.shape[1]) corresponds to columns, with the
#        origin (x1[0, 0]) on the top-left corner. 
x1.shape          # ht_pixels X width_pixels X RGB_channels
# 3.2
x1.ndim

# 3.3 Total no of pixels in the image
x1.size           # ht_pixels * width_pixels * RGB_channels

# 3.4 Minimum/maximum color values
x1.min()
x1.max()


# 4. Look at an image
#    Either supply 'png' filename
io.imshow("robbe.jpg")
io.show()


# 4.1 Or directly supply the
#     numpy array
io.imshow(x1)
io.show()


# 5. Rescale, resize, and downscale
#       Rescale operation resizes an image by a given scaling factor.
#       The scaling factor can either be a single floating point value,
#       or multiple values - one along each axis.
#       Resize serves the same purpose, but allows to specify an output
#       image shape instead of a scaling factor.
#       Note that when down-sampling an image, resize and rescale should
#       perform Gaussian smoothing to avoid aliasing artifacts.
#       See the anti_aliasing and anti_aliasing_sigma arguments to these functions.
#       Downscale serves the purpose of down-sampling an n-dimensional image by
#       integer factors using the local mean on the elements of each block of
#       the size factors given as a parameter to the function.
# http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html


# 5.1 Resize and image to pre-specified pixel size
io.imshow(x1)
x2 = transform.resize(x1,(100,150))
io.imshow(x2)
io.show()


# 5.2 Rescale image by a factor
xr = transform.rescale(x1, 1.0/4.0)
xr.shape
io.imshow(xr)
io.show()


# 5.3 Downscale using local mean
x3 = transform.downscale_local_mean(x1, (2,2,1))
x3.shape
io.imshow(x3)
io.show()

# 5.4 Smooth and then downsample image.
x4 = transform.pyramid_reduce(x1, downscale=3)
io.imshow(x4)
io.show()
x4.shape

# 6. Add random noise to image
x6= util.random_noise(x1, mode = 'gaussian')
x6.shape
io.imshow(x6)
io.show()

# 6.1 Add random noise to image
x7= util.random_noise(x1, mode = 'gaussian', mean = 0.2)
x7.shape
io.imshow(x7)
io.show()

# 6.2 Add random noise to image
x8= util.random_noise(x1, mode = "speckle")
x8.shape
io.imshow(x8)
io.show()


# 7. Transform images using numpy array manipulation
# 7.1 Rotate an image by 90 degree
x90 = np.rot90(x1, 1)
io.imshow(x90)
io.show()

# 7.2 Flip an image vertically
xflip = np.flipud(x1)
io.imshow(xflip)
io.show()


# 7.3 Rotate image by 90 deg counter-clockwise
xc90 = np.rot90(x1,3)
io.imshow(xc90)
io.show()

# 7.4 Rotate image by 180 degrees
x180 = np.rot90(x1,2)
io.imshow(x180)
io.show()

# 7.5 Flip image horizontally
xc180a = np.fliplr(x1)
io.imshow(xc180a)
io.show()

################################################

#%%                             Using opencv

%reset -f

#1
import os
import numpy as np
import pandas as pd

# 1.2
import cv2
import matplotlib.pyplot as plt


# 1.3 
imagepath = "/home/ashok/Images/grey_seal"
os.chdir(imagepath)

# 1.4 What images are here?
files = os.listdir(imagepath)

# 1.5 What is the total no of images?
len(files)

# 2.0 Read image as numpy array
img = cv2.imread(files[2])
img.shape
img.dtype
img.size

# 2.1 Accessing only blue pixel
blue = img[100,100,0]
blue


# 2.2 Better pixel accessing and editing method :
# accessing RED value
img.item(10,10,2)

#  2.3 Modifying RED value
img.itemset((10,10,2),100)
img.item(10,10,2)

# 2.4 Display image
cv2.imshow("My window",img)
cv2.waitKey(0)

# 2.4.1 Or,as:
img = cv2.imread(files[2])
ball = img[260:280, 330:390]
img[200:220, 130:190] = ball
plt.imshow(img)
plt.show()

# 2.4.2 Display multiple images
img1  =  cv2.imread(files[1])
plt.subplot(121)
plt.imshow(img1)
plt.title('Ist Image')
# No xticks and yticks
plt.xticks([])
plt.yticks([])
img2 =  cv2.imread(files[2])
plt.subplot(122)
plt.imshow(img2,cmap = 'gray')
plt.title('IInd Image')
plt.xticks([]), plt.yticks([])
plt.show()

#############################################










