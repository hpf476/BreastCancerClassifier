import os
import random
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from sklearn import preprocessing

def read_class_list_mod(class_list,replacementpath):
    """
    Scan the image file and get the image paths and labels
    """
    with open(class_list) as f:
        lines = f.readlines()
        images = []
        labels = []
        for l in lines:
            items = l.strip().split()
            #z=os.path.basename(items[0])
            z=items[0]
            z=os.path.join(replacementpath,z)

            images.append(z)
            labels.append(int(items[1]))

        #store total number of data
        data_size = len(labels)
        return images, labels

def next_batch(batch_size, images, labels):
    """
    This function gets the next n ( = batch_size) images from the path list
    and labels and loads the images into them into memory
    """
    # Get next batch of image (path) and labels
    #   paths = images[pointer:pointer + batch_size]
    #   labels = labels[pointer:pointer + batch_size]
    paths = images[0:batch_size]
    labels = labels[0:batch_size]

    #update pointer
    #pointer += batch_size
    n_classes = 2
    patchSize = (277,277)
    mean = np.array([104., 117., 124.])
    scale_size=(350, 230)
    horizontal_flip = False
    shuffle = False

    # Read images
    randomized_img_list = []
    randomized_label_list = []
    for i in range(len(paths)):
      images = []
      label_list = []
      temp = []
      #rescale image
      img0 = PIL.Image.open(paths[i])
    #   img0 = img0.resize((scale_size[1], scale_size[0]),PIL.Image.LANCZOS) # PIL.Image.BICUBIC)

      img = np.array(img0,dtype=np.float32)

      if(img.ndim>3):
          img=img[:,:,0:3]
      img=img[:,:,[2,1,0]]

      for (x,y,crop) in resize_and_extract_overlapping(img, patchSize):
          if crop.shape[0] != patchSize[0] or crop.shape[1] != patchSize[1]:
          # check that the image is of the right size before we do further pre-processing
              continue
          if horizontal_flip and np.random.random() < 0.5:       # randomly flips image
              crop = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        #   crop -= mean        # normalize every patch

          images.append(crop)
          label_list.append(labels[i])

# ------------------ READ HERE ZAIZAI-------------------------------------
      temp = random.sample(list(enumerate(images)), 2)
      # This enumerates through the images from a single crop cycle
      # Then, it samples 2 of (index, image) from that list
      for idx, val in temp: # seperate the tuples into index and image
          randomized_label_list.append(label_list[idx]) # append label to a new list
          randomized_img_list.append(val)   # append image to new list

    image_matrix = np.ndarray([len(randomized_img_list), patchSize[0], patchSize[1], 3])
    for i in range(len(randomized_img_list)):
      image_matrix[i] = randomized_img_list[i]

    # Expand labels to one hot encoding
    one_hot_labels = np.zeros((len(randomized_img_list), n_classes))

    for i in range(len(randomized_label_list)):
        one_hot_labels[i][randomized_label_list[i]] = 1

    print ("image matrix size: ", image_matrix.shape)
    print ("hi", len(images))
    print ("One hot labels: ", one_hot_labels.shape)
    #return array of images and labels
    return images, one_hot_labels

def resize_and_extract_overlapping(im, patchSize):
  # step size should be 0.5 * image dim, and windowSize = 32x32
  stepSize = (int(round(patchSize[0]/2)), int(round(patchSize[1]/2)))
  imgDim = (im.shape[0], im.shape[1])
  for y in range(0, imgDim[1], stepSize[1]):
      print (y)
      for x in range(0, imgDim[0], stepSize[0]):
          crop = im[x:x+patchSize[0], y:y + patchSize[1]]
          yield (x,y,crop) # yield gives back a generator object

replacementpath = "/home/pf1404/Documents/ai_coding_tasks/project/"
train_file = "/home/pf1404/Documents/ai_coding_tasks/project/40X_train.txt"

images, labels = read_class_list_mod(train_file, replacementpath)
image_mat, one_hot_labels = next_batch(2, images, labels)
# im = "/home/pf1404/Documents/ai_coding_tasks/project/test_images/adenoma200.png"
# img0 = PIL.Image.open(im)
#
# img0 = img0.resize((350,230),PIL.Image.LANCZOS)
# img = np.array(img0,dtype=np.float32)
#
# scaled = preprocessing.scale(img)
#
# patchSize = (32,32)
# for (x,y,crop) in resize_and_extract_overlapping(img, patchSize):
#     if crop.shape[0] != 32 or crop.shape[1] != 32:
#         continue
