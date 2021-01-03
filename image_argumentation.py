import os
from glob import glob

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

DATASET_PATH = "dataset_clean/train/"
DATASET_VAL_PATH = "dataset_clean/val/"
AUGUMENTED_PATH = "dataset_resized/train/"
AUGUMENTED_VAL_PATH = "dataset_resized/val/"
CLASSES = ["field", "forest", "glacier", "lake", "mountain", "road", "sea", "uncategorized"]
SIZE = 128

# for _class in CLASSES:
#   train_files = glob("./{}{}/*".format(AUGUMENTED_PATH, _class))
#   val_files = glob("./{}{}/*".format(AUGUMENTED_VAL_PATH, _class))

#   for f in train_files:
#     os.remove(f)

#   for f in val_files:
#     os.remove(f)

aug = iaa.Resize({"height": SIZE, "width": SIZE}, "cubic")
# image = imageio.imread("dataset_clean/val/uncategorized/00000002.jpg")
# augumented_img = aug.augment_image(image)
# plt.imshow(augumented_img)
# plt.pause(5)
rotate=iaa.Affine(rotate=(-30, 30))
crop = iaa.Crop(percent=(0, 0.3))
flip_hr=iaa.Fliplr(p=1.0)
flip_vr=iaa.Flipud(p=1.0)

# Resize training images to the given size
for i, _class in enumerate(CLASSES):
  for filename in os.listdir(DATASET_PATH + "/" + CLASSES[i]):
    image = imageio.imread(DATASET_PATH + "/" + CLASSES[i] + "/" + filename)
    augumented_img = aug.augment_image(image)
    imageio.imwrite(AUGUMENTED_PATH + "/" + CLASSES[i] + "/" + filename, augumented_img)

# Resize validation images to the given size
for i, _class in enumerate(CLASSES):
  for filename in os.listdir(DATASET_VAL_PATH + "/" + CLASSES[i]):
    image = imageio.imread(DATASET_VAL_PATH + "/" + CLASSES[i] + "/" + filename)
    augumented_img = aug.augment_image(image)
    imageio.imwrite(AUGUMENTED_VAL_PATH + "/" + CLASSES[i] + "/" + filename, augumented_img)

# Apply argumentation to the images
for i, _class in enumerate(CLASSES):
  for filename in os.listdir(AUGUMENTED_PATH + "/" + CLASSES[i]):
    image = imageio.imread(AUGUMENTED_PATH + "/" + CLASSES[i] + "/" + filename)

    # Rotate image between -30 and 30 degrees
    rotated_image=rotate.augment_image(image)
    imageio.imwrite(AUGUMENTED_PATH + "/" + CLASSES[i] + "/rotated_" + filename, rotated_image)

    # Crop image by a factor of 30%
    cropped_image=crop.augment_image(image)
    imageio.imwrite(AUGUMENTED_PATH + "/" + CLASSES[i] + "/croped_" + filename, cropped_image)

    # Flip image horizontaly
    flip_hr_image= flip_hr.augment_image(image)
    imageio.imwrite(AUGUMENTED_PATH + "/" + CLASSES[i] + "/hr_flipped_" + filename, flip_hr_image)

    # Flip image verticaly
    flip_vr_image= flip_vr.augment_image(image)
    imageio.imwrite(AUGUMENTED_PATH + "/" + CLASSES[i] + "/vr_flipped_" + filename, flip_vr_image)
