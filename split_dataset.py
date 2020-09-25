import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

CLASSES = ["field", "forest", "glacier", "lake", "mountain", "road", "sea", "uncategorized"]
DATASET_PATH = "dataset_clean/"
NEW_DATASET_PATH = "dataset_clean_random_split/"

# for _class in CLASSES:
#   train_files = glob("./{}/train/{}/*".format(NEW_DATASET_PATH, _class))
#   val_files = glob("./{}/val/{}/*".format(NEW_DATASET_PATH, _class))

#   for f in train_files:
#     os.remove(f)

#   for f in val_files:
#     os.remove(f)

images_train = []

for _class in CLASSES:
  images_train.append(glob(DATASET_PATH + "train/" + _class + "/*.jpg"))

images_train = [item for sublist in images_train for item in sublist]

images_val = []

for _class in CLASSES:
  images_val.append(glob(DATASET_PATH + "val/" + _class + "/*.jpg"))

images_val = [item for sublist in images_val for item in sublist]

all_images = images_train + images_val

train_names, test_names = train_test_split(all_images, test_size=0.2)

for image in train_names:
  path = image.split('/')
  final_path = path[len(path) - 2:]
  final_path = "/".join(final_path)

  shutil.copy(os.path.join("./", image), os.path.join("./{}train/".format(NEW_DATASET_PATH), final_path))

for image in test_names:
  path = image.split('/')
  final_path = path[len(path) - 2:]
  final_path = "/".join(final_path)

  shutil.copy(os.path.join("./", image), os.path.join("./{}val/".format(NEW_DATASET_PATH), final_path))