#!/usr/bin/env python3
import os
import sys
import argparse
import time
import copy
import shutil
import random

from PIL import Image

# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF

from model import Model
from utils import to_rgb, encode_bins, deserialize_bins, load_img
from train import train
from validate import validate
from evaluate import evaluate

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
  dest='data_dir', type=str,
  help='Data: Path to read-only directory containing image *.jpeg files.'
)

parser.add_argument(
  '--saved-model-dir', type=str, default=None,
  help='Data: Path of dir of last saved model'
)

parser.add_argument(
  '--saved-model-file', type=str, default=None,
  help='Data: File of last saved model'
)

parser.add_argument(
  '--image-file', type=str, default=None,
  help='Data: File of the image'
)

parser.add_argument(
  '--num-bins', type=int, default=36,
  help='Training: Number of bins. Default: 36'
)

parser.add_argument(
  '--seed', type=int, default=None,
  help='Parameter: Seed for the visualization'
)

parser.add_argument(
  '--temperature', type=float, default=1,
  help='Parameter: Temperature parameter to tune the predictions.'
)

args = parser.parse_args()
temperature = args.temperature
seed = args.seed

if seed is not None:
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

data_dir = os.path.expanduser(args.data_dir)

SAVED_MODEL_PATH = None

if args.saved_model_dir is not None and args.saved_model_file is not None:
  SAVED_MODEL_PATH = os.path.join(os.path.expanduser(args.saved_model_dir), args.saved_model_file)

if args.image_file is not None:
  IMAGE_PATH = os.path.join(os.path.expanduser(args.data_dir), args.image_file)

N_BINS = args.num_bins
W_BIN  = np.sqrt(N_BINS).astype(int)

use_gpu = torch.cuda.is_available()

'''
  Evaluate one image with trained model
'''
model = Model(N_BINS)

if SAVED_MODEL_PATH is not None:
  model.load_state_dict(torch.load(SAVED_MODEL_PATH))
  print(SAVED_MODEL_PATH)
  print('Model loaded')
  
image = Image.open(IMAGE_PATH)
image = TF.resize(image, 128)
image = TF.center_crop(image, 128)
gray, image_ab, bins = load_img(image, N_BINS)

output_image = evaluate(gray, image_ab, bins, model, {
  'mode': 'cached_colors/second-dataset/mode_color_bins_'+str(N_BINS)+'.npy',
  'mean': 'cached_colors/second-dataset/mean_color_bins_'+str(N_BINS)+'.npy'
}, temperature)

f, axarr = plt.subplots(len(gray), 2)

axarr[0].set_title('Ground truth')
axarr[0].imshow(to_rgb(gray, image_ab))

axarr[1].set_title('Generated')
axarr[1].imshow(output_image)

plt.pause(5)