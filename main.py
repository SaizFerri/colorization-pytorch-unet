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
  '--checkpoints-dir', type=str, default=None,
  help='Data: Path to writable directory for the checkpoint files'
)

parser.add_argument(
  '--learning-rate', type=float, default=0.001,
  help='Training: Learning rate. Default: 0.001'
)

parser.add_argument(
  '--batch-size', type=int, default=64,
  help='Training: Batch size. Default: 64'
)

parser.add_argument(
  '--num-bins', type=int, default=36,
  help='Training: Number of bins. Default: 36'
)

parser.add_argument(
  '--from-epoch', type=int, default=0,
  help='Training: From epoch. Default: 0'
)

parser.add_argument(
  '--num-epochs', type=int, default=100,
  help='Training: Number of epochs. Default: 100'
)

parser.add_argument(
  '--log-dir', type=str, default=None,
  help='Debug: Path to writable directory for a log file to be created. Default: log to stdout / stderr'
)

parser.add_argument(
  '--log-file-name', type=str, default='training.log',
  help='Debug: Name of the log file, generated when --log-dir is set. Default: training.log'
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

# Redirect output streams for logging
if args.log_dir:
  log_file = open(os.path.join(os.path.expanduser(args.log_dir), args.log_file_name), 'w')
  sys.stdout = log_file
  sys.stderr = log_file

data_dir = os.path.expanduser(args.data_dir)

TRAIN_PATH = os.path.join(data_dir, 'train')
VAL_PATH = os.path.join(data_dir, 'val')

if args.checkpoints_dir is not None:
  CHECKPOINTS_PATH = os.path.expanduser(args.checkpoints_dir)

SAVED_MODEL_PATH = None

if args.saved_model_dir is not None and args.saved_model_file is not None:
  SAVED_MODEL_PATH = os.path.join(os.path.expanduser(args.saved_model_dir), args.saved_model_file)

N_BINS = args.num_bins
W_BIN  = np.sqrt(N_BINS).astype(int)

use_gpu = torch.cuda.is_available()

class GrayscaleImageFolder(datasets.ImageFolder):
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)

    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      
      # rgb to lab
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]

      # form bins
      bins = torch.from_numpy(encode_bins(img_ab, N_BINS))

      #ab channels
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

      # greyscale image
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    
    return img_original, img_ab, bins

# Training
train_transforms = transforms.Compose([
  transforms.RandomResizedCrop(128),
  transforms.RandomHorizontalFlip()
])
train_imagefolder = GrayscaleImageFolder(TRAIN_PATH, train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=args.batch_size, shuffle=True)

# Validation
val_transforms = transforms.Compose([
  transforms.Resize(128),
  transforms.CenterCrop(128)
])
val_imagefolder = GrayscaleImageFolder(VAL_PATH, val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=args.batch_size, shuffle=True)

'''
Training
'''
# model = Model(N_BINS)

# if SAVED_MODEL_PATH is not None:
#   model.load_state_dict(torch.load(SAVED_MODEL_PATH))
#   print(SAVED_MODEL_PATH)
#   print('Model loaded')

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

# if use_gpu: 
#   criterion = criterion.cuda()
#   model = model.cuda()

# epochs = args.num_epochs
# best_losses = 3

# for epoch in range(args.from_epoch, epochs):
#   if use_gpu and epoch > args.from_epoch:
#     model.cuda()
#   # Train for one epoch, then validate
#   train(train_loader, model, criterion, optimizer, epoch, use_gpu)
#   with torch.no_grad():
#     losses = validate(val_loader, model, criterion, epoch, use_gpu)
#     scheduler.step(losses)
#   # Save checkpoint and replace old best model if current model is better
#   if losses < best_losses:
#     best_losses = losses
#     torch.save(model.to('cpu').state_dict(), '{}/model-{}-{}-{:.3f}.pth'.format(CHECKPOINTS_PATH, N_BINS, epoch+1,losses))

'''
  Evaluate one image with trained model
'''
model = Model(N_BINS)

if SAVED_MODEL_PATH is not None:
  model.load_state_dict(torch.load(SAVED_MODEL_PATH))
  print(SAVED_MODEL_PATH)
  print('Model loaded')
#model.load_state_dict(torch.load('{}model-324-43-208.819.pth'.format(CHECKPOINTS_PATH)))

image = Image.open('dataset_1/val/field/243202127_1f7da59043.jpg')
image = TF.resize(image, 128)
image = TF.center_crop(image, 128)
gray, image_ab, bins = load_img(image, N_BINS)

output_image = evaluate(gray, image_ab, bins, model, {
  'mode': 'cached_colors/second-dataset/mode_color_bins_'+str(N_BINS)+'.npy',
  'mean': 'cached_colors/second-dataset/mean_color_bins_'+str(N_BINS)+'.npy'
}, temperature)
# gray, image_ab, bins = next(iter(val_loader))
# Show grayscale image
# plt.imshow(gray[0].numpy().transpose(1, 2, 0).squeeze(2), cmap='gray', vmin=0, vmax=1)
# plt.figure()
# f, axarr = plt.subplots(len(gray), 2)

# for i in range(len(gray)):
#   output_image = evaluate(gray[i], image_ab[i], bins[i], model, {
#     'mode': 'cached_colors/second-dataset/mode_color_bins_'+str(N_BINS)+'.npy',
#     'mean': 'cached_colors/second-dataset/mean_color_bins_'+str(N_BINS)+'.npy'
#   }, temperature)

#   axarr[i][0].set_title('Ground truth')
#   axarr[i][0].imshow(to_rgb(gray[i], image_ab[i]))

#   axarr[i][1].set_title('Generated')
#   axarr[i][1].imshow(output_image)

# plt.pause(5)

f, axarr = plt.subplots(len(gray), 2)

axarr[0].set_title('Ground truth')
axarr[0].imshow(to_rgb(gray, image_ab))

axarr[1].set_title('Generated')
axarr[1].imshow(output_image)

plt.pause(5)

# dataset_bin_colors = {i: [[], []] for i in range(N_BINS)}

# def get_dataset_bin_mode(colors_dict):
#   x = np.linspace(0,1,W_BIN+1)
#   distance = x[1]
#   mode_dict = copy.deepcopy(colors_dict)

#   for bin in mode_dict:
#     for channel, _ in enumerate(mode_dict[bin]):
#       if (len(mode_dict[bin][channel]) > 0):
#         mode_dict[bin][channel] = np.mean(np.array(mode_dict[bin][channel]))
#       else:
#         mode_dict[bin][channel] = 0

#   np.save('mean_color_bins_36.npy', mode_dict)
#   del mode_dict

# def calculate_bin(a, b, width):
#   return (width * b) + a

# def add_to_dict(bin, a, b):
#   dataset_bin_colors[bin][0].append(a)
#   dataset_bin_colors[bin][1].append(b)

# def _encode_bins(ab_image):
#   x = np.linspace(0,1,W_BIN+1)
#   indices = np.digitize(ab_image, x) - 1
  
#   bins = np.vectorize(calculate_bin)(indices[:,:,0], indices[:,:,1], W_BIN)
#   np.vectorize(add_to_dict)(bins, ab_image[:,:,0], ab_image[:,:,1])

#   return bins

# counter = 0

# for index, y in enumerate(train_loader):
#   gray_images, ab_images, bins = y

#   for ab in ab_images:
#     _encode_bins(ab)

# get_dataset_bin_mode(dataset_bin_colors)
