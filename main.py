#!/usr/bin/env python3
import os
import sys
import argparse
import time
import copy
import shutil

# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms

from model import Model
from utils import to_rgb, encode_bins
from train import train
from validate import validate

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
  dest='data_dir', type=str,
  help='Data: Path to read-only directory containing image *.jpeg files.'
)

# parser.add_argument(
#   '--outputs-dir', type=str, default=None,
#   help='Data: Path to writable directory for the output images'
# )

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

args = parser.parse_args()

# Redirect output streams for logging
if args.log_dir:
  log_file = open(os.path.join(os.path.expanduser(args.log_dir), args.log_file_name), 'w')
  sys.stdout = log_file
  sys.stderr = log_file

data_dir = os.path.expanduser(args.data_dir)

TRAIN_PATH = os.path.join(data_dir, 'train')
VAL_PATH = os.path.join(data_dir, 'val')

CHECKPOINTS_PATH = os.path.expanduser(args.checkpoints_dir)

N_BINS = args.num_bins
W_BIN  = np.sqrt(N_BINS).astype(int)

# dataset_bin_colors_mode = {}

use_gpu = False# torch.cuda.is_available()

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

model = Model(N_BINS)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if use_gpu: 
  criterion = criterion.cuda()
  model = model.cuda()

epochs = args.num_epochs
best_losses = 400

for epoch in range(0, epochs):
  if use_gpu and epoch > 0:
    model.cuda()
  # Train for one epoch, then validate
  train(train_loader, model, criterion, optimizer, epoch, use_gpu)
  with torch.no_grad():
    losses = validate(val_loader, model, criterion, epoch, use_gpu)
  # Save checkpoint and replace old best model if current model is better
  if losses < best_losses:
    best_losses = losses
    torch.save(model.to('cpu').state_dict(), '{}/model-{}-{}-{:.3f}.pth'.format(CHECKPOINTS_PATH, N_BINS, epoch+1,losses))