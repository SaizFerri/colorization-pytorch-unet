from __future__ import print_function, division
import time
import os
import copy

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

import numpy as np
import torch
from torchvision import datasets, models, transforms

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
'''
  Generate a dictionary with the mode of every bin

  Output: BIN_NUMBER: [A_COLOR_MODE, B_COLOR_MODE] 
'''
def get_dataset_bin_mode(colors_dict):
  mode_dict = copy.deepcopy(colors_dict)

  for bin in mode_dict:
    for channel, _ in enumerate(mode_dict[bin]):
      if (len(mode_dict[bin][channel]) > 0):
        mode_dict[bin][channel] = np.max(np.array(mode_dict[bin][channel]))
      else:
        mode_dict[bin][channel] = 0

  return mode_dict

'''
  Calculate the bin from the indices

  Output: N
'''
def calculate_bin(a, b, width):
  return (width * b) + a

'''
  Add color to each bin dictionary value
'''
def add_to_dict(bin, a, b):
  dataset_bin_colors[bin][0].append(a)
  dataset_bin_colors[bin][1].append(b)


def encode_index(u):
  x = np.linspace(0, 1, W_BIN+1)
  return np.argmax(x>u)-1

'''
  Encode each pixel from the image into a bin

  Output: (W, H) where each value is a bin
'''
def encode_bins(ab_image, n_bins):
  w_bin  = np.sqrt(n_bins).astype(int)

  x = np.linspace(0,1,w_bin+1)
  indices = np.digitize(ab_image, x) - 1
  
  bins = np.vectorize(calculate_bin)(indices[:,:,0], indices[:,:,1], w_bin)

  return bins

'''
  Return the mode of the predicted bin for each color chanel
  TODO: calculate annealed mean
'''
def decode_pixel(bin, T, dataset_bin_colors_mode, dataset_bin_colors_mean):
  a_mode = dataset_bin_colors_mode[bin][0]
  b_mode = dataset_bin_colors_mode[bin][1]

  a_mean = dataset_bin_colors_mean[bin][0]
  b_mean = dataset_bin_colors_mean[bin][1]

  if a_mode == 0:
    a_mode = assign_next_bin(bin, a_mode, 0, dataset_bin_colors_mode)

  if b_mode == 0:
    b_mode = assign_next_bin(bin, b_mode, 1, dataset_bin_colors_mode)

  if a_mean == 0:
    a_mean = assign_next_bin(bin, a_mean, 0, dataset_bin_colors_mean)

  if b_mean == 0:
    b_mean = assign_next_bin(bin, b_mean, 1, dataset_bin_colors_mean)
  
  a_distance = a_mode - a_mean
  b_distance = b_mode - b_mean

  a = a_mode - (a_distance * T)
  b = b_mode - (b_distance * T)

  return a, b

'''
  Assign predicted color from next bin
'''
def assign_next_bin(bin, channel, index, colormap):
  counter = [1, -1]
  length = len(colormap) - 1

  while channel == 0:
    plus_index = bin + counter[0] if bin + counter[0] <= length else length
    channel = colormap[plus_index][index]

    if channel == 0:
      counter[0] += 1
      minus_index = bin + counter[1] if bin + counter[1] >= 0 else 0
      channel = colormap[minus_index][index]
      counter[1] -= 1

  return channel

def annealed_mean(z, T):
  return np.exp(z / T) / np.sum(np.exp(z / T))

def serialize_bins(ab_image):
  return encode_bins(ab_image)

def deserialize_bins(bins, T, mode_path, mean_path):
  bins = bins.numpy()
  dataset_bin_colors_mode = np.load(mode_path, allow_pickle=True)
  dataset_bin_colors_mean = np.load(mean_path, allow_pickle=True)

  return np.array(np.vectorize(decode_pixel)(bins, T, dataset_bin_colors_mode, dataset_bin_colors_mean))

def to_rgb(grayscale_input, ab_input):
  '''
    Convert to rgb
  '''
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() #combine channels
  color_image = color_image.transpose((1, 2, 0)) # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
  color_image = lab2rgb(color_image)
  
  return (color_image * 255).astype(int)