from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
  def __init__(self, D_in, n_filters, kernel_size=3):
    super(Conv2dBlock, self).__init__()

    # first layer
    self.conv1       = nn.Conv2d(D_in, n_filters, kernel_size, stride=1, padding=1)
    self.batch_norm1 = nn.BatchNorm2d(n_filters)

    # second layer
    self.conv2       = nn.Conv2d(n_filters, n_filters, kernel_size, stride=1, padding=1)
    self.batch_norm2 = nn.BatchNorm2d(n_filters)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.batch_norm1(x)
    # x = F.relu(x)
    x = F.tanh(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    # out = F.relu(x)
    out = F.tanh(x)

    return out

class Model(nn.Module):
  def __init__(self, n_out, divider=1):
    super(Model, self).__init__()
    
    # Encoder
    self.conv1 = Conv2dBlock(1, int(16 / divider))
    self.pool1 = nn.MaxPool2d(2, 2)

    self.conv2 = Conv2dBlock(int(16 / divider), int(32 / divider))
    self.pool2 = nn.MaxPool2d(2, 2)

    self.conv3 = Conv2dBlock(int(32 / divider), int(64 / divider))
    self.pool3 = nn.MaxPool2d(2, 2)

    self.conv4 = Conv2dBlock(int(64 / divider), int(128 / divider))
    self.pool4 = nn.MaxPool2d(2, 2)

    self.conv5 = Conv2dBlock(int(128 / divider), int(256 / divider))

    # Decoder
    self.upconv6 = nn.ConvTranspose2d(int(256 / divider), int(128 / divider), kernel_size=3, stride=2, padding=1, output_padding=1)

    self.conv6   = Conv2dBlock(int(256 / divider), int(128 / divider))
    self.upconv7 = nn.ConvTranspose2d(int(128 / divider), int(64 / divider), kernel_size=3, stride=2, padding=1, output_padding=1)

    self.conv7   = Conv2dBlock(int(128 / divider), int(64 / divider))
    self.upconv8 = nn.ConvTranspose2d(int(64 / divider), int(32 / divider), kernel_size=3, stride=2, padding=1, output_padding=1)

    self.conv8   = Conv2dBlock(int(64 / divider), int(32 / divider))
    self.upconv9 = nn.ConvTranspose2d(int(32 / divider), int(16 / divider), kernel_size=3, stride=2, padding=1, output_padding=1)

    self.conv9   = Conv2dBlock(int(32 / divider), int(16 / divider))
    self.conv10   = nn.Conv2d(int(16 / divider), n_out, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    c1 = self.conv1(x)
    x  = self.pool1(c1)

    c2 = self.conv2(x)
    x  = self.pool2(c2)

    c3 = self.conv3(x)
    x  = self.pool3(c3)

    c4 = self.conv4(x)
    x  = self.pool4(c4)

    c5 = self.conv5(x)

    u6 = self.upconv6(c5)
    x  = torch.cat([u6, c4], dim=1)

    x = self.conv6(x)

    u7 = self.upconv7(x)
    x  = torch.cat([u7, c3], dim=1)

    x = self.conv7(x)

    u8 = self.upconv8(x)
    x  = torch.cat([u8, c2], dim=1)

    x  = self.conv8(x)

    u9 = self.upconv9(x)
    x  = torch.cat([u9, c1], dim=1)

    x  = self.conv9(x)

    out = self.conv10(x)

    return out