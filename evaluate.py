import numpy as np
import torch
from utils import to_rgb, deserialize_bins

def evaluate(gray, ab_input, bins, model, colors_path):
  model.eval()

  with torch.no_grad():
    bins = bins.squeeze(0)

    # Use GPU
    # if use_gpu: gray, ab_input, bins = gray.cuda(), ab_input.cuda(), bins.cuda()

    # Run model and record loss
    # add batch size 1 for single image
    output_bins = model(gray.unsqueeze(1))

    # print(annealed_mean(output_bins.squeeze(0).numpy(), 0.36))
    # print(output_bins.squeeze(0).shape)
    
    # remove batch size and get the max index of the predicted bin for each pixel
    color_image = to_rgb(
      gray,
      torch.from_numpy(
        deserialize_bins(
          output_bins.squeeze(0).argmax(0), 
          colors_path
        ),
      ).float()
    )

  return color_image