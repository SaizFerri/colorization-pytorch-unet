import time
from utils import AverageMeter

def validate(val_loader, model, criterion, epoch, use_gpu=False):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()

  for i, (input_gray, input_ab, bins) in enumerate(val_loader):
    data_time.update(time.time() - end)
    bins = bins.squeeze(0)

    # Use GPU
    if use_gpu: input_gray, input_ab, bins = input_gray.cuda(), input_ab.cuda(), bins.cuda()

    # Run model and record loss
    output_bins = model(input_gray)
    loss = criterion(output_bins, bins)
    losses.update(loss.item(), input_gray.size(0))

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg