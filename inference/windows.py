"""Historical sequential overlap averaging for FSC147."""
import torch

def window_starts(width, window=384, stride=128):
    if width < window:
        raise ValueError('Input width must be at least 384')
    starts = list(range(0, width-window+1, stride))
    if starts[-1] != width-window:
        starts.append(width-window)
    return starts

def merge_windows(predictions, starts, width):
    height, window = predictions[0].shape
    density = predictions[0].new_zeros((height, width))
    coverage = predictions[0].new_zeros((1, width))
    for start, prediction in zip(starts, predictions):
        region = density[:, start:start+window]
        count = coverage[:, start:start+window]
        region.copy_(torch.where(count > 0, (region+prediction)/2, prediction))
        count.add_(1)
    if torch.any(coverage <= 0):
        raise ValueError('Uncovered pixels in density assembly')
    return density
