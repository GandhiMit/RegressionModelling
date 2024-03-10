# Import the required libraries
# For this example we will use pytorch to manage the construction of the neural networks and the training
# torchvision is a module that is part of pytorch that supports vision datasets and it will be where we will source the mnist - handwritten digits - data

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

