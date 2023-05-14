# importing pytorch libraries for the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import init
from torch.nn import Sequential

"Create a convolutional block in the UNet architecture using pytorch"


def upsample_conv(filters, kernel_size, strides, padding):
    return nn.ConvTranspose2d(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return nn.Upsample(scale_factor=strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
    Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = nn.Conv2d(
        n_intermediate_filters,
        kernel_size=1,
        stride=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = nn.Conv2d(
        n_intermediate_filters,
        kernel_size=1,
        stride=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = F.relu(torch.add(inp_1_conv, inp_2_conv))
    g = nn.Conv2d(
        filters=1,
        kernel_size=1,
        stride=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = F.sigmoid(g)
    return torch.mul(inp_1, h)
