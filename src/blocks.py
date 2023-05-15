import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConv(nn.Module):
    """
    UpsampleConv block that performs transposed convolution to upsample the input tensor.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.

    Returns:
        Output tensor of the block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConv, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        return x


class UpsampleSimple(nn.Module):
    """
    UpsampleSimple block that performs nearest neighbor upsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.

    Returns:
        Output tensor of the block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleSimple, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, x):
        x = self.upsample(x)
        return x


class AttentionGate(nn.Module):
    """
    AttentionGate block that performs attention mechanism across two input tensors.

    Args:
        inp_1_channels: Number of channels in the first input tensor.
        inp_2_channels: Number of channels in the second input tensor.
        n_intermediate_filters: Number of intermediate filters.

    Returns:
        Output tensor of the block.
    """

    def __init__(self, inp_1_channels, inp_2_channels, n_intermediate_filters):
        super().__init__()
        self.inp_1_conv = nn.Conv2d(
            inp_1_channels, n_intermediate_filters, kernel_size=1
        )
        self.inp_2_conv = nn.Conv2d(
            inp_2_channels, n_intermediate_filters, kernel_size=1
        )
        self.f = nn.ReLU(inplace=True)
        self.g = nn.Conv2d(n_intermediate_filters, 1, kernel_size=1)
        self.h = nn.Sigmoid()

    def forward(self, inp_1, inp_2):
        inp_1_conv = self.inp_1_conv(inp_1)
        inp_2_conv = self.inp_2_conv(inp_2)
        f = self.f(inp_1_conv + inp_2_conv)
        g = self.g(f)
        h = self.h(g)
        return inp_1 * h


class AttentionConcat(nn.Module):
    """
    AttentionConcat block that concatenates the output of AttentionGate with the input tensor.

    Args:
        conv_below: Number of channels in the input tensor.
        skip_connection: Number of channels in the skip connection tensor.

    Returns:
        Output tensor of the block.
    """

    def __init__(self, conv_below, skip_connection):
        super().__init__()
        self.attention_gate = AttentionGate(skip_connection, conv_below, conv_below)

    def forward(self, conv_below, skip_connection):
        attention_across = self.attention_gate(skip_connection, conv_below)
        return torch.cat([conv_below, attention_across], dim=1)


class Conv2dBlock(nn.Module):
    """
    Conv2dBlock block that performs convolution on the input tensor.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_batch_norm: Whether to use batch normalization.
        dropout: Dropout probability.
        dropout_type: Type of dropout to use.
        kernel_size: Size of the convolution kernel.
        activation: Activation function to use.
        padding: Padding type to use.

    Returns:
        Output tensor of the block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_batch_norm=True,
        dropout=0.3,
        dropout_type="spatial",
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    ):
        super().__init__()
        if dropout_type == "spatial":
            self.DO = nn.Dropout2d(p=dropout)
        elif dropout_type == "standard":
            self.DO = nn.Dropout(p=dropout)
        else:
            raise ValueError(
                f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
            )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=not use_batch_norm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=not use_batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = (
            nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.DO(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x
