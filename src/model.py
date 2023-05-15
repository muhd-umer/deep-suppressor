"""
Implementation of Custom UNet Model using blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import UpsampleConv, UpsampleSimple, AttentionConcat, Conv2dBlock


class DownBlock(nn.Module):
    """
    DownBlock class for the down-sampling path of the UNet model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        x: Output tensor after applying the down-sampling block.
        skip: Skip connection tensor.
    """

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv_block = Conv2dBlock(
            in_channels,
            out_channels,
            use_batch_norm=True,
            dropout=0.0,
            dropout_type="spatial",
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )

    def forward(self, x):
        x = self.conv_block(x)
        skip = x
        x = F.max_pool2d(x, kernel_size=(2, 2))
        return x, skip


class UpBlock(nn.Module):
    """
    UpBlock class for the up-sampling path of the UNet model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        upsample_mode: Type of up-sampling method to use. Must be one of ['deconv', 'simple'].

    Returns:
        x: Output tensor after applying the up-sampling block.
    """

    def __init__(self, in_channels, out_channels, upsample_mode="deconv"):
        super(UpBlock, self).__init__()
        if upsample_mode == "deconv":
            self.upsample = UpsampleConv(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        elif upsample_mode == "simple":
            self.upsample = UpsampleSimple(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            raise ValueError(
                f"upsample_mode must be one of ['deconv', 'simple'], got {upsample_mode}"
            )

        self.conv_block = Conv2dBlock(
            in_channels + out_channels,
            out_channels,
            use_batch_norm=True,
            dropout=0.0,
            dropout_type="spatial",
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class CustomUNet(nn.Module):
    """
    CustomUNet class for the UNet model.

    Args:
        input_shape: Shape of the input tensor.
        num_classes: Number of output classes.
        activation: Type of activation function to use.
        use_batch_norm: Whether to use batch normalization.
        upsample_mode: Type of up-sampling method to use. Must be one of ['deconv', 'simple'].
        dropout: Dropout rate.
        dropout_change_per_layer: Dropout change per layer.
        dropout_type: Type of dropout to use.
        use_dropout_on_upsampling: Whether to use dropout on up-sampling.
        use_attention: Whether to use attention.
        filters: Number of filters.
        num_layers: Number of layers.

    Returns:
        outputs: Output tensor after applying the UNet model.
    """

    def __init__(
        self,
        input_shape,
        num_classes=1,
        activation="relu",
        use_batch_norm=True,
        upsample_mode="deconv",
        dropout=0.3,
        dropout_change_per_layer=0.0,
        dropout_type="spatial",
        use_dropout_on_upsampling=False,
        use_attention=False,
        filters=16,
        num_layers=4,
    ):
        super(CustomUNet, self).__init__()
        if upsample_mode == "deconv":
            self.upsample = UpsampleConv
        elif upsample_mode == "simple":
            self.upsample = UpsampleSimple
        else:
            raise ValueError(
                f"upsample_mode must be one of ['deconv', 'simple'], got {upsample_mode}"
            )

        self.attention = AttentionConcat

        # Build U-Net model
        self.inputs = nn.Identity()
        self.inputs_copy = nn.Identity()
        self.filters = filters
        self.dropout = dropout
        self.dropout_change_per_layer = dropout_change_per_layer
        self.use_dropout_on_upsampling = use_dropout_on_upsampling
        self.use_attention = use_attention
        self.num_layers = num_layers

        self.down_layers = nn.ModuleList()
        for l in range(num_layers):
            self.in_channels = 1 if l == 0 else filters // 2
            self.out_channels = filters
            self.down_layers.append(
                DownBlock(
                    self.in_channels,
                    self.out_channels,
                )
            )
            filters *= 2

        filters = filters // 2
        self.conv = Conv2dBlock(
            filters,
            filters,
            use_batch_norm,
            dropout,
            dropout_type,
            (3, 3),
            activation,
        )

        if not use_dropout_on_upsampling:
            self.dropout = 0.0
            self.dropout_change_per_layer = 0.0

        self.up_layers = nn.ModuleList()
        for l in reversed(range(num_layers)):
            filters //= 2
            self.dropout -= dropout_change_per_layer
            self.up_layers.append(
                UpBlock(
                    filters * 2,
                    filters,
                    upsample_mode=upsample_mode,
                )
            )

        self.output_mask = nn.Conv2d(filters, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.outputs = nn.Identity()

    def forward(self, x):
        x = self.inputs(x)
        x_copy = self.inputs_copy(x)
        x = x / torch.max(x)

        down_layers = []
        for l in range(self.num_layers):
            x, skip = self.down_layers[l](x)
            down_layers.append(skip)
            self.dropout += self.dropout_change_per_layer

        x = self.conv(x)

        for l in range(self.num_layers):
            x = self.up_layers[l](x, down_layers[-l - 1])
            self.dropout -= self.dropout_change_per_layer

        output_mask = self.output_mask(x)
        output_mask = self.sigmoid(output_mask)
        outputs = torch.mul(output_mask, x_copy)
        return outputs
