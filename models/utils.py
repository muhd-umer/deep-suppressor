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

def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.shape[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return torch.cat([conv_below, attention_across], dim=1)

def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = nn.SpatialDropout2d
    elif dropout_type == "standard":
        DO = nn.Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = nn.Conv2d(
        filters,
        kernel_size,
        stride=1,
        padding=padding,
        kernel_initializer=kernel_initializer,
    )(inputs)
    if use_batch_norm:
        c = nn.BatchNorm2d(filters)(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = nn.Conv2d(
        filters,
        kernel_size,
        stride=1,
        padding=padding,
        kernel_initializer=kernel_initializer,
    )(c)
    if use_batch_norm:
        c = nn.BatchNorm2d(filters)(c)
    return c

"Defining Unet Model"
def custom_unet(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    if upsample_mode == "deconv":
        upsample = nn.ConvTranspose2d
    else:
        upsample = nn.Upsample

    # Build U-Net model
    inputs = nn.Input(input_shape)
    inputs_copy = inputs.clone()
    x = inputs / torch.max(inputs)

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = nn.MaxPool2d((2, 2))(x)
        dropout += dropout_change_per_layer
        filters *= 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), stride=2, padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = torch.cat([x, conv], dim=1)
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    output_mask = nn.Conv2d(num_classes, (1, 1), activation=output_activation)(x)
    outputs = torch.mul(output_mask, inputs_copy)
    model = nn.Model(inputs=[inputs], outputs=[outputs])
    return model



