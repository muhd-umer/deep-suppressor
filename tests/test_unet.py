import os
import sys

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import unittest
import torch
from src import CustomUNet


import os
import sys
import unittest
import torch
from src import (
    CustomUNet,
    UpsampleConv,
    UpsampleSimple,
    AttentionConcat,
    Conv2dBlock,
)


class TestCustomUNet(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 128, 256)
        self.num_classes = 1
        self.activation = "relu"
        self.use_batch_norm = True
        self.upsample_mode = "deconv"
        self.dropout = 0.3
        self.dropout_change_per_layer = 0.0
        self.dropout_type = "spatial"
        self.use_dropout_on_upsampling = False
        self.use_attention = False
        self.filters = 16
        self.num_layers = 4

    def test_forward_pass(self):
        model = CustomUNet(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            upsample_mode=self.upsample_mode,
            dropout=self.dropout,
            dropout_change_per_layer=self.dropout_change_per_layer,
            dropout_type=self.dropout_type,
            use_dropout_on_upsampling=self.use_dropout_on_upsampling,
            use_attention=self.use_attention,
            filters=self.filters,
            num_layers=self.num_layers,
        )
        x = torch.randn(2, *self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (2, *self.input_shape))


if __name__ == "__main__":
    unittest.main()
