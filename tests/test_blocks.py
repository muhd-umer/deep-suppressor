import os
import sys

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import unittest
import torch
from src import (
    UpsampleConv,
    UpsampleSimple,
    AttentionGate,
    AttentionConcat,
    Conv2dBlock,
)


class TestBlocks(unittest.TestCase):
    def test_upsample_conv(self):
        upsample = UpsampleConv(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=2,
        )
        x = torch.randn(2, 3, 32, 32)
        output = upsample(x)
        self.assertEqual(output.shape, (2, 6, 64, 64))

    def test_upsample_simple(self):
        upsample = UpsampleSimple(strides=2)
        x = torch.randn(2, 3, 32, 32)
        output = upsample(x)
        self.assertEqual(output.shape, (2, 3, 64, 64))

    def test_attention_gate(self):
        attention_gate = AttentionGate(32, 64, 128)
        x1 = torch.randn(2, 32, 32, 32)
        x2 = torch.randn(2, 64, 32, 32)
        output = attention_gate(x1, x2)
        self.assertEqual(output.shape, (2, 32, 32, 32))

    def test_attention_concat(self):
        attention_concat = AttentionConcat(32, 64)
        x1 = torch.randn(2, 32, 32, 32)
        x2 = torch.randn(2, 64, 32, 32)
        output = attention_concat(x1, x2)
        self.assertEqual(output.shape, (2, 96, 32, 32))

    def test_conv2d_block(self):
        conv_block = Conv2dBlock(3, 16, True, 0.3, "spatial", (3, 3), "relu", "same")
        x = torch.randn(2, 3, 32, 32)
        output = conv_block(x)
        self.assertEqual(output.shape, (2, 16, 32, 32))


if __name__ == "__main__":
    unittest.main()
