import torch
import torch.nn.functional as F

import numpy as np


def test_linear_2d():
    Di = 2
    Do = 3

    def _test_linear_2d_with_sequence(Di: int, Do: int):
        x = torch.arange(Di, dtype=torch.float32)
        y = torch.arange(Di * Do, dtype=torch.float32).reshape(Do, Di)
        b = torch.ones(Do, dtype=torch.float32)
        z = F.linear(x, y, b)
        assert z.shape == (Do,), z.shape
