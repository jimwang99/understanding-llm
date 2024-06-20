import torch
import torch.nn.functional as F
import numpy as np
import unittest

from loguru import logger

import conv2d

logger.add("/tmp/test_conv2d.log", level="TRACE")


def to_numpy(t: torch.Tensor, nchw_to_nhwc: bool = True) -> np.ndarray:
    if nchw_to_nhwc:
        t = t.permute(0, 2, 3, 1)
    return t.contiguous().numpy()


def from_numpy(a: np.ndarray, nhwc_to_nchw: bool = True) -> torch.Tensor:
    t = torch.from_numpy(a)
    if nhwc_to_nchw:
        t = t.permute(0, 3, 1, 2)
    return t


class TestBasicConv2d(unittest.TestCase):
    def test_3x3x4_over_7x7x4(self):
        # reference use pytorch
        tensor_input = torch.randint(
            low=0,
            high=10,
            size=(1, 4, 7, 7),
            dtype=torch.int64,
        )
        tensor_kernel = torch.randint(
            low=0,
            high=10,
            size=(2, 4, 3, 3),
            dtype=torch.int64,
        )
        tensor_output_ref = F.conv2d(
            tensor_input,
            tensor_kernel,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )

        np_input = to_numpy(tensor_input)
        np_kernel = to_numpy(tensor_kernel)

        np_output = conv2d.conv2d(np_input, np_kernel)
        tensor_output = from_numpy(np_output)

        logger.debug(f"{tensor_input.shape=}")
        logger.debug(f"{tensor_kernel.shape=}")
        logger.debug(f"{tensor_output.shape=}")
        logger.debug(f"{tensor_output_ref.shape=}")

        logger.debug(f"{np_input.shape=}")
        logger.debug(f"{np_kernel.shape=}")
        logger.debug(f"{np_output.shape=}")

        logger.trace(f"{tensor_input=}")
        logger.trace(f"{tensor_kernel=}")
        logger.trace(f"{tensor_output=}")
        logger.trace(f"{tensor_output_ref=}")

        logger.trace(f"{np_input=}")
        logger.trace(f"{np_kernel=}")
        logger.trace(f"{np_output=}")

        self.assertTrue(torch.equal(tensor_output, tensor_output_ref))


if __name__ == "__main__":
    unittest.main()
