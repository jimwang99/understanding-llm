import numpy as np
import ullm
import unittest

from loguru import logger


class TestElemWiseFp32(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.add = ullm.add_fp32
        self.sub = ullm.sub_fp32
        self.mul = ullm.mul_fp32
        self.div = ullm.div_fp32
        self.add_scalar = ullm.add_scalar_fp32
        self.sub_scalar = ullm.sub_scalar_fp32
        self.mul_scalar = ullm.mul_scalar_fp32
        self.div_scalar = ullm.div_scalar_fp32

        self.numpy_op = {
            self.add: np.add,
            self.sub: np.subtract,
            self.mul: np.multiply,
            self.div: np.divide,
            self.add_scalar: np.add,
            self.sub_scalar: np.subtract,
            self.mul_scalar: np.multiply,
            self.div_scalar: np.divide,
        }

    def _test_op(self, op):
        x = np.array([1, 2, 3], dtype=self.dtype)
        y = np.array([4, 5, 6], dtype=self.dtype)
        z = op(x, y)
        self.assertEqual(z.shape, x.shape)
        r = self.numpy_op[op](x, y)
        self.assertTrue(np.allclose(z, r), f"{z} != {r}, {x=} {y=}")

        x = np.linspace(0, 1, 120, dtype=self.dtype).reshape([2, 3, 4, 5])
        y = np.linspace(1, 2, 120, dtype=self.dtype).reshape([2, 3, 4, 5])
        z = op(x, y)
        self.assertEqual(z.shape, x.shape)
        r = self.numpy_op[op](x, y)
        self.assertTrue(np.allclose(z, r), f"{z} != {r}")

        for _ in range(10):
            shape = np.random.randint(1, 10, size=5)
            logger.debug(f"{shape=} {shape.dtype=}")
            x = np.random.rand(*shape).astype(self.dtype)
            y = np.random.rand(*shape).astype(self.dtype)
            z = op(x, y)
            self.assertEqual(z.shape, x.shape)
            r = self.numpy_op[op](x, y)
            self.assertTrue(np.allclose(z, r), f"{z} != {r}")

    def test_add(self):
        self._test_op(self.add)

    def test_sub(self):
        self._test_op(self.sub)

    def test_mul(self):
        self._test_op(self.mul)

    def test_div(self):
        self._test_op(self.div)

    def _test_op_scalar(self, op):
        x = np.array([1, 2, 3], dtype=self.dtype)
        z = op(x, 4)
        self.assertEqual(z.shape, x.shape)
        r = self.numpy_op[op](x, 4)
        self.assertTrue(np.allclose(z, r), f"{z} != {r}")

        x = np.linspace(0, 1, 120, dtype=self.dtype).reshape([2, 3, 4, 5])
        z = op(x, 0.1)
        self.assertEqual(z.shape, x.shape)
        r = self.numpy_op[op](x, 0.1)
        self.assertTrue(np.allclose(z, r), f"{z} != {r}")

        for _ in range(10):
            shape = np.random.randint(1, 10, size=5)
            logger.debug(f"{shape=} {shape.dtype=}")
            x = np.random.rand(*shape).astype(self.dtype)
            z = op(x, 0.1)
            self.assertEqual(z.shape, x.shape)
            r = self.numpy_op[op](x, 0.1)
            self.assertTrue(np.allclose(z, r), f"{z} != {r}")

    def test_add_scalar(self):
        self._test_op_scalar(self.add_scalar)

    def test_sub_scalar(self):
        self._test_op_scalar(self.sub_scalar)

    def test_mul_scalar(self):
        self._test_op_scalar(self.mul_scalar)

    def test_div_scalar(self):
        self._test_op_scalar(self.div_scalar)
