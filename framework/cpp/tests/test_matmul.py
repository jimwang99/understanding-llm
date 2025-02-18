import numpy as np
import ullm
import unittest


class TestMatmulFp32(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.matmul = ullm.matmul_fp32
        self.matmulT = ullm.matmulT_fp32

    def test_matmul(self):
        x = np.linspace(0, 1, 6, dtype=self.dtype).reshape([2, 3])
        y = np.linspace(1, 2, 12, dtype=self.dtype).reshape([3, 4])

        z = self.matmul(x, y)
        self.assertEqual(z.shape, [2, 4])
        r = np.dot(x, y)
        self.assertTrue(np.allclose(z, r), f"{z} != {r}")

        y = y.T
        z = self.matmulT(x, y)
        self.assertEqual(z.shape, [2, 4])
        r = np.dot(x, y.T)
        self.assertTrue(np.allclose(z, r), f"{z} != {r}")

        for _ in range(10):
            shape = np.random.randint(1, 10, size=3)
            n = shape[0]
            m = shape[1]
            k = shape[2]

            x = np.random.rand((n, k), dtype=self.dtype)
            y = np.random.rand((k, m), dtype=self.dtype)

            z = self.matmul(x, y)
            self.assertEqual(z.shape, [n, m])
            r = np.dot(x, y)
            self.assertTrue(np.allclose(z, r), f"{z} != {r}")

            y = y.T
            z = self.matmulT(x, y)
            self.assertEqual(z.shape, [n, m])
            r = np.dot(x, y.T)
            self.assertTrue(np.allclose(z, r), f"{z} != {r}")
