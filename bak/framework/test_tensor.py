import torch
import test_tensor_lib as lib


def test_tensor4_init():
    def _test(a: torch.Tensor, dtype: torch.dtype):
        name = f"{dtype}".split(".")[1]
        func = getattr(lib, f"test_tensor4_init_{name}")
        z = func(a)
        assert z.shape == a.shape, f"{z.shape=} {a.shape=}"
        assert z.dtype == a.dtype, f"{z.dtype=} {a.dtype=}"
        assert z == a, f"{z=} {a=}"

    for dtype in [torch.float32, torch.int32]:
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            a = torch.arange(*shape, dtype=dtype)
            _test(a, dtype)

            a = torch.randn(*shape, dtype=dtype)
            _test(a, dtype)
