AVAIL_MODELS = (
    "karpathy_tinystories260k",
    "karpathy_tinystories15m",
    "karpathy_tinystories42m",
    "karpathy_tinystories110m",
)

import torch

from perf import PerfMonitor
from loguru import logger
from typing import Optional, Sequence, Tuple
from dataclasses import dataclass
from huggingface_hub import hf_hub_download


@dataclass
class HyperParam:
    """Model hyperparameters
    The default value is for llama2-7B
    """

    D: int = 4096  # dimension of embedding, dim
    Nl: int = 32  # num of transformer layers
    Nh: int = 32  # num of heads, n_heads
    Nkv: int = -1  # num of KV heads, n_kv_heads
    V: int = 32000  # size of the vocabulary, vocab_size
    Dh: int = (
        -1
    )  # hidden dimension of FFN layer, default is calculated in __post_init__
    E: float = (
        1e-06  # a small number that is used in RMS normalization to avoid divide by 0
    )

    Lm: int = 2048  # max sequence length

    # following params are used to calculate Dh
    _multiple_of: int = 256  # make hidden dimension the multiple of large power of 2
    _ffn_dim_multiplier: float = (
        1.0  # custom multiplier for hidden dimension of FFN layer
    )

    _dropout: float = 0.0

    tokenizer_name: str = "sentencepiece_tok32k"

    def __post_init__(self):
        assert self.D % self.Nh == 0

        if self.Nkv == -1:
            self.Nkv = self.Nh
        assert self.Nh % self.Nkv == 0, f"{self.Nh=} {self.Nkv=}"

        if self.Dh == -1:
            self.Dh = 4 * self.D
            self.Dh = int(2 * self.Dh / 3)
            self.Dh = self._multiple_of * (
                (self.Dh + self._multiple_of - 1) // self._multiple_of
            )  # round up to N * self._multiple_of

    def print_tensor_sizes(self, B: int, L: int):
        D = self.hparam.D
        Nl = self.hparam.Nl
        Nh = self.hparam.Nh
        Nkv = self.hparam.Nkv
        V = self.hparam.V
        Dh = self.hparam.Dh
        E = self.hparam.E
        Lm = self.hparam.Lm

        size = {}
        size["Ti"] = [B, L]
        size["Xi"] = [B, L, D]
        size["Xi1"] = [B, L, D]
        size["Q"] = [B, L, D]
        size["K"] = [B, L, D * Nkv // Nh]
        size["V"] = [B, L, D * Nkv // Nh]
        size["Q1"] = [B, L, Nh, D // Nh]
        size["K1"] = [B, L, Nkv, D // Nh]
        size["V1"] = [B, L, Nkv, D // Nh]

        size["Qt"] = [B, Nh, L, D // Nh]
        size["Kt"] = [B, Nh, L, D // Nh]
        size["Vt"] = [B, Nh, L, D // Nh]

        size["Kt1"] = [B, Nh, D // Nh, L]

        size["A"] = [B, Nh, L, L]
        size["M"] = [B, Nh, L, L]
        size["Am"] = [B, Nh, L, L]
        size["As"] = [B, Nh, L, L]
        size["Sa"] = [B, Nh, L, D // Nh]

        logger.info("=== Hyper params ===")
        logger.info(f"B: batch size = {B}")
        logger.info(f"L: sequence length = {L}")
        logger.info(f"D: embedding dimension = {self.D}")
        logger.info(f"Nl: num of transformer layers = {self.Nl}")
        logger.info(f"Nh: num of heads = {self.Nh}")
        logger.info(f"Nkv: num of KV heads = {self.Nkv}")
        logger.info(f"V: vocab size = {self.V}")
        logger.info(f"Dh: hidden dimension of FFN layer = {self.Dh}")
        logger.info(f"Lm: max sequence length = {self.Lm}")

        logger.info(f"=== Tensor sizes ===")
        for name in size:
            logger.info(f"{name:<4} = {size[name]}")
        logger.info(f"====================")


def make_model(
    name: str,
    is_train: bool = True,
    is_debug: bool = False,
    is_compiled: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    logger.debug(f"{name=} {is_train=} {is_debug=} {is_compiled=} {device=} {dtype=}")
    assert name in AVAIL_MODELS, f"{name=} {AVAIL_MODELS=}"

    pm = PerfMonitor(logger, "make_model")

    if name == "karpathy_tinystories260k":
        hparam = HyperParam(
            D=64,
            Nl=5,
            Nh=8,
            Nkv=4,
            V=512,
            Lm=512,
            _multiple_of=4,
            tokenizer_name="sentencepiece_tok512",
        )
    elif name == "karpathy_tinystories15m":
        hparam = HyperParam(
            D=288,
            Nl=6,
            Nh=6,
            Nkv=6,
            V=32000,
            Lm=256,
            _multiple_of=32,
            tokenizer_name="sentencepiece_tok32k",
        )
    elif name == "karpathy_tinystories42m":
        hparam = HyperParam(
            D=512,
            Nl=8,
            Nh=8,
            Nkv=8,
            V=32000,
            Lm=1024,
            _multiple_of=32,
            tokenizer_name="sentencepiece_tok32k",
        )
    elif name == "karpathy_tinystories110m":
        hparam = HyperParam(
            D=768,
            Nl=12,
            Nh=12,
            Nkv=12,
            V=32000,
            Lm=1024,
            _multiple_of=32,
            tokenizer_name="sentencepiece_tok32k",
        )
    logger.debug(f"{name=} {hparam=}")

    if is_debug:
        from llama_debug import Model
    else:
        from llama import Model

    model = Model(name=name, hparam=hparam, device=device, dtype=dtype)

    if not is_train:
        fpath = hf_hub_download(
            repo_id="jimwang99/TinyStoriesV2-Tokenized",
            repo_type="dataset",
            filename=f"{name}.pt",
        )
        state_dict = torch.load(fpath)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        model.train()

    if is_compiled:
        model = torch.compile(model)

    model.to(device=device, dtype=dtype)

    pm.once()
    return model


def test_model():
    import time

    def eval(m):
        num_params = sum(p.numel() for p in m.parameters())
        logger.debug(
            f"Number of parameters of model {n} = {num_params / 1000000} Million"
        )

        m.eval()
        ti = torch.randint(0, m.hparam.V, (1, m.hparam.Lm)).cuda()
        logger.debug(f"{ti.dtype=}")
        with torch.inference_mode():
            pm = PerfMonitor(logger, "inference")
            to = m.forward(ti)  # warmup
            pm.once()
            start_time = time.time()
            to = m.forward(ti)
            end_time = time.time()
        logger.debug(
            f"{ti.shape=} {to.shape=} inference_latency={(end_time - start_time) * 1000}ms"
        )

    n = "karpathy_tinystories260k"
    m = make_model(n, is_train=False)
    eval(m)


if __name__ == "__main__":
    hparam = HyperParam(
        D=768,
        Nl=12,
        Nh=12,
        Nkv=12,
        V=32000,
        Lm=1024,
        _multiple_of=32,
        tokenizer_name="sentencepiece_tok32k",
    )
    hparam.print_tensor_sizes(4, 1024)
