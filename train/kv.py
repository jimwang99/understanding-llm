# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# minimalist version llama2

# References:
# https://github.com/facebookresearch/llama/blob/main/llama/model.py
# https://github.com/karpathy/llama2.c/blob/master/model.py

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from loguru import logger

NDEBUG = os.environ["NDEBUG"] == "True"


@dataclass
class HyperParams:  # default hyper-parameters for llama2 7B model
    """Hyper-parameters of the model. The default value is for llama2-7B
    The namings are different from OSS llama/model.py,
    because I want to be consistent with my diagram
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

    # dropout: float = 0.0

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


hparams = {}
hparams["260K"] = HyperParams(
    D=64, Nl=5, Nh=8, Nkv=4, V=512, Dh=-1, E=1e-05, Lm=512, _multiple_of=4
)
hparams["110M"] = HyperParams(
    D=768, Nl=12, Nh=12, Nkv=12, V=32000, Dh=-1, E=1e-05, Lm=1024, _multiple_of=32
)


class RMSNorm(torch.nn.Module):
    def __init__(self, D: int, E: float):
        super().__init__()
        self.D = D
        self.E = E
        self.weight = torch.nn.Parameter(torch.ones(self.D))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.E)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class Model(torch.nn.Module):
    def __init__(self, hparam: HyperParams) -> None:
        super().__init__()

        self.hparam = hparam

        D = hparam.D
        Nl = hparam.Nl
        Nh = hparam.Nh
        Nkv = hparam.Nkv
        V = hparam.V
        Dh = hparam.Dh
        E = hparam.E
        Lm = hparam.Lm  # noqa: F841

        if not NDEBUG:
            logger.debug("Creating Model:" + repr(hparam))

        # -------------------------------------------------------------------
        # define model

        # embedding layer
        self.embedding = torch.nn.Embedding(V, D)

        # transformer
        self.layers = torch.nn.ModuleList()
        for _ in range(Nl):
            layer = torch.nn.ModuleDict()
            self.layers.append(layer)

            layer["attention_norm"] = RMSNorm(D, E)

            # Multi-head attention MHA
            ## Projection of inputs in MHA
            layer["projection_q"] = torch.nn.Linear(D, D, bias=False)
            layer["projection_k"] = torch.nn.Linear(D, D * Nkv // Nh, bias=False)
            layer["projection_v"] = torch.nn.Linear(D, D * Nkv // Nh, bias=False)

            ## Projection of output in MHA
            layer["projection_a"] = torch.nn.Linear(D, D, bias=False)

            # Feed-forward network FFN
            layer["ffn_norm"] = RMSNorm(D, E)

            ## Projections inside FFN
            layer["projection_gate"] = torch.nn.Linear(D, Dh, bias=False)
            layer["projection_up"] = torch.nn.Linear(D, Dh, bias=False)
            layer["projection_down"] = torch.nn.Linear(Dh, D, bias=False)

        # output normalization
        self.output_norm = RMSNorm(D, E)

        # model head
        self.output_linear = torch.nn.Linear(D, V, bias=False)

        if not NDEBUG:
            self._debug_fpath = "/tmp/tinystories_debug.pt"
            if os.path.isfile(self._debug_fpath):
                logger.debug(f"loading debug info from {self._debug_fpath}")
                self._debug = torch.load(self._debug_fpath)
            else:
                self._debug = {}

    def _trace_and_check(
        self,
        tensor: torch.Tensor,
        name: str,
        shape: Optional[Sequence[int]] = None,
        layer_idx: Optional[int] = None,
        save_to_file: bool = False,
    ) -> None:
        if not NDEBUG:
            prefix = f"layer-{layer_idx} " if layer_idx else ""
            logger.trace(f"{prefix}{name}.shape={tensor.shape}")
            if shape:
                assert (
                    tensor.shape == shape
                ), f"{name}.shape={tensor.shape} != {shape} while {self.hparam=}"
            if save_to_file:
                if name not in self._debug.keys():
                    self._debug[name] = []
                self._debug[name].append(tensor)

    def __del__(self) -> None:
        if not NDEBUG:
            logger.debug(f"saving debug info to {self._debug_fpath}")
            torch.save(self._debug, self._debug_fpath)

    def forward(
        self,
        ti: torch.Tensor,  # input tokens
        cos: torch.Tensor,  # pre-computed cosine constant
        sin: torch.Tensor,  # pre-computed sine constant
        m: torch.Tensor,  # pre-computed mask matrix
        # ck: torch.Tensor,  # K cache buffer          NOTE: KC: Dissable kv_cache for ExecuTorch compile
        # cv: torch.Tensor,  # V cache buffer          NOTE: KC: Dissable kv_cache for ExecuTorch compile
        # Lc: int,  # existing cache length            NOTE: KC: Dissable kv_cache for ExecuTorch compile
        # use_kv_cache: bool = True,                   NOTE: KC: Dissable kv_cache for ExecuTorch compile
    ) -> torch.Tensor:
        D = self.hparam.D
        Nl = self.hparam.Nl  # noqa: F841
        Nh = self.hparam.Nh
        Nkv = self.hparam.Nkv
        V = self.hparam.V
        Dh = self.hparam.Dh
        E = self.hparam.E  # noqa: F841
        Lm = self.hparam.Lm  # noqa: F841

        B, L = ti.shape
        self._trace_and_check(ti, "ti")

        # KC: Dissable kv_cache for ExecuTorch compile until caching for ExecuTorch is figured out.
        Lc = 0  # kcoopman
        # use_kv_cache = False  # kcoopman
        # end of kv_cache disable

        La = Lc + L  # full context length

        self._trace_and_check(cos, "cos", (1, L, 1, D // Nh // 2))
        self._trace_and_check(sin, "sin", (1, L, 1, D // Nh // 2))
        self._trace_and_check(m, "m", (1, 1, L, La))

        if not NDEBUG:
            logger.debug(f"{ti.shape=} {cos.shape=} {sin.shape=} {m.shape=}")
            # if use_kv_cache:
            # logger.debug(f"{ck.shape=} {cv.shape=} {Lc=}")

        # -------------------------------------------------------------------
        # model inference

        # input embeddings
        xi = self.embedding(ti)
        self._trace_and_check(xi, "xi", (B, L, D))

        # transformer
        for _layer_idx, layer in enumerate(self.layers):  # disable kv cache

            # Attention norm
            xi1 = layer["attention_norm"](xi)
            self._trace_and_check(xi1, "xi1", (B, L, D))

            # ===============================================================
            # MHA

            # Projection
            q = layer["projection_q"](xi1)
            k = layer["projection_k"](xi1)
            v = layer["projection_v"](xi1)
            self._trace_and_check(q, "q", (B, L, D))
            self._trace_and_check(k, "k", (B, L, D * Nkv // Nh))
            self._trace_and_check(v, "v", (B, L, D * Nkv // Nh))

            q1 = q.view(B, L, Nh, D // Nh)
            k1 = k.view(B, L, Nkv, D // Nh)
            v1 = v.view(B, L, Nkv, D // Nh)

            # ---------------------------------------------------------------
            # RoPE

            # To complex
            q1 = q1.reshape(B, L, Nh, -1, 2)
            k1 = k1.reshape(B, L, Nkv, -1, 2)
            qr, qi = q1.unbind(-1)
            kr, ki = k1.unbind(-1)
            self._trace_and_check(qr, "qr", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(qi, "qi", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(kr, "kr", (B, L, Nkv, D // Nh // 2))
            self._trace_and_check(ki, "ki", (B, L, Nkv, D // Nh // 2))

            # Rotate
            qr1 = qr * cos - qi * sin
            qi1 = qr * sin + qi * cos
            kr1 = kr * cos - ki * sin
            ki1 = kr * sin + ki * cos
            self._trace_and_check(qr1, "qr1", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(qi1, "qi1", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(kr1, "kr1", (B, L, Nkv, D // Nh // 2))
            self._trace_and_check(ki1, "ki1", (B, L, Nkv, D // Nh // 2))

            # Merge
            q2 = torch.stack([qr1, qi1], dim=-1).flatten(3)
            k2 = torch.stack([kr1, ki1], dim=-1).flatten(3)
            self._trace_and_check(q2, "q2", (B, L, Nh, D // Nh))
            self._trace_and_check(k2, "k2", (B, L, Nkv, D // Nh))

            # ---------------------------------------------------------------
            # KV Cache
            """
            if use_kv_cache:
                # prefill phase: save the full context
                # decode phase: append 1 token
                ck[layer_idx, :B, Lc:La, :, :] = k2
                cv[layer_idx, :B, Lc:La, :, :] = v1
                if Lc > 0:
                    # decode phase: get the full context
                    k2 = ck[layer_idx, :B, :La, :, :].view(B, La, Nkv, D // Nh)
                    v1 = cv[layer_idx, :B, :La, :, :].view(B, La, Nkv, D // Nh)
            """
            # ---------------------------------------------------------------
            # GQA

            self._trace_and_check(v1, "v1c", (B, La, Nkv, D // Nh))
            self._trace_and_check(k2, "k2c", (B, La, Nkv, D // Nh))

            kx = k2.repeat_interleave(repeats=(Nh // Nkv), dim=2)
            vx = v1.repeat_interleave(repeats=(Nh // Nkv), dim=2)
            self._trace_and_check(kx, "kx", (B, La, Nh, D // Nh))
            self._trace_and_check(kx, "vx", (B, La, Nh, D // Nh))

            qt = q2.transpose(1, 2)
            kt = kx.transpose(1, 2)
            vt = vx.transpose(1, 2)
            self._trace_and_check(qt, "qt", (B, Nh, L, D // Nh))
            self._trace_and_check(kt, "kt", (B, Nh, La, D // Nh))
            self._trace_and_check(vt, "vt", (B, Nh, La, D // Nh))

            # scaled dot product attention
            kt1 = kt.transpose(2, 3)
            self._trace_and_check(kt1, "kt1", (B, Nh, D // Nh, La))

            a = torch.matmul(qt, kt1) / math.sqrt(D // Nh)
            self._trace_and_check(a, "a", (B, Nh, L, La))

            am = a + m
            self._trace_and_check(am, "am", (B, Nh, L, La))

            as_ = torch.nn.functional.softmax(am, dim=-1)
            self._trace_and_check(as_, "as", (B, Nh, L, La))

            sa = torch.matmul(as_, vt)
            self._trace_and_check(sa, "sa", (B, Nh, L, D // Nh))

            # concate
            sac = sa.transpose(1, 2).contiguous()
            sac1 = sac.view(B, L, D)
            self._trace_and_check(sac1, "sac1", (B, L, D))

            # self-attention projection
            sap = layer["projection_a"](sac1)
            self._trace_and_check(sap, "sap", (B, L, D))

            # ---------------------------------------------------------------

            ha = sap + xi  # residual
            self._trace_and_check(ha, "ha", (B, L, D))

            hi = layer["ffn_norm"](ha)
            self._trace_and_check(hi, "hi", (B, L, D))

            # ---------------------------------------------------------------
            # FFN

            hg = layer["projection_gate"](hi)
            self._trace_and_check(hg, "hg", (B, L, Dh))

            hu = layer["projection_up"](hi)
            self._trace_and_check(hu, "hu", (B, L, Dh))

            hs = torch.nn.functional.silu(hg)
            self._trace_and_check(hs, "hs", (B, L, Dh))

            hm = hs * hu
            self._trace_and_check(hm, "hm", (B, L, Dh))

            hd = layer["projection_down"](hm)
            self._trace_and_check(hd, "hd", (B, L, D))

            # ---------------------------------------------------------------

            hf = hd + ha  # residual
            self._trace_and_check(hf, "hf", (B, L, D))

            xi = hf  # for the next layer

        if not self.training:
            # slice the last token which is the only one needed in inferece
            # JW: donot combine the following 2 lines, because of ExecuTorch limitation
            # JW: also [:, [-1], :] doesn't work with ExecuTorch
            hf = hf[:, -1, :].view(B, 1, D)

        xo = self.output_norm(hf)

        lo = self.output_linear(xo)

        if not self.training:
            self._trace_and_check(xo, "xo", (B, 1, D))
            self._trace_and_check(lo, "lo", (B, 1, V))
        else:
            self._trace_and_check(xo, "xo", (B, L, D))
            self._trace_and_check(lo, "lo", (B, L, V))

        return lo


class ModelWrapperForInference:
    """A wrapper around `Model` for inference
    because `Model` is stateless and need to be modified for ExecuTorch's convenience
    """

    def __init__(
        self, model: Model, use_kv_cache: bool = True, max_batch_size: int = 1
    ):
        self.model = model
        self.use_kv_cache = use_kv_cache

        self.hparam = model.hparam
        self._precompute_freqs_cis()
        self._precompute_mask()
        self.cache_k, self.cache_v = self._init_cache(max_batch_size)
        self.cache_len = 0

    def _precompute_freqs_cis(self, theta_base: float = 10000.0):
        D = self.hparam.D
        Nh = self.hparam.Nh
        Lm = self.hparam.Lm

        theta = 1.0 / (
            theta_base
            ** (
                torch.arange(0, D // Nh, 2)[: (D // Nh // 2)].to(torch.float32)
                / (D // Nh)
            )
        )
        abs_pos = torch.arange(Lm)
        freqs = torch.outer(abs_pos, theta)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def _precompute_mask(self):
        Lm = self.hparam.Lm

        mf = torch.full((1, 1, Lm, Lm), float("-inf"), dtype=torch.float32)
        self.m = torch.triu(mf, diagonal=1).view(1, 1, Lm, Lm)

    def _init_cache(self, max_batch_size: int):
        Bm = max_batch_size
        D = self.hparam.D
        Nh = self.hparam.Nh
        Nkv = self.hparam.Nkv
        Nl = self.hparam.Nl
        Lm = self.hparam.Lm

        cache_k = torch.empty((Nl, Bm, Lm, Nkv, D // Nh), dtype=torch.float32)
        cache_v = torch.empty((Nl, Bm, Lm, Nkv, D // Nh), dtype=torch.float32)
        return (cache_k, cache_v)

    def preproc(self, ti: torch.Tensor) -> Sequence[torch.Tensor]:
        """Prepare cis and mask outside of model's forward for executorch"""
        B, L = ti.shape
        D = self.hparam.D
        Nh = self.hparam.Nh
        Lc = self.cache_len

        self.saved_ti = ti

        # prepare input tensor
        if self.use_kv_cache and self.cache_len > 0:
            # in decode phase only take the last input token
            # otherwise in prefill phase, or KV cache disabled, use full input tokens
            ti = ti[:, [-1]]
            L = 1
        La = Lc + L

        # prepare cis
        cos = self.cos[Lc:La].view((1, L, 1, D // Nh // 2))  # type: ignore
        sin = self.sin[Lc:La].view((1, L, 1, D // Nh // 2))  # type: ignore

        # prepare mask matrix
        m = self.m[
            :, :, Lc:La, :La
        ]  # when KV cache is enabled, need to select the right row

        return (ti.contiguous(), cos.contiguous(), sin.contiguous(), m.contiguous())

    @torch.inference_mode()
    def forward(self, ti: torch.Tensor) -> torch.Tensor:
        # preproc
        assert ti.ndim == 2
        B, L = ti.shape
        if self.use_kv_cache:
            assert (
                B <= self.cache_k.shape[1]
            ), f"batch size exceeded {B=} {self.cache_k.shape=}"
        assert (
            L <= self.hparam.Lm
        ), f"input sequence length exceeded {L=} {self.cache_len=} {self.hparam.Lm=}"
        ti, cos, sin, m = self.preproc(ti)

        # inference
        lo = self.model.forward(
            ti,
            cos,
            sin,
            m,
            # self.cache_k,                        NOTE: KC: Dissable kv_cache for ExecuTorch compile
            # self.cache_v,                        NOTE: KC: Dissable kv_cache for ExecuTorch compile
            # self.cache_len,                      NOTE: KC: Dissable kv_cache for ExecuTorch compile
            # use_kv_cache=self.use_kv_cache,      NOTE: KC: Dissable kv_cache for ExecuTorch compile
        )

        if self.use_kv_cache:
            self.cache_len += L if self.cache_len == 0 else 1
        return lo


def create_model(
    model_size: str, use_kv_cache: bool = True
) -> ModelWrapperForInference:
    try:
        from export import export_from_tinyllamas
    except ImportError:
        from .export import export_from_tinyllamas

    fpaths = {
        "260K": "/tmp/tetris2_applications_tinystories/tinyllamas/stories260K.pt",
        "15M": "/tmp/tetris2_applications_tinystories/tinyllamas/stories15M.pt",
        "42M": "/tmp/tetris2_applications_tinystories/tinyllamas/stories42M.pt",
        "110M": "/tmp/tetris2_applications_tinystories/tinyllamas/stories110M.pt",
    }

    logger.debug(
        f"create Model() instance and load state_dict from {fpaths[model_size]}"
    )
    state_dict = export_from_tinyllamas(fpath_in=fpaths[model_size])
    mdl = Model(hparam=hparams[model_size])
    mdl.load_state_dict(state_dict)
    mdl.eval()
    return ModelWrapperForInference(mdl, use_kv_cache=use_kv_cache)
