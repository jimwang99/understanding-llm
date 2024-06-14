import math
import torch

from model import HyperParam
from typing import Optional, Sequence, Tuple
from loguru import logger

logger.add("/tmp/understanding-llm-train-llama.log", level="TRACE")


def check_and_trace(
    tensor: torch.Tensor,
    name: str,
    shape: Optional[Sequence[int]] = None,
    layer_idx: Optional[int] = None,
):
    prefix = f"layer-{layer_idx} " if layer_idx else ""
    logger.trace(f"{prefix}{name}.shape={tensor.shape} {name}.dtype={tensor.dtype}")
    if shape:
        assert tensor.shape == shape, f"{name}.shape={tensor.shape} != {shape}"


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
    """Vanilla model of llama architecture"""

    def __init__(
        self,
        name: str,
        hparam: HyperParam,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        logger.debug(f"{name=} {hparam=} {device=} {dtype=}")

        self.name = name
        self.hparam = hparam
        self.device = device
        self.dtype = dtype

        D = hparam.D
        Nl = hparam.Nl
        Nh = hparam.Nh
        Nkv = hparam.Nkv
        V = hparam.V
        Dh = hparam.Dh
        E = hparam.E
        Lm = hparam.Lm

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

        # -------------------------------------------------------------------
        # prepare cos|sin(mθ)
        cos, sin = self._precompute_freqs_cis(D, Nh, Lm)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        check_and_trace(cos, "cosm", (Lm, D // Nh // 2))
        check_and_trace(sin, "sinm", (Lm, D // Nh // 2))

        # # prepare mask
        # mf = torch.triu(
        #     torch.full((1, 1, Lm, Lm), float("-inf"), device=device, dtype=dtype),
        #     diagonal=1,
        # )
        # self.register_buffer("mf", mf, persistent=False)
        # # > ifdef 
        # check_and_trace(mf, "mf", (1, 1, Lm, Lm))
        # # > endif

        self._layer_idx = -1

    def _precompute_freqs_cis(
        self, D: int, Nh: int, Lm: int, theta_base: float = 10000.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = 1.0 / (
            theta_base
            ** (
                torch.arange(0, D // Nh, 2)[: (D // Nh // 2)].to(torch.float32)
                / (D // Nh)
            )
        )
        abs_pos = torch.arange(Lm)
        freqs = torch.outer(abs_pos, theta)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return (freqs_cos, freqs_sin)

    def forward(self, ti: torch.Tensor) -> torch.Tensor:
        D = self.hparam.D
        Nl = self.hparam.Nl
        Nh = self.hparam.Nh
        Nkv = self.hparam.Nkv
        V = self.hparam.V
        Dh = self.hparam.Dh
        E = self.hparam.E
        Lm = self.hparam.Lm

        assert ti.ndim == 2, f"{ti.shape=}"
        B, L = ti.shape
        check_and_trace(ti, "ti")
        assert L <= Lm

        # -------------------------------------------------------------------
        # prepare

        # prepare cis
        cos = self.cos[:L].view((1, L, 1, D // Nh // 2))  # type: ignore
        sin = self.sin[:L].view((1, L, 1, D // Nh // 2))  # type: ignore
        check_and_trace(cos, "cos")
        check_and_trace(sin, "sin")

        # prepare mask matrix
        # m = self.mf[:, :, :L, :L].contiguous()
        # m = self.mf[:, :, :L, :L]
        m = torch.triu(
            torch.full(
                (1, 1, L, L), float("-inf"), device=self.device, dtype=self.dtype
            ),
            diagonal=1,
        )
        check_and_trace(m, "m", (1, 1, L, L))

        # -------------------------------------------------------------------
        # model inference

        # Embeddings
        xi = self.embedding(ti)
        check_and_trace(xi, "xi", (B, L, D))

        # transformer
        for layer_idx, layer in enumerate(self.layers):
            logger.trace(f"> transformer layer {layer_idx}")
            self._layer_idx = layer_idx

            # Attention norm
            xi1 = layer["attention_norm"](xi)
            check_and_trace(xi1, "xi1", (B, L, D))

            # ===============================================================
            # MHA

            # Projection
            q = layer["projection_q"](xi1)
            k = layer["projection_k"](xi1)
            v = layer["projection_v"](xi1)
            check_and_trace(q, "q", (B, L, D))
            check_and_trace(k, "k", (B, L, D * Nkv // Nh))
            check_and_trace(v, "v", (B, L, D * Nkv // Nh))

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
            check_and_trace(qr, "qr", (B, L, Nh, D // Nh // 2))
            check_and_trace(qi, "qi", (B, L, Nh, D // Nh // 2))
            check_and_trace(kr, "kr", (B, L, Nkv, D // Nh // 2))
            check_and_trace(ki, "ki", (B, L, Nkv, D // Nh // 2))

            # Rotate
            qr1 = qr * cos - qi * sin
            qi1 = qr * sin + qi * cos
            kr1 = kr * cos - ki * sin
            ki1 = kr * sin + ki * cos
            check_and_trace(qr1, "qr1", (B, L, Nh, D // Nh // 2))
            check_and_trace(qi1, "qi1", (B, L, Nh, D // Nh // 2))
            check_and_trace(kr1, "kr1", (B, L, Nkv, D // Nh // 2))
            check_and_trace(ki1, "ki1", (B, L, Nkv, D // Nh // 2))

            # Merge
            q2 = torch.stack([qr1, qi1], dim=-1).flatten(3)
            k2 = torch.stack([kr1, ki1], dim=-1).flatten(3)
            check_and_trace(q2, "q2", (B, L, Nh, D // Nh))
            check_and_trace(k2, "k2", (B, L, Nkv, D // Nh))

            # ---------------------------------------------------------------
            # GQA

            kx = k2.repeat_interleave(repeats=(Nh // Nkv), dim=2)
            vx = v1.repeat_interleave(repeats=(Nh // Nkv), dim=2)
            check_and_trace(kx, "kx", (B, L, Nh, D // Nh))
            check_and_trace(kx, "vx", (B, L, Nh, D // Nh))

            qt = q2.transpose(1, 2)
            kt = kx.transpose(1, 2)
            vt = vx.transpose(1, 2)
            check_and_trace(qt, "qt", (B, Nh, L, D // Nh))
            check_and_trace(kt, "kt", (B, Nh, L, D // Nh))
            check_and_trace(vt, "vt", (B, Nh, L, D // Nh))

            # scaled dot product attention
            kt1 = kt.transpose(2, 3)
            check_and_trace(kt1, "kt1", (B, Nh, D // Nh, L))

            a = torch.matmul(qt, kt1) / math.sqrt(D // Nh)
            check_and_trace(a, "a", (B, Nh, L, L))

            # am = a + m[:, :, :L, :L]
            am = a + m
            check_and_trace(am, "am", (B, Nh, L, L))

            as_ = torch.nn.functional.softmax(am, dim=-1)
            check_and_trace(as_, "as", (B, Nh, L, L))

            sa = torch.matmul(as_, vt)
            check_and_trace(sa, "sa", (B, Nh, L, D // Nh))

            # concate
            sac = sa.transpose(1, 2).contiguous()
            sac1 = sac.view(B, L, D)
            check_and_trace(sac1, "sac1", (B, L, D))

            # self-attention projection
            sap = layer["projection_a"](sac1)
            check_and_trace(sap, "sap", (B, L, D))

            # ---------------------------------------------------------------

            ha = sap + xi  # residual
            check_and_trace(ha, "ha", (B, L, D))

            hi = layer["ffn_norm"](ha)
            check_and_trace(hi, "hi", (B, L, D))

            # ---------------------------------------------------------------
            # FFN

            hg = layer["projection_gate"](hi)
            check_and_trace(hg, "hg", (B, L, Dh))

            hu = layer["projection_up"](hi)
            check_and_trace(hu, "hu", (B, L, Dh))

            hs = torch.nn.functional.silu(hg)
            check_and_trace(hs, "hs", (B, L, Dh))

            hm = hs * hu
            check_and_trace(hm, "hm", (B, L, Dh))

            hd = layer["projection_down"](hm)
            check_and_trace(hd, "hd", (B, L, D))

            # ---------------------------------------------------------------

            hf = hd + ha  # residual
            check_and_trace(hf, "hf", (B, L, D))

            xi = hf  # for the next layer

        xo = self.output_norm(hf)
        check_and_trace(xo, "xo", (B, L, D))

        lo = self.output_linear(xo[:, :, :])
        check_and_trace(lo, "lo", (B, L, V))

        return lo
