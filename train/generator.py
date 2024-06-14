import torch

from perf import PerfMonitor
from model import make_model
from loguru import logger
from typing import Callable, Iterator, List
from tokenizer import make_tokenizer


class Generator:
    def __init__(
        self,
        model_name: str,
        prompt: str = "",
        max_new_len: int = 1024,
        temperature: float = 0.6,
        sampling_method: str = "top_k",
    ) -> None:
        self.device = "cuda"
        # self.model = make_model(
        #     model_name,
        #     is_train=False,
        #     is_debug=False,
        #     is_compiled=True,
        #     # is_compiled=False,
        #     device=self.device,
        #     dtype=torch.bfloat16,
        # )
        # self.model.eval()
        self.model = make_model(model_name, is_train=False, is_compiled=False)
        self.tokenizer = make_tokenizer(self.model.hparam.tokenizer_name)

        assert type(prompt) == str, type(prompt)
        assert max_new_len > 0, max_new_len
        assert temperature >= 0.0, temperature
        assert sampling_method in ["greedy", "top_k", "top_p"]
        sampling_method = "greedy" if temperature == 0.0 else sampling_method

        self.temperature = temperature
        self.sampling: Callable[[torch.Tensor, float], int] = {
            "greedy": Generator._sampling_greedy,
            "top_k": Generator._sampling_top_k,
            "top_p": Generator._sampling_top_p,
        }[sampling_method]

        # tokenize input
        self.tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        L = len(self.tokens)

        self.Lt = min(self.model.hparam.Lm, L + max_new_len)
        assert (
            self.Lt > L
        ), f"either prompt={L} is too long, or max_new_len={max_new_len} is too small"

    @staticmethod
    def _sampling_greedy(logits: torch.Tensor, temperature: float) -> int:
        return torch.argmax(logits[:, -1], dim=-1)

    @staticmethod
    def _sampling_top_k(
        logits: torch.Tensor, temperature: float, top_k: int = 300
    ) -> int:
        assert len(logits.shape) == 1, logits.shape
        V = logits.shape[0]

        top_k = min(top_k, V)

        logits = logits / temperature  # scale by temperature
        values, _ = torch.topk(logits, top_k)  # find top-k values
        logits[logits < values[-1]] = float(
            "-inf"
        )  # set unselected logit's value to -Infinite
        probs = torch.nn.functional.softmax(
            logits, dim=-1
        )  # to (normalized) probabilities
        new_token = (
            torch.multinomial(probs, num_samples=1).view(-1).to(torch.long).item()
        )  # one sample from the multinomial probability distribution
        assert type(new_token) == int, f"{new_token=} {type(new_token)=}"

        return new_token

    @staticmethod
    def _sampling_top_p(
        logits: torch.Tensor, temperature: float, top_p: float = 0.9
    ) -> int:
        assert len(logits.shape) == 1, logits.shape

        assert temperature > 0, temperature
        probs = torch.nn.functional.softmax(
            logits / temperature, dim=-1
        )  # to (normalized) probabilities

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)  # cumulative sum

        # filter the elements whose cumulative sum is larger than top_p
        mask = (probs_sum - probs_sort) > top_p
        probs_sort[mask] = 0.0

        # scale so that sum is 1.0
        probs_sort.div_(probs_sort.sum(dim=-1))

        new_token_idx = torch.multinomial(probs_sort, num_samples=1)
        new_token = torch.gather(probs_idx, -1, new_token_idx)

        return new_token.view(-1).to(torch.long).item()

    def _next(self) -> Iterator[List[int]]:
        L = len(self.tokens)
        V = self.model.hparam.V

        while L < self.Lt:
            with torch.inference_mode():
                logits = self.model.forward(
                    torch.tensor(
                        self.tokens, device=self.device, dtype=torch.long
                    ).view(1, -1)
                )  # inference with model
            logits = logits[:, [-1], :]  # only pick the last (future) token
            assert logits.shape == (1, 1, V), f"{logits.shape=} {L=} {V=}"
            new_token = self.sampling(logits.view(-1), self.temperature)  # sampling
            if new_token in [self.tokenizer.eos, self.tokenizer.bos]:
                return
            self.tokens.append(new_token)
            L = len(self.tokens)

            yield self.tokens

    def next(self) -> Iterator:
        for _ in self._next():
            yield self.tokenizer.decode(self.tokens)

    def all(self) -> str:
        logger.debug("> Start generation")
        pm = PerfMonitor(logger, "inference")
        for _ in self._next():
            pass
        pm.once()
        logger.info(f"Tokens per second = {len(self.tokens) / pm.acc_latency:.2f}")

        pm = PerfMonitor(logger, "decode")
        s = self.tokenizer.decode(self.tokens)
        pm.once()
        return s


def test_generator():
    g = Generator("karpathy_tinystories260k")
    logger.info(g.all())

    # g = Generator("karpathy_tinystories110m")
    # logger.info(g.all())
