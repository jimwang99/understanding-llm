import torch
import datasets

from perf import PerfMonitor
from loguru import logger
from typing import Dict, List, Tuple, Iterator


class Dataset:
    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        max_seq_length: int,
        batch_size: int,
        device: str,
    ):

        self.split = split
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

        self.dataset = datasets.load_dataset(
            "jimwang99/TinyStoriesV2-Tokenized", split=split
        )

        self.iter = iter(self)

    def summary(self) -> None:
        self.total_num_token = sum([len(x[self.tokenizer_name]) for x in self.dataset])
        self.total_num_iter = self.total_num_token // (
            self.max_seq_length * self.batch_size
        )
        logger.info(
            f"Dataset: # of entries = {len(self.dataset)}, # of tokens = {self.total_num_token}, # of iterations = {self.total_num_iter}"
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        batch_seq_length = self.max_seq_length * self.batch_size

        idx = 0
        buf = []
        while True:
            # get enough data from dataset
            while len(buf) < batch_seq_length and idx < len(self.dataset):
                buf.extend(self.dataset[idx][self.tokenizer_name])
                idx += 1

            # not enough data
            if len(buf) < batch_seq_length and idx >= len(self.dataset):
                raise StopIteration

            # construct a list of tokens from data
            tokens = []
            for batch in range(self.batch_size):
                start_idx = batch * self.max_seq_length
                end_idx = start_idx + self.max_seq_length
                tokens.append(buf[start_idx:end_idx])
            buf = buf[batch_seq_length:]

            # construct tensor
            tokens = torch.tensor(tokens)
            assert tokens.ndim == 2, f"{tokens.shape=}"

            # transfer tensor to device
            input_tokens = tokens[:, :-1].to(device=self.device, non_blocking=True)
            output_targets = tokens[:, 1:].to(device=self.device, non_blocking=True)

            # yield
            yield (input_tokens, output_targets)

    def __next__(self):
        try:
            return next(self.iter)
        except RuntimeError as e:
            logger.warning(e)
            self.shuffle()
            return next(self.iter)

    def shuffle(self):
        self.dataset.shuffle()
        self.dataset.flatten_indices()
        self.iter = iter(self)


def make_dataset(
    split: str,
    tokenizer_name: str,
    max_seq_length: int,
    batch_size: int,
    device: str,
) -> datasets.DatasetDict:
    logger.debug(
        f"{split=} {tokenizer_name=} {max_seq_length=} {batch_size=} {device=}"
    )
    ds = Dataset(split, tokenizer_name, max_seq_length, batch_size, device)
    return ds


def test_dataset():
    ds = make_dataset(
        "train",
        "r50k_base",
        max_seq_length=16,
        batch_size=2,
        device="cpu",
    )
    for i, (a, b) in enumerate(ds):
        assert a.shape == torch.Size([2, 15])
        assert b.shape == torch.Size([2, 15])
        logger.debug(f"{a=}")
        logger.debug(f"{b=}")
        if i == 2:
            break

    it = iter(ds)
    n = next(it)
    logger.debug(f"{n=}")
    n = next(it)
    logger.debug(f"{n=}")


def test_end():
    ds = make_dataset(
        "train",
        "r50k_base",
        max_seq_length=1000,
        batch_size=8,
        device="cpu",
    )
    ds.dataset = ds.dataset.select(range(1000))
    ds.summary()
    # for i, (a, b) in enumerate(ds):
    #     assert a.shape == torch.Size([8, 999])
    #     assert b.shape == torch.Size([8, 999])
    #     logger.debug(f"{i}", end=" ")
    # logger.debug(f"Last {i=}")

    ds.shuffle()
    i = 0
    while True:
        a, b = next(ds)
        assert a.shape == torch.Size([8, 999])
        assert b.shape == torch.Size([8, 999])
        logger.debug(f"{i}", end=" ")
        i += 1
        if i >= ds.total_num_iter * 2:
            break
    logger.debug(f"Last {i=}")
