AVAIL_TOKENIZERS = (
    "cl100k_base",
    "r50k_base",
    "sentencepiece_tok32k",
    "sentencepiece_tok512",
)

import tiktoken

from perf import PerfMonitor
from loguru import logger
from typing import List, Optional, Tuple
from sentencepiece import SentencePieceProcessor
from huggingface_hub import hf_hub_download


class Tokenizer:
    def __init__(self, name):
        self.name = name
        if name.startswith("sentencepiece_"):
            fpath = hf_hub_download(
                repo_id="jimwang99/TinyStoriesV2-Tokenized",
                repo_type="dataset",
                filename=f"{name}.pt",
            )
            self.model = SentencePieceProcessor(model_file=fpath)
            self.bos = self.model.bos_id()
            self.eos = self.model.eos_id()
            self.vocab_size = self.model.vocab_size()
        else:
            self.model = tiktoken.get_encoding(name)
            self.bos = None
            self.eos = self.model._special_tokens["<|endoftext|>"]
            self.vocab_size = self.model.n_vocab

    def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:
        tokens = self.model.encode(text)
        if bos and self.bos:
            tokens.insert(0, self.bos)
        if eos:
            tokens.append(self.eos)
        return tokens

    def encode_batch(
        self,
        texts: List[str],
        max_seq_length: Optional[int] = None,
        pad: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[int]]:
        batch = [self.encode(text) for text in texts]

        # truncate
        if max_seq_length is not None:
            batch = [x[:max_seq_length] for x in batch]

        # lengths
        lens = [len(x) for x in batch]

        # padding
        if pad is not None:
            max_len = max([len(x) for x in batch])
            batch = [x + [pad] * (max_len - len(x)) for x in batch]

        return batch

    def decode(self, tokens: List[int]) -> str:
        return self.model.decode(tokens)


def make_tokenizer(name: str) -> Tokenizer:
    assert name in AVAIL_TOKENIZERS, f"{name=} not in {AVAIL_TOKENIZERS=}"
    logger.debug(f"{name=}")
    pm = PerfMonitor(logger, "make_tokenizer")
    t = Tokenizer(name)
    pm.once()
    return t


def test_tokenizer():
    n = "r50k_base"
    t = make_tokenizer(n)
    assert t.vocab_size == 50257
    assert t.encode("Hello, world!") == [15496, 11, 995, 0]
    assert t.encode("Hello, world!", eos=True) == [15496, 11, 995, 0, 50256]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, pad=-1) == [
        [15496, 11, 995, 0, 50256],
        [31373, 50256, -1, -1, -1],
    ]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, max_seq_length=2) == [
        [15496, 11],
        [31373, 50256],
    ]

    n = "cl100k_base"
    t = make_tokenizer(n)
    assert t.vocab_size == 100277
    assert t.encode("Hello, world!") == [9906, 11, 1917, 0]
    assert t.encode("Hello, world!", eos=True) == [9906, 11, 1917, 0, 100257]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, pad=-1) == [
        [9906, 11, 1917, 0, 100257],
        [15339, 100257, -1, -1, -1],
    ]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, max_seq_length=2) == [
        [9906, 11],
        [15339, 100257],
    ]

    n = "sentencepiece_tok32k"
    t = make_tokenizer(n)
    assert t.vocab_size == 32000
    assert t.encode("Hello, world!") == [15043, 29892, 3186, 29991]
    assert t.encode("Hello, world!", eos=True) == [15043, 29892, 3186, 29991, 2]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, pad=-1) == [
        [15043, 29892, 3186, 29991, 2],
        [22172, 2, -1, -1, -1],
    ]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, max_seq_length=2) == [
        [15043, 29892],
        [22172, 2],
    ]

    n = "sentencepiece_tok512"
    t = make_tokenizer(n)
    assert t.vocab_size == 512
    assert t.encode("Hello, world!") == [346, 306, 414, 432, 263, 304, 341, 443]
    assert t.encode("Hello, world!", eos=True) == [
        346,
        306,
        414,
        432,
        263,
        304,
        341,
        443,
        2,
    ]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, pad=-1) == [
        [346, 306, 414, 432, 263, 304, 341, 443, 2],
        [281, 306, 414, 2, -1, -1, -1, -1, -1],
    ]
    assert t.encode_batch(["Hello, world!", "hello"], eos=True, max_seq_length=2) == [
        [346, 306],
        [281, 306],
    ]
