import math
import time
import torch
import numpy as np
import dataclasses

from perf import PerfMonitor
from tqdm import tqdm
from model import make_model
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, List, Optional, Iterator, Tuple, Sequence
from dataset import make_dataset
from tokenizer import Tokenizer, make_tokenizer

logger.disable("llama")
logger.disable("llama_debug")

assert torch.cuda.is_available()
logger.info(f"{torch.cuda.device_count()=}")
logger.info(f"{torch.cuda.get_device_name(0)=}")
logger.info(f"{torch.cuda.get_device_properties(0).total_memory/1e9=} G")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

################################################################################
# training configurations
# 1 step = 1 batch
# N steps = 1 iter
# M iters = 1 epoch
#
# We use 1 iteration to accumulate gradient to mimic larger batch size with
# limited GPU memory usage
################################################################################


@dataclasses.dataclass
class TrainingConfig:
    name: str
    tokenizer_name: str
    model_name: str

    device: str = "cuda"

    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    eval_iterations: int = 256
    max_iterations: int = 256_000

    # learning rate
    is_learning_rate_decay: bool = True
    num_warmup_iterations: int = 1000
    max_decay_iterations: int = max_iterations
    max_learning_rate: float = 1e-4
    min_learning_rate: float = max_learning_rate * 0.1

    # optimizer
    weight_decay: float = 0.01
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    gradient_clip: float = 1.0


################################################################################
# learning rate decay scheduler
################################################################################


def get_learning_rate(iter_idx: int, config: TrainingConfig) -> float:
    if config.is_learning_rate_decay is False:
        return config.max_learning_rate

    # warmup phase: linear increase
    if iter_idx < config.num_warmup_iterations:
        return config.max_learning_rate * iter_idx / float(config.num_warmup_iterations)
    # beyond phase: to min
    if iter_idx > config.max_decay_iterations:
        return config.min_learning_rate
    # decay phase: cosine decay down to min
    decay_ratio = (iter_idx - config.num_warmup_iterations) / (
        config.max_decay_iterations - config.num_warmup_iterations
    )
    assert 0.0 <= decay_ratio <= 1.0
    decay_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + decay_coeff * (
        config.max_learning_rate - config.min_learning_rate
    )


def test_learning_rate():
    import matplotlib.pyplot as plt

    config = TrainingConfig(
        "test_learning_rate", "dummy_tokenizer_name", "dummy_model_name"
    )
    y = np.asarray([get_learning_rate(x, config) for x in range(0, 100000)])
    plt.plot(y)
    plt.savefig("/tmp/test_learning_rate.png", format="png")


################################################################################
# optimizer: AdamW
################################################################################


def make_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    # use weight decay for parameters >= 2D
    decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    nodecay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=config.max_learning_rate,
        betas=(config.adamw_beta1, config.adamw_beta2),
        fused=(config.device == "cuda"),
    )


def test_optimizer():
    config = TrainingConfig(
        "test_optimizer",
        "sentencepiece_tok512",
        "karpathy_tinystories260k",
        device="cpu",
    )
    make_optimizer(
        make_model(
            config.model_name, is_debug=True, is_compiled=False, device=config.device
        ),
        config,
    )
    config = TrainingConfig(
        "test_optimizer",
        "sentencepiece_tok512",
        "karpathy_tinystories260k",
        device="cuda",
    )
    make_optimizer(
        make_model(
            config.model_name, is_debug=False, is_compiled=True, device=config.device
        ),
        config,
    )


################################################################################
# loss function: cross-entropy
################################################################################


def get_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logger.trace(f"{logits.shape=} {targets.shape=}")
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1
    )


def test_loss_func():
    B = 4
    L = 1024
    V = 50000
    logits = torch.rand((B, L, V), dtype=torch.float32)
    targets = torch.randint(0, V, (B, L), dtype=torch.long)
    get_loss(logits, targets)


################################################################################


def train(config: TrainingConfig) -> None:
    logger.info("> Training config")
    logger.info(config)

    logger.info(f"> Create tokenizer: {config.tokenizer_name}")
    tokenizer = make_tokenizer(config.tokenizer_name)

    logger.info(f"> Create model: {config.model_name}")
    model = make_model(
        config.model_name,
        is_train=True,
        is_debug=False,
        is_compiled=True,
        device=config.device,
        dtype=torch.float32,
    )
    model.train()

    logger.info("> Create dataset")
    dataset = make_dataset(
        "train",
        tokenizer.name,
        max_seq_length=model.hparam.Lm,
        batch_size=config.batch_size,
        device=config.device,
    )

    dataset_eval = {}
    dataset_eval["train"] = make_dataset(
        "train",
        tokenizer.name,
        max_seq_length=model.hparam.Lm,
        batch_size=config.batch_size,
        device=config.device,
    )
    dataset_eval["validation"] = make_dataset(
        "validation",
        tokenizer.name,
        max_seq_length=model.hparam.Lm,
        batch_size=config.batch_size,
        device=config.device,
    )

    logger.info("> Create optimizer")
    optimizer = make_optimizer(model, config)

    logger.info("> Training starts")
    iter_idx = 0
    min_loss = 1e9

    pmon = PerfMonitor(logger, "training_eval_loop", print_iterations=1)

    start_time = time.time()
    for iter_idx in tqdm(range(config.max_iterations)):
        logger.trace(f"Training iteration {iter_idx}")

        # ----------------------------------------------------------------------
        # set learning rate
        # ----------------------------------------------------------------------
        lr = get_learning_rate(iter_idx, config)
        for g in optimizer.param_groups:
            g["lr"] = lr
        logger.trace(f"{lr=}")

        # ----------------------------------------------------------------------
        # forward / backward
        # ----------------------------------------------------------------------
        input_tokens, output_targets = next(dataset)
        for _ in range(config.gradient_accumulation_steps):
            # foward
            output_logits = model.forward(input_tokens)

            # calculate loss
            loss = get_loss(output_logits, output_targets)
            loss = loss / config.gradient_accumulation_steps

            # backward
            loss.backward()

            # prefetch next batch of data
            input_tokens, output_targets = next(dataset)

        # ----------------------------------------------------------------------
        # step advance optimizer
        # ----------------------------------------------------------------------
        if config.gradient_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip
            )
        optimizer.step()
        optimizer.zero_grad()

        # ----------------------------------------------------------------------
        # evaluation iteration: eval the loss and write checkpoints
        # ----------------------------------------------------------------------
        if iter_idx > 0 and iter_idx % config.eval_iterations == 0:
            logger.debug(">> Evaluation starts")

            loss = {}
            model.eval()
            with torch.inference_mode():
                for split in dataset_eval.keys():
                    losses = torch.zeros(config.eval_iterations)
                    for idx, (input_tokens, output_targets) in enumerate(
                        dataset_eval[split]
                    ):
                        if idx >= config.eval_iterations:
                            break
                        output_logits = model.forward(input_tokens)
                        losses[idx] = get_loss(output_logits, output_targets)
                    loss[split] = losses.mean()
                for split in dataset_eval.keys():
                    logger.debug(f"dataset={split} loss={loss[split]}")
            model.train()

            # save if loss is smaller than minimum
            if loss["validation"] < min_loss:
                logger.debug(f"min_loss: {min_loss:.4f} -> {loss['validation']:.4f}")
                min_loss = loss["validation"]

                if iter_idx > 0:
                    logger.debug("Saving checkpoint")
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_idx": iter_idx,
                        "min_loss": min_loss,
                        "config": config,
                    }
                    torch.save(checkpoint, f"checkpoint.{config.name}.pt")

            elapsed_time_hour = (time.time() - start_time) / 3600.0
            logger.info(
                f"<SUMMARY> {iter_idx} | loss {min_loss:.4f} | lr {lr:e} | elapsed {elapsed_time_hour:.2f}hrs"
            )
            pmon.loop()


def test_train():
    config = TrainingConfig(
        name="test_cpu",
        tokenizer_name="sentencepiece_tok512",
        model_name="karpathy_tinystories260k",
        device="cpu",
        batch_size=2,
        gradient_accumulation_steps=4,
        eval_iterations=10,
        max_iterations=100,
    )
    logger.add(f"train.{config.name}.log")
    train(config)

    config = TrainingConfig(
        name="test_cuda",
        tokenizer_name="sentencepiece_tok512",
        model_name="karpathy_tinystories260k",
        device="cuda",
        batch_size=2,
        gradient_accumulation_steps=4,
        eval_iterations=10,
        max_iterations=100,
    )
    logger.add(f"train.{config.name}.log")
    train(config)


def main(name: str):
    if name == "baseline_debug":
        config = TrainingConfig(
            name=name,
            tokenizer_name="sentencepiece_tok512",
            model_name="karpathy_tinystories260k",
            batch_size=8,
            gradient_accumulation_steps=4,
            eval_iterations=100,
            max_iterations=10_000,
        )
    if name == "baseline_15m":
        config = TrainingConfig(
            name=name,
            tokenizer_name="sentencepiece_tok32k",
            model_name="karpathy_tinystories15m",
            batch_size=32,
            gradient_accumulation_steps=4,
            eval_iterations=100,
            max_iterations=100_000,
        )

    logger.add(f"train.{name}.log")
    train(config)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
