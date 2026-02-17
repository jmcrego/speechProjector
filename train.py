# train.py

import os
import json
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

from scripts.utils import get_device_dtype
from scripts.utils import JSONMetricsLogger

from AudioLLM  import AudioLLM
from Trainer import Trainer
from Dataset import Dataset

logger = logging.getLogger("train")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a speech ASR/STT decoder (audio-embedder ➔ Projector ➔ LLM).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    # dataset paths
    parser.add_argument("--train", nargs="+", required=True, help="Training samples.jsonl files")
    parser.add_argument("--eval", nargs="+", default=None, help="Evaluation samples.jsonl files")
    # opt pars
    parser.add_argument("--lr_proj", type=float, default=1e-4, help="Learning rate for projector (the only part we train)")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Number of warmup steps for learning rate scheduler (should be ~10% of total steps)")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum number of training steps (0 for no limit)")
    parser.add_argument("--max_epochs", type=int, default=0, help="Maximum number of training epochs (0 for no limit)")
    parser.add_argument("--alpha", type=float, default=0.5, help="MSE loss = alpha * MSE_txt + (1 - alpha) * MSE_pad")
    parser.add_argument("--weight_mse", type=float, default=0., help="Weight of MSE loss (0 to disable it)")
    parser.add_argument("--weight_cos", type=float, default=0., help="Weight of cosine loss (0 to disable it)")
    parser.add_argument("--weight_scale", type=float, default=0., help="Weight of scale loss (0 to disable it)")
    parser.add_argument("--weight_ce", type=float, default=0., help="Weight of cross-entropy loss (0 to disable it)")
    parser.add_argument("--temp_ce", type=float, default=1.0, help="Temperature for cross-entropy loss")
    # train pars
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples in a batch")
    parser.add_argument("--accum_steps", type=int, default=1, help="Accumulate this many batchs before optimizing")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluation (and saving checkpoint) after this many optimization steps")
    parser.add_argument("--log_every", type=int, default=100, help="Logging after this many optimization steps")
    parser.add_argument("--save_best_n", type=int, default=5, help="Save top N checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume previous training")
    # output
    parser.add_argument("--output_dir", type=str, default="./stage1", help="Output directory of training")
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    assert args.alpha >= 0 and args.alpha <= 1, "Alpha must be in [0, 1]"
    assert args.weight_mse >= 0, "Weight MSE must be >= 0"
    assert args.weight_cos >= 0, "Weight cosine must be >= 0"
    assert args.weight_scale >= 0, "Weight scale must be >= 0"
    assert args.weight_ce >= 0, "Weight CE must be >= 0"
    assert args.weight_mse > 0 or args.weight_cos > 0 or args.weight_scale > 0 or args.weight_ce > 0, "At least one loss must have weight > 0"

    if args.max_steps == 0 and args.max_epochs == 0:
        raise ValueError("At least one of max_steps or max_epochs must be > 0 to define a stopping criterion for training")

    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json_logger = JSONMetricsLogger(Path(args.output_dir) / "metrics.jsonl")

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(Path(args.output_dir) / "train.log" , mode='a', encoding='utf-8'),  # append mode
            logging.StreamHandler()  # and log to console
        ]
    )
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    logger.info(f"Starting new run @ {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    json_logger.log(
        type="run",
        args=vars(args),
        config=config,
        device=str(device),
        dtype=str(dtype),
    )

    model = AudioLLM(
        config=config,
        alpha=args.alpha,
        weight_mse=args.weight_mse,
        weight_cos=args.weight_cos,
        weight_scale=args.weight_scale,
        weight_ce=args.weight_ce,
        temp_ce=args.temp_ce,
        device=device,
        dtype=dtype 
    )

    # -----------------------------
    # Datasets 
    # -----------------------------

    train_dataset = Dataset(
        jsonl_paths=args.train,
        tokenizer=model.tokenizer,
        seq_len=model.projector.seq_len_out,
    )

    eval_dataset = Dataset(
        jsonl_paths=args.eval,
        tokenizer=model.tokenizer,
        seq_len=model.projector.seq_len_out,
        # n_samples=200, #max number of eval samples (for quick eval during training, set to 0 for all)
    ) if args.eval is not None else None


    # -----------------------------
    # Create Trainer
    # -----------------------------

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        lr_proj=args.lr_proj,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_best_n=args.save_best_n,
        eval_every=args.eval_every,
        log_every=args.log_every,
        accum_steps=args.accum_steps,
        output_dir=args.output_dir,
        json_logger=json_logger,
        resume=args.resume,
    )

    # -----------------------------
    # Start training
    # -----------------------------

    trainer.train()

