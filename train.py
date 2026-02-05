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
    parser.add_argument("--train", required=True, help="Training dataset file")
    parser.add_argument("--eval", default=None, help="Evaluation dataset file")
    # opt pars
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps (must be >0 for scheduler)")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of training epochs (0 for no limit)")
    # train pars
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samplse in a batch")
    parser.add_argument("--accum_steps", type=int, default=4, help="Accumulate this many batchs before optimizing")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluation (and saving checkpoint) after this many optimization steps")
    parser.add_argument("--log_every", type=int, default=10, help="Logging after this many optimization steps")
    parser.add_argument("--save_best_n", type=int, default=3, help="Save top N checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume previous training")
    # output
    parser.add_argument("--output_dir", type=str, default="./stage1", help="Output directory of training")
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

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

    logger.info("=" * 80)
    logger.info(f"Starting new run @ {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"CONFIG FILE: {args.config}\n" + json.dumps(config, indent=2) + "\n")
    logger.info("=" * 80)

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    json_logger.log(
        type="run",
        config=config,
        train=args.train,
        eval=args.eval,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        resume=args.resume,
        output_dir=args.output_dir,
    )

    model = AudioLLM(
        config=config,
        device=device,
        dtype=dtype 
    )

    # -----------------------------
    # Datasets 
    # -----------------------------

    train_dataset = Dataset(
        file_path=args.train,
        tokenizer=model.tokenizer,
        seq_len=model.projector.seq_len_out,
    )

    eval_dataset = Dataset(
        file_path=args.eval,
        tokenizer=model.tokenizer,
        seq_len=model.projector.seq_len_out,
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

