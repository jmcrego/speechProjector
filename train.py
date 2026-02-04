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

from Projector import Projector
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
    parser.add_argument("--lr_proj", type=float, default=5e-4, help="Learning rate for projector layers")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps (must be >0 for scheduler)")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of training epochs (0 for no limit)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Scheduler warmup steps (use ~5%)")
    # train pars
    parser.add_argument("--batch_size", type=int, default=8, help="Number of sampels in a batch")
    parser.add_argument("--accum_steps", type=int, default=4, help="Accumulate this many batchs before optimizing")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluation (and saving checkpoint) after this many optimization steps")
    parser.add_argument("--log_every", type=int, default=10, help="Logging after this many optimization steps")
    parser.add_argument("--save_best_n", type=int, default=3, help="Save top N checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume previous training")
    # output
    parser.add_argument("--output_dir", type=str, default="./sft_output", help="Output directory of training")
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
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        resume=args.resume,
        output_dir=args.output_dir,
    )

    llm_path = config['llm']['path']
    audio_path = config['audio']['path']
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    audio_embedding_dim = AutoConfig.from_pretrained(audio_path).d_model
    llm_embedding_dim = AutoConfig.from_pretrained(llm_path).d_model

    logger.info(f"Loaded Tokenizer from {llm_path}")
    logger.info(f"bos_token = {tokenizer.bos_token} {tokenizer.bos_token_id}")
    logger.info(f"eos_token = {tokenizer.eos_token} {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        raise ValueError("""Tokenizer does not have a PAD token defined (use an LLM with defined pad_token).\nDuring pretraining, the model forces audio embeddings to match text embeddings. Due to length mismatch between audio frames and text tokens, PAD tokens are used to fill the remaining length of transcriptions. During inference, the LLM ignores PAD tokens without additional processing.""")
    logger.info(f"pad_token = {tokenizer.pad_token} {tokenizer.pad_token_id}")
    logger.info(f"Audio embedding dimension: {audio_embedding_dim}")
    logger.info(f"LLM embedding dimension: {llm_embedding_dim}")

    projector = Projector(config=config['projector'], audio_embedding_dim=audio_embedding_dim, llm_embedding_dim=llm_embedding_dim).to(device, dtype)

    # -----------------------------
    # Datasets 
    # -----------------------------

    train_dataset = Dataset(
        file_path=args.train,
        tokenizer=tokenizer,
        seq_len=projector.seq_len_out,
    )

    eval_dataset = Dataset(
        file_path=args.eval,
        tokenizer=tokenizer,
        seq_len=projector.seq_len_out,
    ) if args.eval is not None else None

    # -----------------------------
    # Create Trainer
    # -----------------------------

    trainer = Trainer(
        config=config,
        model=projector,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        lr_proj=args.lr_proj,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
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

