# Dataset.py

import os
import json
#import torch
import logging
import numpy as np
from tqdm import tqdm
#import soundfile as sf
from pathlib import Path
from collections import defaultdict
# from typing import Iterator, List, Dict, Optional
from torch.utils.data import Dataset, BatchSampler

logger = logging.getLogger("Dataset")

class BatchedBucketSampler(BatchSampler):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        """
        Given a dataset with samples containing an audio sample with fields:
        - pt_path: path to .pt file with audio embeddings   
        - ...

        Create batches of indices with elements (idx) that fall within the same pt_path (audio bucket).
        consecutive samples are within the same pt_path, so we can just group consecutive indices until batch_size is reached or pt_path changes.
        A batch must contain samples of the same pt_path (audio bucket) to boost disk io.
        Once batches are built, they are shuffled if shuffle=True (default) or kept in order if shuffle=False.
        """
        self.dataset = dataset
        self.bucket_size = dataset.info['bucket_size']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size2n = defaultdict(int) # dict to keep track of number of batches with n samples (for logging)

        assert dataset.info['bucket_size'] % batch_size == 0, f"Bucket size ({dataset.info['bucket_size']}) must be divisible by batch size ({batch_size}) to maximize batches containing samples from the same bucket (pt_path)."

        # Build indices of same bucket together (indices of samples with same pt_path)
        pt_path2idxs = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            pt_path = sample["pt_path"]
            pt_path2idxs[pt_path].append(idx)
        logger.info(f"Grouped {len(self.dataset)} samples into {len(pt_path2idxs)} buckets (pt_paths)")

        # shuffle pt_paths (buckets)
        pt_paths = list(pt_path2idxs.keys())
        if self.shuffle:
            self.shuffle(pt_paths)
            logger.info(f"Shuffled {len(pt_paths)} pt_paths (buckets) for batching")

        # build batches using indices of same pt_path (bucket)
        self.batches = []
        for pt_path in pt_paths:
            idxs = pt_path2idxs[pt_path]
            # split idxs into batches of batch_size
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i+self.batch_size]
                self.batches.append(batch)
                self.size2n[len(batch)] += 1
        logger.info(f"Created {len(self.batches)} batches with batch_size={self.batch_size}")
        logger.info(f"Batch size distribution: {dict(self.size2n)}")

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.dataset)


class Dataset(Dataset):
    """
    PyTorch Dataset for audio-to-LLM SFT training.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int,
        seed: int = 42,
    ):
        """
        Read audio embedding cache metadata.

        Args:
            file_path (str): Path to meta.json
            tokenizer: LLM tokenizer (used to convert text to target token ids)
            seq_len: int, maximum sequence length for target token ids (must match projector output length)
            seed: random seed for reproducibility (shuffling samples)

        Returns:
            dict with keys:
                - target_ids: tensor of shape [T] with token ids for ASR transcription (padded to seq_len)
                - pt_path: path to .pt file containing audio embeddings (tensor of shape [T', D])
                - duration: float, duration of original audio in seconds (for logging and analysis)
        """

        #random seed for reproducibility
        np.random.seed(seed)

        if not Path(file_path).is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if Path(file_path).name != "meta.json":
            raise ValueError(f"Only cached datasets with meta.json are supported: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.info = self.meta['info']
        samples = self.meta['samples']
        file_path_dir = Path(file_path).parent

        n_empty = 0
        n_maxlen = 0
        self.samples = []
        for idx, sample in enumerate(samples):
            target = sample.get("transcription",{}).get("text", "")
            if not target:
                logger.warning(f"Skipping empty transcription->text field for sample idx={idx}")
                n_empty += 1
                continue

            tokenizer.padding_side = "right"
            toks = tokenizer(
                target, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=seq_len,
                truncation=False, 
                add_special_tokens=False, 
                return_attention_mask=True,
            )

            target_ids = toks.input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)
            attention_mask = toks.attention_mask[0].bool() #tensor([1, 1, 1, 0, 0], dtype=torch.bool)

            if target_ids.size(0) == 0:
                logger.warning(f"Skipping empty target_ids for sample idx={idx}")
                n_empty += 1
                continue

            if target_ids.size(0) > seq_len:
                logger.warning(f"skipping too long target_ids for sample idx={idx}: {target_ids.size(0)} > {seq_len}")
                n_maxlen += 1
                continue

            sample["pt_path"] = file_path_dir / sample["pt_path"]
            sample["target_ids"] = target_ids
            sample["attention_mask"] = attention_mask

            self.samples.append(sample)

        logger.info(f"Skipped {n_empty} samples with empty target/target_ids")
        logger.info(f"Skipped {n_maxlen} samples with target_ids longer than seq_len={seq_len}")
        logger.info(f"Final dataset size: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoConfig
    import argparse

    parser = argparse.ArgumentParser(description="Test Dataset loading and batching.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--data_file", required=True, help="Dataset file")
    parser.add_argument("--seq_len", type=int, default=1500 // 15, help="Projector audio emnbedding sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    llm_path = config['llm']['path']
    audio_path = config['audio']['path']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    if tokenizer.pad_token is None:
        raise ValueError("""Tokenizer does not have a PAD token defined (use an LLM with defined pad_token).\nDuring pretraining, the model forces audio embeddings to match text embeddings. Due to length mismatch between audio frames and text tokens, PAD tokens are used to fill the remaining length of transcriptions. During inference, the LLM ignores PAD tokens without additional processing.""")

    audio_embedding_dim = AutoConfig.from_pretrained(audio_path).d_model
    llm_embedding_dim = AutoConfig.from_pretrained(llm_path).hidden_size

    # Create dataset from file
    ds = Dataset(file_path=args.data_file, tokenizer=tokenizer, seq_len=args.seq_len)

    # Create sampler from datset
    sampler = BatchedBucketSampler(ds, batch_size=args.batch_size, shuffle=True)

    # Iterate over sampler and print batch info
    for i, batch in enumerate(sampler):
        print(f"Batch {i}")
        for idx in batch:
            sample = ds[idx]
            target_ids = sample["target_ids"]
            masks = sample["attention_mask"]
            pt_path = sample["pt_path"]
            duration = sample["duration"]
            # for each ids in target_ids build the tuple (ids, token_str)
            target = [(ids.item(), tokenizer.decode(ids)) for ids in target_ids]
            print(f"\tidx={idx}\n\tduration={duration:.2f}\n\ttarget={target}\n\tmasks={masks.tolist()}\n\tpt_path={pt_path}\n")