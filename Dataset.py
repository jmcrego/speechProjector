# Dataset.py

import sys
import json
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger("Dataset")


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        target_ids = torch.stack([x["target_ids"].squeeze(0) for x in batch], dim=0)  # (B, T)
        prompt_list = [x["prompt_ids"].squeeze(0) for x in batch]
        prompt_ids = pad_sequence(prompt_list, batch_first=True, padding_value=self.pad_token_id)
        pt_paths = [x["pt_path"] for x in batch]
        offsets = [x["offset"] for x in batch]

        return {
            "pt_paths": pt_paths,
            "offsets": offsets,
            "target_ids": target_ids,
            "prompt_ids": prompt_ids,
        }
        
# def collate_fn(batch):
#     target_ids = torch.stack([x["target_ids"] for x in batch], dim=0)  # (B, T)
#     prompt_ids = torch.stack([x["prompt_ids"] for x in batch], dim=0)  # (B, T)
#     pt_paths = [x["pt_path"] for x in batch] # List[str] (B,)
#     offsets = [x["offset"] for x in batch] # List[int] (B,)

#     return {
#         "pt_paths": pt_paths,         # List[str] (B,)
#         "offsets": offsets,           # (B,)
#         "target_ids": target_ids,     # (B, T)
#         "prompt_ids": prompt_ids,     # (B, T)
#     }


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
        self.bucket_size = dataset.bucket_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size2n = defaultdict(int) # dict to keep track of number of batches with n samples (for logging)

        if self.bucket_size is not None and self.bucket_size % self.batch_size != 0:
            logger.warning(f"Bucket size ({self.bucket_size}) is not divisible by batch size ({batch_size}). This may lead to suboptimal batching. Consider setting batch_size to a multiple of bucket_size for optimal performance.")
            sys.exit(1)

        # Build indices of same bucket together (indices of samples with same pt_path)
        pt_path2idxs = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            pt_path = sample["pt_path"]
            pt_path2idxs[pt_path].append(idx)
        logger.info(f"Grouped {len(self.dataset)} samples into {len(pt_path2idxs)} buckets (pt_paths)")

        # shuffle pt_paths (buckets)
        pt_paths = list(pt_path2idxs.keys())
        if self.shuffle:
            random.shuffle(pt_paths)
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
        jsonl_paths: [str],  #string or list of strings with expand characters like "/my/path/*/??/samples.jsonl"
        tokenizer,
        seq_len: int,
        audio_token: str = "<extra_id_0>",
        n_samples: int = 0, # if >0 select randomly n samples from the dataset (deterministic with seed)
        seed: int = 42,
    ):
        """
        Read audio embedding cache metadata.

        Args:
            jsonl_paths ([str]): list of strings with expand characters like "/my/path/*/??/samples.jsonl"
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

        tokenizer.padding_side = "right"

        self.bucket_size = None

        ### read all files matching "/my/path/*/??/samples.jsonl" and concatenate samples into a single list
        samples = []
        for f_jsonl in jsonl_paths:

            # read meta.json file to get bucket_size (for logging)
            meta_path = Path(f_jsonl).parent / "meta.json"
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    bucket_size = info.get("bucket_size", None)
                    if bucket_size is None:
                        logger.error(f"No bucket_size found in {meta_path}")
                        exit(1)

                    if self.bucket_size is None:
                        self.bucket_size = bucket_size
                        logger.info(f"bucket_size set to {self.bucket_size}")

                    if bucket_size != self.bucket_size:
                        logger.error(f"Bucket size mismatch for {meta_path}: {bucket_size} vs {self.bucket_size}")
                        sys.exit(1)

            if not Path(f_jsonl).is_file():
                logger.warning(f"File not found: {f_jsonl}")
                continue

            if Path(f_jsonl).name != "samples.jsonl":
                logger.warning(f"File {f_jsonl} does not have expected name 'samples.jsonl', skipping")
                continue

            curr_samples = []
            with open(f_jsonl, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Reading {f_jsonl}", unit=" lines"):
                    entry = json.loads(line)
                    lang = entry.get("slang", "English")
                    sample = {
                        "pt_path": entry["pt_path"] if Path(entry["pt_path"]).is_absolute() else Path(f_jsonl).parent / entry["pt_path"],
                        "offset": entry["offset"],
                        "target": entry["text"],
                        "prompt": f"Input:\n{audio_token}\nRepeat the above {lang} Input text:\n"
                    }
                    curr_samples.append(sample)
                if n_samples > 0 and n_samples < len(curr_samples):
                    curr_samples = random.sample(curr_samples, n_samples)
            samples.extend(curr_samples)


        n_empty = 0
        n_maxlen = 0
        self.samples = []
        for idx, sample in enumerate(tqdm(samples, desc="Processing samples", unit=" samples")):
            if not sample['target']:
                logger.warning(f"Skipping empty transcription->text field for sample idx={idx}")
                n_empty += 1
                continue

            ### tokenize with padding to max_length=seq_len (projector output length) and truncation=False 
            ### discard samples longer than seq_len
            
            # target_ids = tokenizer(
            #     sample['target'], 
            #     return_tensors="pt", 
            #     padding="max_length", 
            #     max_length=seq_len,
            #     truncation=False, 
            #     add_special_tokens=False, 
            #     return_attention_mask=False,
            # )
            target_ids = tokenizer(
                sample['target'],
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )

            target_ids = target_ids.input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)
            len = target_ids.size(0)

            if len >= seq_len:
                logger.warning(f"Skipping sample idx={idx} with target_ids length {len} longer or equal than seq_len={seq_len}")
                n_maxlen += 1
                continue

            if len == 0:
                logger.warning(f"Skipping sample idx={idx} with empty target_ids after tokenization")
                n_empty += 1
                continue

            #pad to max_length=seq_len (projector output length) with tokenizer.pad_token_id
            padding_len = seq_len - len
            if padding_len > 0:
                padding = torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
                target_ids = torch.cat([target_ids, padding], dim=0)

            prompt_ids = tokenizer(
                sample['prompt'], 
                return_tensors="pt", 
                truncation=False, 
                add_special_tokens=False, 
                return_attention_mask=False,
            )
            prompt_ids = prompt_ids.input_ids[0].long() #tensor([ p₁, p₂, p₃, … ], dtype=torch.long)

            sample["target_ids"] = target_ids
            sample["prompt_ids"] = prompt_ids

            self.samples.append(sample)

        logger.info(f"Skipped {n_empty} samples with empty target/target_ids")
        logger.info(f"Skipped {n_maxlen} samples with target_ids longer than seq_len={seq_len}")
        logger.info(f"Final dataset size: {len(self.samples)} samples")

        if n_samples > 0 and n_samples < len(self.samples):
            self.samples = random.sample(self.samples, n_samples)
            logger.info(f"Randomly selected {n_samples} samples from the dataset (seed={seed})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse

    parser = argparse.ArgumentParser(description="Test Dataset loading and batching.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--jsonl_paths", nargs="+", required=True, help="Dataset files (use expand characters like * if needed)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sampling")
    parser.add_argument("--seq_len", type=int, default=1500 // 15, help="Projector audio emnbedding sequence length")
    parser.add_argument("--n_samples", type=int, default=0, help="If >0, randomly select this many samples from the dataset (deterministic with seed) [breaks bucketing]")
    args = parser.parse_args()


    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    audio_path = config['audio']['path']
    llm_path = config['llm']['path']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    if tokenizer.pad_token is None:
        raise ValueError("""Tokenizer does not have a PAD token defined (use an LLM with defined pad_token).\nDuring pretraining, the model forces audio embeddings to match text embeddings. Due to length mismatch between audio frames and text tokens, PAD tokens are used to fill the remaining length of transcriptions. During inference, the LLM ignores PAD tokens without additional processing.""")

    # Create dataset from file
    ds = Dataset(jsonl_paths=args.jsonl_paths, tokenizer=tokenizer, seq_len=args.seq_len, n_samples=args.n_samples)

    # Create sampler from datset
    sampler = BatchedBucketSampler(ds, batch_size=args.batch_size, shuffle=True)

    # Iterate over sampler and print batch info
    for i, batch in enumerate(sampler):
        print(f"Batch {i}")
        for idx in batch:
            target_ids = ds[idx]["target_ids"]
            pt_path = ds[idx]["pt_path"]
            duration = ds[idx].get("duration", 0)
            # for each ids in target_ids build the tuple (ids, token_str)
            target = [(ids.item(), tokenizer.decode(ids)) for ids in target_ids]
            print(f"\tidx={idx}\n\tduration={duration:.2f}\n\ttarget={target}\n\tpt_path={pt_path}\n")