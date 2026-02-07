import argparse
from tqdm import tqdm
import logging
import torch
import json
import os
import sys
import time
from transformers import AutoTokenizer
from collections import defaultdict
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import read_samples_from_jsonl
from scripts.text_normalize import normalize_text
from Embedder import Embedder

logger = logging.getLogger("build_audio_cache")


def process_batch(audio_embedder, samples, batch_indices, device, dtype):
    """
    Embed audio for a batch of indices.
    Returns embeddings on CPU.
    """
    audio_paths = [samples[idx]["audio_file"] for idx in batch_indices]

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device == "cuda")):
            audio_res = audio_embedder(audio_paths)  # Whisper: ignore mask

        audio_embs = audio_res[0] if isinstance(audio_res, tuple) else audio_res

    return audio_embs.cpu().contiguous()


def split_batch(batch_indices, audio_embs):
    """
    Convert batch embeddings into a list of (index, embedding) tuples.
    """
    return [(idx, audio_embs[i]) for i, idx in enumerate(batch_indices)]


def save_bucket(samples, bucket, cache_dir, bucket_id):
    """
    Save a bucket of embeddings to disk and update sample metadata.
    """
    pt_path = os.path.join(cache_dir, f"bucket_bs{len(bucket)}_{bucket_id:06d}.pt")
    tmp_path = pt_path + ".tmp"

    indices = [idx for idx, _ in bucket]
    embs = torch.stack([emb for _, emb in bucket])  # (B, T, D)

    # do not save if file already exists (can happen if multiple processes are running this script with same cache_dir)
    if not os.path.exists(pt_path):
        torch.save({"audio_embs": embs}, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, pt_path)

    # update sample metadata
    for i, idx in enumerate(indices):
        samples[idx]["pt_path"] = os.path.basename(pt_path)
        samples[idx]["offset"] = i
        #not used: samples[idx]["n_audio_embs"] = embs.shape[1]  # T


def save_sorted_samples(audio_embedder, samples, cache_dir, batch_size, bucket_size, device, torch_dtype):
    # embed (batch_size) samples and save embeddings in files containing bucket_size samples
    batch_indices = []
    bucket = []
    bucket_id = 0
    t_embedding = 0.0
    t_saving = 0.0

    os.makedirs(cache_dir, exist_ok=True)

    for idx in tqdm(range(len(samples)), total=len(samples), desc="Embedding audio", unit=" sample"):

        batch_indices.append(idx)

        # process batch
        if len(batch_indices) == batch_size:
            tic = time.time()
            audio_embs_cpu = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)
            t_embedding += time.time() - tic
            bucket.extend(split_batch(batch_indices, audio_embs_cpu))
            batch_indices = []

        # process bucket
        while len(bucket) >= bucket_size:
            tic = time.time()
            save_bucket(samples, bucket[:bucket_size], cache_dir, bucket_id)
            t_saving += time.time() - tic
            bucket = bucket[bucket_size:]
            bucket_id += 1

    # process remaining batch
    if batch_indices:
        tic = time.time()
        audio_embs_cpu = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)
        t_embedding += time.time() - tic
        bucket.extend(split_batch(batch_indices, audio_embs_cpu))

    # process remaining bucket
    while bucket:
        tic = time.time()
        save_bucket(samples, bucket[:bucket_size], cache_dir, bucket_id)
        t_saving += time.time() - tic
        bucket = bucket[bucket_size:]
        bucket_id += 1

    logger.info(f"Saved {len(samples)} embeddings in {bucket_id} buckets dir={cache_dir}")
    logger.info(f"Embedding time = {t_embedding:.2f}s, Saving time = {t_saving:.2f}s")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from JSON (bucketed)")
    parser.add_argument("--json_path", type=str, required=True, help="JSON file with audio metadata")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium")
    parser.add_argument("--tokenizer_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype for embeddings")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of samples to fed to embedder")
    parser.add_argument("--bucket_size", type=int, default=256, help="Number of samples per saved bucket")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(f"{args.json_path}_CACHE_ASR.log", mode='a', encoding='utf-8')
    ])

    #################################################################################
    ### Compute tokenized lengths and sort samples by length (shortest → longest) ###
    #################################################################################

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    # Read JSON samples
    logger.info("-"* 40 + f" Reading samples from {args.json_path} ")
    data = read_samples_from_jsonl(args.json_path)
    logger.info(f"Read {len(data)} samples from {args.json_path}")

    splits = set()
    slangs = set()
    combination2samples = defaultdict(list) # dict of (split, slang) → list of samples
    unique_audio_files = set() 

    logger.info("-"* 40 + f" Filtering samples " )
    samples = []
    for s in tqdm(data, total=len(data), desc="Tokenizing text", unit=" sample"):
        audio_file = s.get("audio_file", "")
        if not isinstance(audio_file, str) or not audio_file.strip():
            continue

        if audio_file in unique_audio_files:
            continue

        text = s.get("transcription", {}).get("text", "")
        text = normalize_text(text) if text else ""
        if not isinstance(text, str) or not text.strip():
            continue

        split = s.get("split", "None")
        slang = s.get("transcription", {}).get("lang", "None")


        ids = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        s = {"audio_file": audio_file, "text": text, "split": split, "slang": slang, "len": len(ids)}
        combination2samples[(split, slang)].append(s)
        splits.add(split)
        slangs.add(slang)
        unique_audio_files.add(audio_file)


    logger.info(f"Found {len(unique_audio_files)} unique audio files after filtering")
    logger.info(f"Splits: {splits}")
    logger.info(f"slangs: {slangs}")
    ### log combinations and their counts, sorted by value (count) descending and then by split and slang lexicographically ascending
    combinations = list(combination2samples.keys())
    combinations.sort(key=lambda x: (len(combination2samples[x]), x[0], x[1]), reverse=True)
    logger.info("Combinations (split, slang) and their counts:")
    for split, slang in combinations:
        count = len(combination2samples[(split, slang)])
        logger.info(f"  ({split}, {slang}): {count} samples")    

    if len (samples) == 0:
        logger.info("No samples to process after filtering.")
        sys.exit(0)

    #################################################################################
    ### Save audio embeddings in bucketed .pt files #################################
    #################################################################################

    torch_dtype = getattr(torch, args.dtype)

    # Initialize embedder
    audio_embedder = Embedder(config={'path': args.embedder_path})
    audio_embedder.to(args.device, dtype=torch_dtype)
    audio_embedder.eval()

    logger.info("-"* 40 + f" Embedding/saving samples ")
    idx = 0
    for split, slang in combinations:
        idx += 1
        combination_samples = combination2samples[(split, slang)]
        combination_samples.sort(key=lambda x: (x["len"], x["audio_file"]))
        logger.info(f"Combination {idx}/{len(combinations)} ({split}: {slang}): {len(combination_samples)} samples")

        cache_dir = os.path.join(args.json_path + "_CACHE_ASR", f"{split}/{slang}")

        if os.path.exists(os.path.join(cache_dir, "meta.json")):
            logger.info(f"Cache directory {cache_dir} already contains meta.json, skipping embedding/saving")
            continue

        samples = save_sorted_samples(
            audio_embedder, 
            combination_samples, 
            cache_dir,
            args.batch_size, 
            args.bucket_size, 
            args.device, 
            torch_dtype
        )

        # Save meta.json
        meta_path = os.path.join(cache_dir, "meta.json")
        info = {
            "json_path": args.json_path,
            "cache_dir": cache_dir,
            "embedder_path": args.embedder_path,
            "tokenizer_path": args.tokenizer_path,
            "bucket_size": args.bucket_size,
            "dtype": args.dtype,
        }

        # meta.json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        # samples.jsonl
        samples_path = os.path.join(cache_dir, "samples.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(f"Saved {meta_path} {samples_path}")
