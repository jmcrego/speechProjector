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


def save_sorted_samples(audio_embedder, samples, embedder_path, batch_size, bucket_size, json_path, cache_dir, tokenizer_path, device, torch_dtype):
    # embed (batch_size) samples and save embeddings in files containing bucket_size samples
    batch_indices = []
    bucket = []
    bucket_id = 0
    t_embedding = 0.0
    t_saving = 0.0

    if os.path.exists(os.path.join(cache_dir, "meta.json")):
        logger.info(f"Cache directory {cache_dir} already contains meta.json, skipping embedding and saving")
        return

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

    # Save meta.json
    meta_path = os.path.join(cache_dir, "meta.json")
    info = {
        "json_path": json_path,
        "cache_dir": cache_dir,
        "embedder_path": embedder_path,
        "tokenizer_path": tokenizer_path,
        "dtype": str(torch_dtype),
        "bucket_size": bucket_size,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"info": info, "samples": samples}, f, ensure_ascii=False)
    logger.info(f"Saved {meta_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from JSON (bucketed)")
    parser.add_argument("--json_path", type=str, required=True, help="JSON file with audio metadata")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium")
    parser.add_argument("--tokenizer_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    #correct the next line
    parser.add_argument("--split", type=str, default=None, help="Split to use (use all if not given)")
    parser.add_argument("--slang", type=str, default=None, help="Transcription language to use (use all if not given)")
    parser.add_argument("--tlang", type=str, default=None, help="Translation language to use (use all if not given)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype for embeddings")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples to fed to embedder")
    parser.add_argument("--bucket_size", type=int, default=128, help="Number of samples per saved bucket")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    #################################################################################
    ### Compute tokenized lengths and sort samples by length (shortest → longest) ###
    #################################################################################

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    # Read JSON samples
    samples = read_samples_from_jsonl(args.json_path, split=args.split, slang=args.slang, tlang=args.tlang)
    logger.info(f"Read {len(samples)} samples from {args.json_path} with split={args.split}, slang={args.slang}, tlang={args.tlang}")

    # Compute tokenized lengths
    samples_triplets = []

    splits = set()
    slangs = set()
    tlangs = set()
    combinations = set() # set of (split, slang, tlang) combinations found in the data after filtering
    unique_audio_files = set() # for logging purposes only

    for s in tqdm(samples, total=len(samples), desc="Tokenizing text", unit=" sample"):
        audio_file = s.get("audio_file", "")

        if not isinstance(audio_file, str) or not audio_file.strip():
            continue

        split = s.get("split", "None")
        if args.split:
            if split != args.split:
                continue

        slang = s.get("transcription", {}).get("lang", "None")
        if args.slang:
            if slang != args.slang:
                continue

        tlang = s.get("translation", {}).get("lang", "None")
        if args.tlang:
            if tlang != args.tlang:
                continue

        text = s.get("transcription", {}).get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        translation_text = s.get("translation", {}).get("text", "")

        splits.add(split)
        slangs.add(slang)
        tlangs.add(tlang)
        combinations.add((split, slang, tlang))

        ids = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        samples_triplets.append({"audio_file": audio_file, "text": text, "translation": translation_text, "ids": ids, "slang": slang, "tlang": tlang, "split": split, "len": len(ids)})
        unique_audio_files.add(audio_file)

    logger.info(f"Found {len(unique_audio_files)} unique audio files after filtering")
    logger.info(f"Splits: {splits}")
    logger.info(f"slangs: {slangs}")
    logger.info(f"tlangs: {tlangs}")
    logger.info(f"Combinations: {combinations}")

    if len (samples_triplets) == 0:
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

    for split, slang, tlang in combinations:
        combinations_samples = [
            s for s in samples_triplets
            if s['split'] == split and s['slang'] == slang and s['tlang'] == tlang
        ]
        combinations_samples.sort(key=lambda x: (x["len"], x["audio_file"]))
        logger.info(f"Combination (split={split}, {slang}-{tlang}): {len(combinations_samples)} samples")

        save_sorted_samples(
            audio_embedder, 
            combinations_samples, 
            args.embedder_path, 
            args.batch_size, 
            args.bucket_size, 
            args.json_path, 
            os.path.join(args.json_path + "_cache", f"{split}/{slang}/{tlang}"), 
            args.tokenizer_path, 
            args.device, 
            torch_dtype
        )

