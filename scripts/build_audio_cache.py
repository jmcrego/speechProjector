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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset import read_samples_from_jsonl
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
    pt_path = os.path.join(cache_dir, f"bucket_{bucket_id:06d}.pt")
    tmp_path = pt_path + ".tmp"

    indices = [idx for idx, _ in bucket]
    embs = torch.stack([emb for _, emb in bucket])  # (B, T, D)

    torch.save({"audio_embs": embs}, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, pt_path)

    # update sample metadata
    for i, idx in enumerate(indices):
        samples[idx]["pt_path"] = os.path.basename(pt_path)
        samples[idx]["offset"] = i
        #not used: samples[idx]["n_audio_embs"] = embs.shape[1]  # T

def save_sorted_samples(samples, embedder_path, batch_size, bucket_size, cache_dir, device, dtype):
    # embed (batch_size) samples and save embeddings in files containing bucket_size samples
    batch_indices = []
    bucket = []
    bucket_id = 0
    t_embedding = 0.0
    t_saving = 0.0

    torch_dtype = getattr(torch, dtype)

    # Initialize embedder
    audio_embedder = Embedder(config={'path': embedder_path})
    audio_embedder.to(device, dtype=torch_dtype)
    audio_embedder.eval()

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from JSON (bucketed)")
    parser.add_argument("--json_path", type=str, required=True, help="JSON file with audio metadata")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to store bucket .pt files and meta.json")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium")
    parser.add_argument("--tokenizer_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    #correct the next line
    parser.add_argument("--split", type=str, default=None, help="Split to use (use all if not given)")
    parser.add_argument("--lang", type=str, default=None, help="Transcription language to use (use all if not given)")
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
    key2sample = defaultdict(dict)

    for s in tqdm(samples, total=len(samples), desc="Tokenizing text", unit=" sample"):
        audio_file = s.get("audio_file", "")

        if not isinstance(audio_file, str) or not audio_file.strip():
            continue

        if audio_file in key2sample:
            continue

        split = s.get("split", "")
        if args.split:
            if split != args.split:
                continue

        lang = s.get("slang", "")
        if args.lang:
            if lang != args.lang:
                continue

        text = s.get("transcription", {}).get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        ids = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        len = len(ids)
        key2sample[audio_file] = {"audio_file": audio_file, "text": text, "ids": ids, "lang": lang, "split": split, "len": len}

    if len (key2sample) == 0:
        logger.info("No samples to process after filtering.")
        sys.exit(0)

    logger.info(f"Found {len(samples)} unique audio files with valid transcriptions after tokenization.")

    # Sort samples by tokenized length (shortest → longest)
    samples = list(key2sample.values())
    samples.sort(key=lambda x: x["len"])
    logger.info(f"Sorted {len(samples)} samples by tokenized length. Shortest len={samples[0]['len']}, longest len={samples[-1]['len']}")

    sys.exit(0)

    os.makedirs(args.cache_dir, exist_ok=True)

    #################################################################################
    ### Save audio embeddings in bucketed .pt files #################################
    #################################################################################
    save_sorted_samples(samples, args.embedder_path, args.batch_size, args.bucket_size, args.cache_dir, args.device, args.dtype)

    # Save meta.json
    meta_path = os.path.join(args.cache_dir, "meta.json")
    info = {
        "json_path": args.json_path,
        "cache_dir": args.cache_dir,
        "embedder_path": args.embedder_path,
        "tokenizer_path": args.tokenizer_path,
        "dtype": args.dtype,
        "bucket_size": args.bucket_size,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"info": info, "samples": samples}, f, ensure_ascii=False)
    logger.info(f"Saved {meta_path}")
