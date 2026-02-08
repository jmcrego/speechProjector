from tqdm import tqdm
import argparse
import logging
import torch
import json
import os
import gc
import sys
import time
from collections import defaultdict
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import read_samples_from_jsonl
from scripts.text_normalize import normalize_text
from Embedder import Embedder

logger = logging.getLogger("build_asr_cache")


def process_batch(audio_embedder, samples, batch_indices, device, dtype):
    """
    Embed audio for a batch of indices.
    Returns embeddings on CPU.
    """
    audio_paths = [samples[idx]["audio_file"] for idx in batch_indices]

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            audio_res = audio_embedder(audio_paths)  # Whisper: ignore mask

        audio_embs = audio_res[0] if isinstance(audio_res, tuple) else audio_res

    return audio_embs.cpu().contiguous()


def save_bucket_tensor(samples, embs, indices, cache_dir, bucket_id):
    pt_path = os.path.join(cache_dir, f"bucket_bs{len(indices)}_{bucket_id:06d}.pt")
    tmp_path = pt_path + ".tmp"

    if not os.path.exists(pt_path):
        torch.save({"audio_embs": embs}, tmp_path)
        os.replace(tmp_path, pt_path)

    for i, idx in enumerate(indices):
        samples[idx]["pt_path"] = os.path.basename(pt_path)
        samples[idx]["offset"] = i


def save_samples_in_buckets(
    audio_embedder,
    samples,
    cache_dir,
    batch_size,
    bucket_size,
    device,
    torch_dtype,
):
    os.makedirs(cache_dir, exist_ok=True)

    bucket_id = 0
    bucket_embs = None
    bucket_indices = []
    bucket_fill = 0

    for start in tqdm(range(0, len(samples), batch_size), desc="Embedding audio", unit=" sample",):
        end = min(start + batch_size, len(samples))
        batch_indices = list(range(start, end))

        # Embed a batch of audio and move to CPU immediately to free GPU memory asap, non_blocking if pinned memory
        audio_embs = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)

        for i, idx in enumerate(batch_indices):
            emb = audio_embs[i]

            # allocate bucket tensor lazily
            if bucket_embs is None:
                bucket_embs = torch.empty( (bucket_size, *emb.shape), device="cpu", dtype=emb.dtype, pin_memory=True)

            bucket_embs[bucket_fill].copy_(emb, non_blocking=True) # copy to bucket tensor on CPU (to free GPU memory asap), non_blocking if pinned memory
            bucket_indices.append(idx)
            bucket_fill += 1 # bucket_fill tracks how many samples are currently in bucket_embs

            if bucket_fill == bucket_size:
                save_bucket_tensor(samples, bucket_embs, bucket_indices, cache_dir, bucket_id)

                bucket_id += 1
                bucket_embs = None
                bucket_indices.clear()
                bucket_fill = 0

        del audio_embs
        torch.cuda.empty_cache()

    # flush remainder
    if bucket_fill > 0:
        save_bucket_tensor(samples, bucket_embs[:bucket_fill], bucket_indices, cache_dir, bucket_id)
        bucket_id += 1

    logger.info(f"Saved {len(samples)} embeddings in {bucket_id} buckets dir={cache_dir}")

    return samples


# def save_samples_in_buckets(audio_embedder, samples, cache_dir, batch_size, bucket_size, device, torch_dtype):
#     # embed (batch_size) samples and save embeddings in files containing bucket_size samples
#     batch_indices = []
#     bucket = []
#     bucket_id = 0
#     t_embedding = 0.0
#     t_saving = 0.0

#     os.makedirs(cache_dir, exist_ok=True)

#     for idx in tqdm(range(len(samples)), total=len(samples), desc="Embedding audio", unit=" sample"):

#         batch_indices.append(idx)

#         # process batch
#         if len(batch_indices) == batch_size:
#             tic = time.time()
#             audio_embs_cpu = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)
#             t_embedding += time.time() - tic

#             # one by one copy audio_embs_cpu to bucket, clear audio_embs_cpu as soon as possible to free memory
#             for i, idx in enumerate(batch_indices):
#                 bucket.append((idx, audio_embs_cpu[i]))
#                 del audio_embs_cpu[i]
#                 gc.collect()

#                 # and save bucket when it reaches bucket_size
#                 if len(bucket) == bucket_size:
#                     tic = time.time()
#                     save_bucket(samples, bucket, cache_dir, bucket_id)
#                     t_saving += time.time() - tic
#                     bucket_id += 1
#                     bucket.clear()
#                     gc.collect()

#             batch_indices = []

#     # process remaining batch
#     if batch_indices:
#         tic = time.time()
#         audio_embs_cpu = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)
#         t_embedding += time.time() - tic
#         for i, idx in enumerate(batch_indices):
#             bucket.append((idx, audio_embs_cpu[i]))
#             del audio_embs_cpu[i]
#             gc.collect()

#             # and save bucket when it reaches bucket_size
#             if len(bucket) == bucket_size:
#                 tic = time.time()
#                 save_bucket(samples, bucket, cache_dir, bucket_id)
#                 t_saving += time.time() - tic
#                 bucket_id += 1
#                 bucket.clear()
#                 gc.collect()
#         batch_indices = []

#     # process remaining bucket
#     if len(bucket):
#         tic = time.time()
#         save_bucket(samples, bucket, cache_dir, bucket_id)
#         t_saving += time.time() - tic
#         bucket_id += 1
#         bucket.clear()
#         gc.collect()

#     logger.info(f"Saved {len(samples)} embeddings in {bucket_id} buckets dir={cache_dir}")
#     logger.info(f"Embedding time = {t_embedding:.2f}s, Saving time = {t_saving:.2f}s")

#     return samples


def filter_and_group_samples(samples, tokenizer=None, max_seq_len=None):

    combination2samples = defaultdict(list) # dict of (split, slang) → list of samples
    unique_audio_files = set() 
    splits = set()
    slangs = set()
    stats = defaultdict(int)

    for s in tqdm(samples, total=len(samples), desc="Filtering samples", unit=" sample"):
        audio_file = s.get("audio_file", "")
        if not isinstance(audio_file, str) or not audio_file.strip():
            stats['empty_audio_file'] += 1
            continue

        if audio_file in unique_audio_files:
            stats['repeated_audio_file'] += 1
            continue

        text = s.get("transcription", {}).get("text", "")
        text = normalize_text(text) if text else ""
        if not isinstance(text, str) or not text.strip():
            stats['empty_text'] += 1
            continue

        split = s.get("split", "None")
        slang = s.get("transcription", {}).get("lang", "None")
        idx = s.get("idx", -1)

        if tokenizer is not None:
            ids = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
            if max_seq_len is not None and len(ids) > max_seq_len:
                stats['too_long_text'] += 1
                continue

        s = {"audio_file": audio_file, "text": text, "len": len(ids) if tokenizer is not None else 0} #, "idx": idx, "split": split, "slang": slang}
        combination2samples[(split, slang)].append(s)
        splits.add(split)
        slangs.add(slang)
        unique_audio_files.add(audio_file)


    logger.info(f"Found {len(unique_audio_files)} unique audio files after filtering")
    for k in sorted(stats.keys()):
        logger.info(f"{k}: {stats[k]}")

    logger.info(f"Splits: {splits}")
    logger.info(f"slangs: {slangs}")

    # sort combinations by numbmer of samples (descending), then by split and slang (ascending)
    combinations_sorted = sorted(combination2samples.keys(), key=lambda x: (len(combination2samples[x]), x[0], x[1]), reverse=False)
    logger.info("Combinations (split, slang) and their counts:")
    for split, slang in combinations_sorted:
        logger.info(f"  ({split}, {slang}): {len(combination2samples[(split, slang)])} samples")

    return combination2samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from JSON (bucketed)")
    parser.add_argument("--json_path", type=str, required=True, help="JSON file with audio metadata")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium")
    parser.add_argument("--tokenizer_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype for embeddings")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of samples to fed to embedder")
    parser.add_argument("--bucket_size", type=int, default=256, help="Number of samples per saved bucket")
    parser.add_argument("--max_seq_len", type=int, default=1500 // 15, help="Max sequence length of the transcription. Usually the number of embeddings output by the projector (WHISPER_frames=1500 // PROJECTOR_conv_stride)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(f"{args.json_path}_CACHE_ASR.log", mode='a', encoding='utf-8')
    ])

    tic = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    # Read JSON samples
    samples = read_samples_from_jsonl(args.json_path, max_duration=30.0)

    # Filter and group samples by (split, slang)
    combination2samples = filter_and_group_samples(samples, tokenizer, max_seq_len=args.max_seq_len)

    if len (combination2samples) == 0:
        logger.info("No samples to process after filtering.")
        sys.exit(0)


    # Initialize embedder
    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    audio_embedder = Embedder(config={'path': args.embedder_path})
    audio_embedder.to(args.device, dtype=torch_dtype)
    audio_embedder.eval()

    # sort combinations by numbmer of samples (descending), then by split and slang (ascending)
    combinations_sorted = sorted(combination2samples.keys(), key=lambda x: (len(combination2samples[x]), x[0], x[1]), reverse=False)

    for idx, (split, slang) in enumerate(combinations_sorted, 1):
        samples = combination2samples[(split, slang)]
        logger.info(f"Combination {idx}/{len(combination2samples.keys())} ({split}, {slang}): {len(samples)} samples")

        cache_dir = os.path.join(args.json_path + "_CACHE_ASR", f"{split}/{slang}")
        meta_path = os.path.join(cache_dir, "meta.json")
        samples_path = os.path.join(cache_dir, "samples.jsonl")

        if os.path.exists(meta_path) and os.path.exists(samples_path):
            logger.info(f"Cache directory {cache_dir} already contains meta.json/samples.jsonl, skipping embedding/saving")
            continue

        ### sorting is not really needed for stage1 ###
        ### transcriptions are filled with padding tokens to max_seq_len ###
        # samples.sort(key=lambda x: (x["len"], x["audio_file"])) 

        ### embed audio and save buckets of samples with their embeddings ###
        samples = save_samples_in_buckets(audio_embedder, samples, cache_dir, args.batch_size, args.bucket_size, device, torch_dtype)

        ### save meta and samples for this combination (split, slang) ###
        info = {
            "json_path": args.json_path,
            "cache_dir": cache_dir,
            "embedder_path": args.embedder_path,
            "tokenizer_path": args.tokenizer_path,
            "max_seq_len": args.max_seq_len,
            "bucket_size": args.bucket_size,
            "dtype": args.dtype,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        with open(samples_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


    logger.info(f"Total time: {time.time() - tic:.2f}s")