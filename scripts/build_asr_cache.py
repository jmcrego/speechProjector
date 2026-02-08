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
from itertools import product


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
    split_filter,
    slang_filter,
    cache_dir,
    batch_size,
    bucket_size,
    device,
    torch_dtype,
):
    """
    Embed audio for samples matching a given (split_filter, slang_filter) and save embeddings in bucketed .pt files.
    """
    
    os.makedirs(cache_dir, exist_ok=True)
    samples_path = os.path.join(cache_dir, "samples.jsonl")

    bucket_id = 0
    bucket_embs = None
    bucket_indices = []
    bucket_fill = 0

    # collect indices of samples matching the split/slang filter
    matching_indices = [
        i for i, s in enumerate(samples)
        if s.get("len") is not None
        and s.get("text", "")
        and s.get("split", "") == split_filter
        and s.get("transcription", {}).get("lang", "") == slang_filter
        and s.get("audio_file", "")
    ]

    if not matching_indices:
        logger.info(f"No samples found for split={split_filter}, slang={slang_filter}")
        return samples

    # open JSONL file in append mode
    with open(samples_path, "w", encoding="utf-8") as f_jsonl:

        # process in batches
        for start in tqdm(range(0, len(matching_indices), batch_size), desc=f"Embedding {split_filter}-{slang_filter}", unit=" batch"):
            end = min(start + batch_size, len(matching_indices))
            batch_indices = matching_indices[start:end]

            # Embed a batch and move to CPU immediately to free GPU memory
            audio_embs = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)

            for i, idx in enumerate(batch_indices):
                emb = audio_embs[i]

                # allocate bucket tensor lazily
                if bucket_embs is None:
                    bucket_embs = torch.empty((bucket_size, *emb.shape), device="cpu", dtype=emb.dtype)

                # copy embedding to bucket
                bucket_embs[bucket_fill].copy_(emb)
                bucket_indices.append(idx)
                bucket_fill += 1

                # save bucket if full
                if bucket_fill == bucket_size:
                    save_bucket_tensor(samples, bucket_embs, bucket_indices, cache_dir, bucket_id)

                    # write sample metadata to file
                    for s_idx in bucket_indices:
                        s = samples[s_idx]
                        json.dump({
                            "audio_file": s["audio_file"],
                            "text": s["transcription"]["text"],
                            "pt_path": s["pt_path"],
                            "offset": s["offset"]
                        }, f_jsonl, ensure_ascii=False)
                        f_jsonl.write("\n")
                    f_jsonl.flush()

                    bucket_id += 1
                    bucket_embs = None
                    bucket_indices.clear()
                    bucket_fill = 0

            del audio_embs
            torch.cuda.empty_cache()

        # flush remainder
        if bucket_fill > 0:
            save_bucket_tensor(samples, bucket_embs[:bucket_fill], bucket_indices, cache_dir, bucket_id)

            for s_idx in bucket_indices:
                s = samples[s_idx]
                json.dump({
                    "audio_file": s["audio_file"],
                    "text": s["transcription"]["text"],
                    "pt_path": s["pt_path"],
                    "offset": s["offset"]
                }, f_jsonl, ensure_ascii=False)
                f_jsonl.write("\n")
            f_jsonl.flush()

            bucket_id += 1

    logger.info(f"Saved embeddings for split={split_filter}, slang={slang_filter} in {bucket_id} buckets dir={cache_dir}")



def filter_samples(samples, tokenizer, max_seq_len):

    # combination2samples = defaultdict(list) # dict of (split, slang) → list of samples
    unique_audio_files = set() 
    stats = defaultdict(int)

    split_slang = set() # set of (split, slang) combinations found in samples

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
        #idx = s.get("idx", -1)

        ids = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        if len(ids) > max_seq_len:
            stats['too_long_text'] += 1
            continue

        s["len"] = len(ids)
        s["text"] = text
        split_slang.add((split, slang))
        unique_audio_files.add(audio_file)

    logger.info(f"Found {len(split_slang)} split/slang combinations after filtering")
    logger.info(f"Found {len(unique_audio_files)} unique audio files after filtering")
    for k in sorted(stats.keys()):
        logger.info(f"{k}: {stats[k]}")

    return split_slang


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
    split_slang = filter_samples(samples, tokenizer, max_seq_len=args.max_seq_len)

    if len (split_slang) == 0:
        logger.info("No samples to process after filtering.")
        sys.exit(0)


    # Initialize embedder
    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    audio_embedder = Embedder(config={'path': args.embedder_path})
    audio_embedder.to(args.device, dtype=torch_dtype)
    audio_embedder.eval()

    # sort combinations by numbmer of samples (descending), then by split and slang (ascending)
    # combinations_sorted = sorted(combination2samples.keys(), key=lambda x: (len(combination2samples[x]), x[0], x[1]), reverse=False)

    for idx, (split, slang) in enumerate(split_slang):
        logger.info(f"Saving {idx}/{len(split_slang)} combination ({split}, {slang})")

        cache_dir = os.path.join(args.json_path + "_CACHE_ASR", f"{split}/{slang}")
        meta_path = os.path.join(cache_dir, "meta.json")

        if os.path.exists(meta_path):
            logger.info(f"Cache directory {cache_dir} already contains meta.json, skipping embedding/saving")
            continue

        ### sorting is not really needed for stage1 ###
        ### transcriptions are filled with padding tokens to max_seq_len ###
        # samples.sort(key=lambda x: (x["len"], x["audio_file"])) 

        ### embed audio and save buckets of samples with their embeddings ###
        save_samples_in_buckets(audio_embedder, samples, split, slang, cache_dir, args.batch_size, args.bucket_size, device, torch_dtype)

        ### save meta for this combination (split, slang) ###
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


    logger.info(f"Total time: {time.time() - tic:.2f}s")