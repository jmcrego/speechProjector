#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime
import torch
import soxr
import os
import logging
from tqdm import tqdm
import numpy as np
import soundfile as sf
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("utils")

class JSONMetricsLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **data):
        # remove from data None entries to reduce log size and improve readability
        data = {k: v for k, v in data.items() if v is not None}
        data["timestamp"] = datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_device_dtype():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # picks default CUDA device
        try:
            props = torch.cuda.get_device_properties(device)
            name = props.name.lower()
            if "h100" in name:
                dtype = torch.bfloat16
            elif "a100" in name:
                dtype = torch.bfloat16  # optional, you could also use fp16
            else:  # V100, T4, etc.
                dtype = torch.float16
        except Exception:
            dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    dtype = torch.float32
    return device, dtype    


def preprocess_audio(audio_input, sample_rate=None, channel=-1, norm=True):
    """
    Load audio from:
      - file path (str)
      - numpy array (float waveform)

    Returns:
      wav: np.ndarray, shape [T_samples], dtype float32
    """

    # -----------------------------
    # Load audio
    # -----------------------------
    if isinstance(audio_input, str):
        wav, sr = sf.read(audio_input)              # wav: [T] or [T, C]
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input
        sr = sample_rate
    else:
        raise ValueError("audio_input must be a path or np.ndarray")
    # logger.debug(f"preprocess: loaded wav shape={wav.shape}, sr={sr}")

    # -----------------------------
    # Convert to numpy float32
    # -----------------------------
    wav = wav.astype(np.float32)

    # -----------------------------
    # Normalize to [-1,1] if needed (Whisper expects this)
    # -----------------------------
    if norm:
        wav /= np.abs(wav).max() + 1e-9  # prevents div by zero
        # logger.debug(f"preprocess: normalized to [-1, 1]")

    # -----------------------------
    # Convert to mono
    # -----------------------------
    if wav.ndim > 1:
        if channel == -1:
            wav = wav.mean(axis=1)                  # [T]
        else:
            wav = wav[:, channel]                   # [T]
        # logger.debug(f"preprocess: converted to mono wav shape={wav.shape} using channel {channel}")

    # -----------------------------
    # Resample if needed
    # -----------------------------
    if sample_rate is not None and sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        # logger.debug(f"preprocess: resample to {sample_rate} wav shape={wav.shape}")

    return wav


def compute_grad_norm(params, eps=1e-6):
    """
    Compute total gradient norm of a list of parameters.
    Skips parameters with no gradient. Returns a tensor on the same device as the first param.
    """
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    # stack grads and compute total norm
    stacked = torch.stack([g.pow(2).sum() for g in grads])
    total_norm = torch.sqrt(stacked.sum() + eps)
    return total_norm


# def read_samples_from_jsonl(path: str, max_duration: float = 30.0, sep: str = "\t", split=None, slang=None, tlang=None, use_tqdm=True):
#     """
#     Read samples from a JSONL file and build training examples.
#     """    
#     samples = []
#     stats = defaultdict(int)

#     # read jsonl line by line
#     with open(path, "r", encoding="utf-8") as f:
#         for line_no, line in enumerate(tqdm(f, desc=f"Reading {Path(path).name}", unit=" sample", disable=not use_tqdm), start=1):
        
#             entry = json.loads(line)

#             if split is not None:
#                 if entry.get("split", "") != split:
#                     continue

#             audio_path = entry.get("audio_file", None)
#             if audio_path is None:
#                 audio_path = entry.get("audio_path")
#             if audio_path is None:
#                 stats['missing_audio_path'] += 1
#                 continue

#             transcription = entry.get("transcription")

#             if transcription is not None:
#                 # ASR sample
#                 src_lang = transcription.get("lang", "").strip()
#                 if not src_lang:
#                     stats['empty_src_lang'] += 1
#                     continue
#                 if slang is not None and src_lang != slang:
#                     continue

#                 src_text = transcription.get("text", "").strip()
#                 if not src_text:
#                     stats['empty_src_text'] += 1
#                     continue
#             else:
#                 stats['missing_transcriptions'] += 1
            

#             translation = entry.get("translation")

#             if translation is not None:
#                 # STT sample
#                 tgt_lang = translation.get("lang", "").strip()
#                 if not tgt_lang:
#                     stats['empty_tgt_lang'] += 1
#                     continue
#                 if tlang is not None and tgt_lang != tlang:
#                     continue

#                 tgt_text = translation.get("text", "").strip()
#                 if not tgt_text:
#                     stats['empty_tgt_text'] += 1
#                     continue
#             else:
#                 stats['missing_translations'] += 1


#             try:
#                 info = sf.info(audio_path)
#                 if not info.frames or not info.samplerate:
#                     stats['invalid_audio_info'] += 1
#                     continue
#                 duration = info.frames / info.samplerate
#                 if duration > max_duration:
#                     stats['too_long_duration'] += 1
#                     continue

#             except Exception as e:
#                 logger.warning(f"{path}:{line_no} failed to read audio: {e}")
#                 continue                

#             entry["duration"] = duration

#             stats['duration_sum'] += duration
#             stats['duration_max'] = max(stats.get('duration_max', 0), duration)
#             stats['duration_min'] = min(stats.get('duration_min', float('inf')), duration)
#             samples.append(entry)

#     stats['duration_avg'] = stats.get('duration_sum', 0) / len(samples) if samples else 0.0
#     logger.info(f"samples: {len(samples)}")
#     for k in sorted(stats.keys()): # traverse stats lexicographically sorted by key and log content
#         logger.info(f"{k}: {stats[k]:.2f}") if isinstance(stats[k], float) else logger.info(f"{k}: {stats[k]}")

#     return samples

def duration(audio_path):
    try:
        info = sf.info(audio_path)
        if not info.frames or not info.samplerate:
            return 0.0
        return info.frames / info.samplerate
    except Exception as e:
        logger.warning(f"duration: failed to read audio {audio_path}: {e}")
        return 0.0

def _process_entry(entry, path, max_duration, split, slang, tlang):
    """
    Process a single JSONL entry.
    Returns: (entry_or_None, stats_dict)
    """
    stats = defaultdict(int)

    if split is not None and entry.get("split", "") != split:
        return None, stats

    audio_path = entry.get("audio_file") or entry.get("audio_path")
    if audio_path is None:
        stats["missing_audio_path"] += 1
        return None, stats

    transcription = entry.get("transcription")
    if transcription is not None:
        src_lang = transcription.get("lang", "").strip()
        if not src_lang:
            stats["empty_src_lang"] += 1
            return None, stats
        if slang is not None and src_lang != slang:
            return None, stats

        src_text = transcription.get("text", "").strip()
        if not src_text:
            stats["empty_src_text"] += 1
            return None, stats
    else:
        stats["missing_transcriptions"] += 1

    translation = entry.get("translation")
    if translation is not None:
        tgt_lang = translation.get("lang", "").strip()
        if not tgt_lang:
            stats["empty_tgt_lang"] += 1
            return None, stats
        if tlang is not None and tgt_lang != tlang:
            return None, stats

        tgt_text = translation.get("text", "").strip()
        if not tgt_text:
            stats["empty_tgt_text"] += 1
            return None, stats
    else:
        stats["missing_translations"] += 1

    try:
        info = sf.info(audio_path)
        if not info.frames or not info.samplerate:
            stats["invalid_audio_info"] += 1
            return None, stats

        duration = info.frames / info.samplerate
        if duration > max_duration:
            stats["too_long_duration"] += 1
            return None, stats

    except Exception as e:
        logger.warning(f"{path}:{entry['idx']} failed to read audio: {e}")
        stats["audio_read_error"] += 1
        return None, stats

    entry["duration"] = duration
    stats["duration_sum"] += duration
    stats["duration_max"] = duration
    stats["duration_min"] = duration

    return entry, stats


def read_samples_from_jsonl(
    path: str,
    max_duration: float = 30.0,
    sep: str = "\t",
    split=None,
    slang=None,
    tlang=None,
    use_tqdm=True,
    num_workers: int = 32,
):
    """
    Read samples from a JSONL file and build training examples (parallel audio metadata).
    """
    num_workers = min(num_workers, os.cpu_count() * 2)
    samples = []
    stats = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=num_workers) as ex:

        futures = []
        for idx, line in enumerate(f, start=1):
            entry = json.loads(line)
            entry['idx'] = idx
            futures.append( ex.submit(_process_entry, entry, path, max_duration, split, slang, tlang) )

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Reading {Path(path).name}",
            unit=" sample",
            disable=not use_tqdm,
        ):
            entry, local_stats = future.result()

            for k, v in local_stats.items():
                if k in ("duration_max", "duration_min"):
                    if k == "duration_max":
                        stats[k] = max(stats.get(k, 0), v)
                    else:
                        stats[k] = min(stats.get(k, float("inf")), v)
                else:
                    stats[k] += v

            if entry is not None:
                samples.append(entry)

    stats['duration_avg'] = stats.get('duration_sum', 0) / len(samples) if samples else 0.0
    hours, minutes, seconds = 0, 0, 0
    if stats.get('duration_sum', 0) > 0:
        hours = int(stats['duration_sum'] // 3600)
        minutes = int((stats['duration_sum'] % 3600) // 60)
        seconds = int(stats['duration_sum'] % 60)
    stats['duration_sum'] = f"{hours}h{minutes}m{seconds}s"
    logger.info(f"samples: {len(samples)}")
    for k in sorted(stats.keys()): # traverse stats lexicographically sorted by key and log content
        logger.info(f"{k}: {stats[k]:.2f}") if isinstance(stats[k], float) else logger.info(f"{k}: {stats[k]}")

    # sort samples by original line number in JSONL (idx) to ensure deterministic order for reproducibility and easier debugging
    samples.sort(key=lambda x: x["idx"], reverse=False) 
    return samples