#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime
import torch
import soxr
import numpy as np
import soundfile as sf

class JSONMetricsLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **data):
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
