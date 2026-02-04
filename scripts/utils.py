#!/usr/bin/env python3

#import torch
#import logging
#import torch.nn as nn

import soxr
import numpy as np
import soundfile as sf


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
