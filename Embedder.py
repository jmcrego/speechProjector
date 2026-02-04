#!/usr/bin/env python3

import torch
import logging
import torch.nn as nn

from transformers import WhisperFeatureExtractor, WhisperModel
from scripts.utils import preprocess_audio

logger = logging.getLogger("Embedder")

class Embedder(nn.Module):
    """
    Audio → frame embeddings extractor.
    Output:
      frames: [B, T_frames, D]   float32
      mask:   [B, T_frames]      bool (True = valid frame)
    """

    def __init__(self, config):
        super().__init__()

        self.path = config["path"]

        # ----------------------------------------------------
        # Load speech embedder
        # ----------------------------------------------------
        logger.debug(f"Loading whisper")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.path)
        whisper = WhisperModel.from_pretrained(self.path)
        self.embedder = whisper.encoder
        del whisper.decoder  # free decoder weights
        torch.cuda.empty_cache()
        self.embedding_dim = self.embedder.config.d_model
        self.sample_rate = self.feature_extractor.sampling_rate        
        logger.info(f"Loaded {self.path} | embedding_dim={self.embedding_dim} | sample_rate={self.sample_rate}")
        # print model architecture
        logger.debug(self)

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        logger.info("Audio embedder frozen (eval mode)")
 

    def forward(self, audio_inputs):
        """
        Args:
            audio_inputs: list of str paths or np.ndarray audio
        Returns:
            embeddings: [B, T_max, D] float32
            masks: [B, T_max] bool
        """
        device = next(self.parameters()).device

        # ====================================================
        # 1. Preprocess each audio independently
        # ====================================================
        preprocessed = [ preprocess_audio(a, sample_rate=self.sample_rate) for a in audio_inputs ] # waveforms: list of np.ndarray [T_i]

        # ====================================================
        # 2. Feature extractor (handles padding + mask)
        # ====================================================
        feat = self.feature_extractor(preprocessed, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = feat.input_features.to(device, dtype=torch.float32) #[B, n_mels, T_frames]

        logger.debug(f"Audio feats: {inputs.shape}, dtype={inputs.dtype}")

        # ====================================================
        # 3. Encoder forward
        # ====================================================
        with torch.no_grad():
            frames = self.embedder(inputs).last_hidden_state # [B, T_frames=1500, D]

        logger.debug(f"Audio embed: {frames.shape}")

        # ====================================================
        # 4. Frame-level mask 
        # ====================================================
        # Whisper encoder outputs are always dense (no mask needed; all valild)
        frames_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=device) # [B, T_frames=1500]

        logger.debug(f"Audio masks: {frames_mask.shape}")

        return frames, frames_mask


if __name__ == "__main__":
    import argparse
    import time
    import json
    
    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--audio_files", type=str, help="Comma separated list of audio files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    audio_embedder = Embedder(config=config['audio'])
    t = time.time()
    embeddings, masks = audio_embedder(args.audio_files.split(','))
    print(f"Output embeddings {embeddings.shape}, masks {masks.shape}, took {time.time()-t:.2f} sec")
