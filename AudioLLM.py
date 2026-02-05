# AudioToLLM.py

import torch
import logging
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from Projector import Projector

logger = logging.getLogger("AudioLLM")

class AudioLLM(torch.nn.Module):
    """
    Wrapper combining Embedder -> Projector -> LLM
    """
    def __init__(self, config, device, dtype, is_infer=False):
        super().__init__()

        audio_path = config['audio']['path'] #only to get embedding dim
        llm_path = config['llm']['path']

        self.audio_embedding_dim = AutoConfig.from_pretrained(audio_path).d_model # Audio embedding dimension (Whisper)
        self.llm_embedding_dim = AutoConfig.from_pretrained(llm_path).hidden_size # LLM embedding dimension (EuroLLM-1.7B-Instruct)
        logger.info(f"Audio embedding dimension: {self.audio_embedding_dim}")
        logger.info(f"LLM embedding dimension: {self.llm_embedding_dim}")

        ### load Tokenizer ###        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        logger.info(f"Loaded Tokenizer from {llm_path}")
        logger.info(f"bos_token = {self.tokenizer.bos_token} {self.tokenizer.bos_token_id}")
        logger.info(f"eos_token = {self.tokenizer.eos_token} {self.tokenizer.eos_token_id}")
        if self.tokenizer.pad_token is None:
            raise ValueError("""Tokenizer does not have a PAD token defined (use an LLM with defined pad_token).\nDuring pretraining, the model forces audio embeddings to match text embeddings. Due to length mismatch between audio frames and text tokens, PAD tokens are used to fill the remaining length of transcriptions. During inference, the LLM ignores PAD tokens without additional processing.""")
        logger.info(f"pad_token = {self.tokenizer.pad_token} {self.tokenizer.pad_token_id}")

        ### load LLM Embedder ###
        self.llm_embedder = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True).get_input_embeddings()
        ### load Projector ###
        self.projector = Projector(config['projector'], audio_embedding_dim=self.audio_embedding_dim, llm_embedding_dim=self.llm_embedding_dim)

        logger.info(f"Moving models to device={device}, dtype={dtype} and freezing LLM embedder...")
        self.projector.to(device=device, dtype=dtype)      #use float32 to ensure stability during early training of projector
        self.projector.unfreeze()
        self.llm_embedder.to(device=device, dtype=dtype)      #float16/bfloat16 is for memory efficiency
        self.llm_embedder.eval()
        for p in self.llm_embedder.parameters():
            p.requires_grad = False

        logger.info(f"Projector: {next(self.projector.parameters()).dtype} on {next(self.projector.parameters()).device}")
        logger.info(f"LLM Embedder: {next(self.llm_embedder.parameters()).dtype} on {next(self.llm_embedder.parameters()).device}")

        self.alpha = config['optim']['alpha']
        self.gamma = config['optim']['gamma']

        self.summary()


    def save(self, ckpt_path):
        self.projector.save(ckpt_path)
        self.llm_embedder.save(ckpt_path)


    # ========================================================
    # Forward (training)
    # ========================================================
    def forward(self, target_ids, pt_paths, offsets):

        input_embeds = self.read_cache_embs(pt_paths, offsets)
        input_embeds = input_embeds.to(self.device)
        target_ids = target_ids.to(self.device) #.to(torch.long)
        assert input_embeds.dim() == 3, f"Expected input_embeds to have 3 dimensions [B, T, D], got {input_embeds.shape}"
        assert target_ids.dim() == 2, f"Expected target_ids to have 2 dimensions [B, T], got {target_ids.shape}"
        assert input_embeds.size(0) == target_ids.size(0), f"Batch size mismatch between input_embeds and target_ids: {input_embeds.size(0)} vs {target_ids.size(0)}"
        assert input_embeds.size(1) == target_ids.size(1), f"Sequence length mismatch between input_embeds and target_ids: {input_embeds.size(1)} vs {target_ids.size(1)}"

        # Count pad tokens - 1 (result cannot be lower than 0)
        pad_tokens = (target_ids == self.tokenizer.pad_token_id).sum(dim=1) - 1 # compute loss on pad_tokens with lower weight
        pad_tokens = torch.clamp(pad_tokens, min=0)
        # create the corresponding txt_mask/pad_mask
        txt_mask = torch.arange(target_ids.size(1), device=target_ids.device).expand_as(target_ids) >= pad_tokens.unsqueeze(1) # [B, T] True for valid tokens, False for pad tokens to ignore in loss
        pad_mask = ~txt_mask # [B, T] True for pad tokens, False for valid tokens    

        proj_embs, _ = self.projector(input_embeds)      # [B, S_max, D_llm]
        B, S_max, D = proj_embs.shape
        assert S_max == input_embeds.size(1), f"Expected S_max={input_embeds.size(1)}, got {S_max}"
        assert D == self.llm_embedding_dim, f"Expected D={self.llm_embedding_dim}, got {D}"
        audio_norm = proj_embs.norm(dim=-1).mean()

        text_embs = self.llm_embedder(proj_embs) # [B, S_max, D_llm]
        B, T_max, D = text_embs.shape
        assert T_max == txt_mask.size(1), f"Expected T_max={txt_mask.size(1)}, got {T_max}"
        assert D == self.llm_embedding_dim, f"Expected D={self.llm_embedding_dim}, got {D}"
        text_norm = text_embs.norm(dim=-1).mean()

        loss_mse_txt = torch.nn.functional.mse_loss(text_embs[txt_mask], proj_embs[txt_mask], reduction="mean")
        loss_mse_pad = torch.nn.functional.mse_loss(text_embs[pad_mask], proj_embs[pad_mask], reduction="mean")
        loss_mse = self.alpha * loss_mse_txt + (10-self.alpha) * loss_mse_pad

        loss_cos = torch.nn.functional.cosine_embedding_loss(text_embs, proj_embs, torch.ones(B, device=text_embs.device), reduction="mean")

        loss = loss_mse - self.gamma * loss_cos

        return {
            "loss": loss,
            "labels": target_ids,
            "audio_norm": audio_norm,
            "text_norm": text_norm,
        }


    def read_cache_embs(self, pt_paths, offsets):
        """
        Reads the batch embeddings cached in disk as indicated by pt_paths and offsets
        Args:
            pt_paths (List[str]): bucket filenames
            offsets (List[int] or Tensor): index inside each bucket

        Returns:
            audio_embs: Tensor [B, T, D] (on CPU)
        """

        if not hasattr(self, "_bucket_cache"):
            self._bucket_cache = OrderedDict()

        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        assert len(pt_paths) == len(offsets)

        # Group batch positions by pt_path
        path_to_items = {}
        for batch_idx, (pt_path, offset) in enumerate(zip(pt_paths, offsets)):
            path_to_items.setdefault(pt_path, []).append((batch_idx, offset))

        batch_embs = [None] * len(pt_paths)

        for pt_path, items in path_to_items.items():
            # ---- Load or reuse bucket ----
            if pt_path in self._bucket_cache:
                bucket = self._bucket_cache.pop(pt_path)  # mark as recently used
            else:
                bucket = torch.load(pt_path, map_location="cpu")
                # Enforce buffer size (LRU eviction)
                if len(self._bucket_cache) >= self.buffer_size:
                    self._bucket_cache.popitem(last=False)

            self._bucket_cache[pt_path] = bucket
            bucket_embs = bucket["audio_embs"]  # [B_bucket, T, D]

            # ---- Extract needed embeddings ----
            for batch_idx, offset in items:
                batch_embs[batch_idx] = bucket_embs[offset]

        # Stack in original batch order
        audio_embs = torch.stack(batch_embs, dim=0)

        return audio_embs


    def summary(self):
        """
        Logs a clean summary of model parameters, separating:
        - Projector
        - LLM Embedder
        - TOTAL parameters
        Accurately accounts for frozen and trainable embeddings.
        """
        # -----------------------------
        # Projector
        # -----------------------------
        projector_total = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        # -----------------------------
        # LLM Embeddings
        # -----------------------------
        llm_embedder_total = sum(p.numel() for p in self.llm_embedder.parameters())
        
        # -----------------------------
        # TOTAL
        # -----------------------------
        total_params = llm_embedder_total + projector_total 
        trainable_params = projector_trainable
        frozen_params = total_params - trainable_params

        # -----------------------------
        # Logging
        # -----------------------------
        logger.info("=" * 100)
        logger.info("AudioLLM PARS.              TOTAL |       TRAINABLE |          FROZEN")
        logger.info("=" * 100)
        logger.info(f"Projector       : {projector_total:>15,} | {projector_trainable:>15,} | {projector_total - projector_trainable:>15,}")
        logger.info(f"LLM (Embeddings): {llm_embedder_total:>15,} | {llm_embedder_total:>15,} | 0")
        logger.info("-" * 100)
        logger.info(f"TOTAL           : {total_params:>15,} | {trainable_params:>15,} | {frozen_params:>15,}")
        logger.info(f"Trainable %     : {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 100)
