# AudioToLLM.py

import torch
import logging
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F

from Projector import Projector
from Embedder import Embedder

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

        if is_infer:
            ### load LLM ###
            self.llm_embedder = None # not needed during inference, save memory by not loading
            self.llm = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
            self.llm.to(device=device, dtype=dtype) 
            self.llm.eval()
            for p in self.llm.parameters():
                p.requires_grad = False
            logger.info(f"Loaded LLM from {llm_path} and set to eval mode with gradients disabled")

            ### load Audio Embedder ###
            self.embedder = Embedder(config['audio']) 
            self.embedder.to(device=device, dtype=dtype) 
            self.embedder.eval() 
            for p in self.embedder.parameters(): 
                p.requires_grad = False 
            logger.info(f"Loaded Audio Embedder from {config['audio']['path']} and set to eval mode with gradients disabled")

        else:
            ### load LLM Embedder ###
            self.llm = None # not needed during training, save memory by not loading full LLM
            self.embedder = None # not needed during training, save memory by not loading full embedder
            self.llm_embedder = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True).get_input_embeddings()
            self.llm_embedder.to(device=device, dtype=dtype)      #float16/bfloat16 is for memory efficiency
            self.llm_embedder.eval()
            for p in self.llm_embedder.parameters():
                p.requires_grad = False
            logger.info(f"Loaded LLM from {llm_path} and set to eval mode with gradients disabled")

        ### load Projector ###
        self.projector = Projector(config['projector'], audio_embedding_dim=self.audio_embedding_dim, llm_embedding_dim=self.llm_embedding_dim)
        self.projector.to(device=device, dtype=dtype)      #use float32 to ensure stability during early training of projector
        self.projector.unfreeze()

        logger.info(f"Projector: {next(self.projector.parameters()).dtype} on {next(self.projector.parameters()).device}")
        if is_infer:
            logger.info(f"LLM: {next(self.llm.parameters()).dtype} on {next(self.llm.parameters()).device}")
        else:
            logger.info(f"LLM Embedder: {next(self.llm_embedder.parameters()).dtype} on {next(self.llm_embedder.parameters()).device}")

        self.alpha = config['optim']['alpha']
        self.gamma = config['optim']['gamma']

        if not is_infer:
            self.summary()

        self.audio_token = config["llm"]["audio_token"]
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        assert self.audio_token_id is not None, "audio_token_id is None"
        assert isinstance(self.audio_token_id, int), type(self.audio_token_id)
        logger.info(f"Audio token: '{self.audio_token}' -> ID: {self.audio_token_id}")


    def save(self, ckpt_path):
        self.projector.save(ckpt_path)


    # ========================================================
    # Forward (training)
    # ========================================================
    def forward(self, target_ids, pt_paths, offsets):
        # target_ids already in device and dtype of projector, no need to move here
        input_embeds = self.read_cache_embs(pt_paths, offsets) #input_embes already in device and dtype of projector, no need to move here

        assert input_embeds.dim() == 3, f"Expected input_embeds to have 3 dimensions [B, T, D], got {input_embeds.shape}"
        assert target_ids.dim() == 2, f"Expected target_ids to have 2 dimensions [B, T], got {target_ids.shape}"
        assert input_embeds.size(0) == target_ids.size(0), f"Batch size mismatch between input_embeds and target_ids: {input_embeds.size(0)} vs {target_ids.size(0)}"

        # Count txt tokens + 1 (result cannot be more than sequence length of target_ids)
        n_txt_tokens = (target_ids != self.tokenizer.pad_token_id).sum(dim=1) + 1 
        n_txt_tokens = torch.clamp(n_txt_tokens, max=target_ids.size(1)) # ensure n_txt_tokens does not exceed sequence length of target_ids
        # create the corresponding txt_mask/pad_mask
        positions = torch.arange(target_ids.size(1), device=target_ids.device).expand_as(target_ids) # [B, T] 
        txt_mask = positions < n_txt_tokens.unsqueeze(1) # True for valid tokens, False for pad tokens to ignore in loss computation
        pad_mask = ~txt_mask

        # Projector forward
        proj_embs, _ = self.projector(input_embeds) # [B, S_max, D_llm]
        B, S_max, D = proj_embs.shape
        assert S_max == target_ids.size(1), f"Expected S_max={target_ids.size(1)}, got {S_max}"
        assert D == self.llm_embedding_dim, f"Expected D={self.llm_embedding_dim}, got {D}"
        audio_norm = proj_embs.norm(dim=-1).mean()

        # LLM Embedder forward
        text_embs = self.llm_embedder(target_ids) # [B, T_max, D_llm]
        B, T_max, D = text_embs.shape
        assert T_max == proj_embs.size(1), f"Expected T_max={proj_embs.size(1)}, got {T_max}"
        assert D == self.llm_embedding_dim, f"Expected D={self.llm_embedding_dim}, got {D}"
        text_norm = text_embs.norm(dim=-1).mean()

        # ----- MSE: absolute alignment (scale + direction) -----
        loss_mse_txt = F.mse_loss(text_embs[txt_mask], proj_embs[txt_mask], reduction="mean")
        loss_mse_pad = F.mse_loss(text_embs[pad_mask], proj_embs[pad_mask], reduction="mean")

        # Combine MSE contributions
        loss_mse = self.alpha * loss_mse_txt + (1 - self.alpha) * loss_mse_pad

        # ----- Cosine loss: directional alignment -----
        cos = F.cosine_similarity(text_embs[txt_mask], proj_embs[txt_mask], dim=-1)
        # same vectors → cos=1, orthogonal → cos=0, opposite → cos=-1
        loss_cos = 1.0 - cos.mean()

        # ----- Final loss -----
        # loss_mse handles scale + direction, loss_cos handles purely direction
        loss = loss_mse + self.gamma * loss_cos 

        # ----- Logging info -----
        audio_norm = proj_embs.norm(dim=-1).mean()
        text_norm = text_embs.norm(dim=-1).mean()

        return {
            "loss": loss,
            "loss_mse_txt": loss_mse_txt,
            "loss_mse_pad": loss_mse_pad,
            "loss_cos": loss_cos,
            "audio_norm": audio_norm,
            "text_norm": text_norm,
        }

    def generate(
        self, 
        audio_paths, 
        prompt, 
        max_new_tokens=256, 
        temperature=0.7, 
        top_p=0.95,
        no_repeat_ngram_size = 0, #dangerous for ASR/STT, speech allow repetitions
        repetition_penalty = 1.1, #good for ASR/STT, but bad for QA
    ):
        """
        Inference method: generates text given audio paths and prompt
        Args:
            audio_paths (List[str]): list of audio file paths
            prompt (str): prompt text
            max_new_tokens (int): maximum number of new tokens to generate
            temperature (float): sampling temperature
            top_p (float): top-p sampling parameter
            no_repeat_ngram_size (int): no repeat ngram size
            repetition_penalty (float): repetition penalty

        Returns:
            generated_texts (List[str]): list of generated texts
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.projector.linear.weight.device) # move to same device as projector for generation
        formatted_batch = self.format_batch(audio_paths, prompt_ids)

        outputs = self.llm.generate(
            inputs_embeds=formatted_batch["inputs_embeds"],
            attention_mask=formatted_batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            no_repeat_ngram_size = no_repeat_ngram_size, 
            repetition_penalty = repetition_penalty,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        return self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)


    def format_batch(self, audio_paths, prompt_ids):
        """
        Formats a batch by combining prompt, audio, and (optionally) target embeddings.

        Args:
            audio_paths: list of audio file paths
            prompt_ids: [B, T_prompt] input token IDs

        Returns:
            dict with:
                inputs_embeds: [B, L_max, D] final embeddings
                attention_mask: [B, L_max] attention mask
        """
        device = self.llm.device
        llm_dtype = next(self.llm.parameters()).dtype
        B = prompt_ids.size(0)

        # 1) Embed audio
        with torch.no_grad():
            audio_embs, _ = self.embedder(audio_paths)  # [B, T_audio, D_audio]

        audio_embs = audio_embs.to(device)

        # 2) Project audio embeddings
        proj_embs, proj_mask = self.projector(audio_embs)      # [B, S_max, D_llm], [B, S_max]
        proj_mask = proj_mask.bool()
        B, S_max, D = proj_embs.shape
        audio_lens = proj_mask.sum(dim=1)                      # [B]

        # 3) Embed prompt
        prompt_ids = prompt_ids.to(device)
        prompt_embs = self.llm.model.get_input_embeddings()(prompt_ids)  # [B, T_prompt, D]
        prompt_mask = prompt_ids != self.tokenizer.pad_token_id
        prompt_lens = prompt_mask.sum(dim=1)                   # [B]
        T_prompt = prompt_ids.size(1)

        # 4) Locate <extra_id_0> in prompt
        audio_token_mask = prompt_ids == self.audio_token_id
        assert (audio_token_mask.sum(dim=1) == 1).all(), "Each sample must have exactly one <extra_id_0>"
        audio_pos = audio_token_mask.float().argmax(dim=1)  # [B]

        # 5) Allocate final batch
        total_lens = (prompt_lens - 1) + audio_lens

        max_len = total_lens.max().item()
        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=llm_dtype)
        attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.long)

        # 6) Insert prompt tokens before <extra_id_0>
        range_T = torch.arange(T_prompt, device=device).unsqueeze(0)  # [1, T_prompt]
        before_mask = range_T < audio_pos.unsqueeze(1)                # [B, T_prompt]
        b_idx, t_idx = torch.nonzero(before_mask, as_tuple=True)
        inputs_embeds[b_idx, t_idx] = prompt_embs[b_idx, t_idx]
        attention_mask[b_idx, t_idx] = 1

        # 7) Insert audio embeddings
        range_S = torch.arange(S_max, device=device).unsqueeze(0)     # [1, S_max]
        valid_audio = range_S < audio_lens.unsqueeze(1)               # [B, S_max]
        audio_dest = audio_pos.unsqueeze(1) + range_S                 # [B, S_max]
        b_a, s_a = torch.nonzero(valid_audio, as_tuple=True)
        inputs_embeds[b_a, audio_dest[b_a, s_a]] = proj_embs[b_a, s_a]
        attention_mask[b_a, audio_dest[b_a, s_a]] = 1

        # 8) Insert prompt tokens after <extra_id_0>
        after_mask = range_T > audio_pos.unsqueeze(1)                 # [B, T_prompt]
        b_p, t_p = torch.nonzero(after_mask, as_tuple=True)
        after_offset = audio_lens[b_p] + audio_pos[b_p]
        dest_pos = after_offset + (t_p - (audio_pos[b_p] + 1))
        inputs_embeds[b_p, dest_pos] = prompt_embs[b_p, t_p]
        attention_mask[b_p, dest_pos] = 1

        attn_sum = attention_mask.sum(dim=1)
        if not torch.all(attn_sum == total_lens):
            raise RuntimeError(f"Attention mismatch:\nattn_sum={attn_sum}\ntotal_lens={total_lens}")

        output = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        return output
        
    
    def read_cache_embs(self, pt_paths, offsets):
        """
        Reads the batch embeddings cached in disk as indicated by pt_paths and offsets
        Args:
            pt_paths (List[str]): bucket filenames
            offsets (List[int]): index inside each bucket

        Returns:
            audio_embs: Tensor [B, T, D] (on CPU)
        """

        if not hasattr(self, "_bucket_cache"):
            self._bucket_cache = OrderedDict()
            self.buffer_size = 2  # max number of buckets to keep in memory (LRU eviction)

        # if isinstance(offsets, torch.Tensor):
        #     offsets = offsets.tolist()

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
        audio_embs = audio_embs.to(self.projector.linear.weight.device) # move to same device as projector for forward pass

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
        logger.info(f"LLM Embedder    : {llm_embedder_total:>15,} | {0:>15,} | {llm_embedder_total:>15,}")
        logger.info("-" * 100)
        logger.info(f"TOTAL           : {total_params:>15,} | {trainable_params:>15,} | {frozen_params:>15,}")
        logger.info(f"Trainable %     : {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 100)
