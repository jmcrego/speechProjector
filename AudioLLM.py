# AudioLLM.py

import torch
import logging
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from Embedder import Embedder
from Projector import Projector
from LLM import LLM

logger = logging.getLogger("AudioLLM")

class AudioLLM(torch.nn.Module):
    """
    Wrapper combining Embedder -> Projector -> LLM
    """
    def __init__(
            self, 
            config, 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            dtype=torch.float32, 
            weights={}, 
            is_infer=False
        ):

        super().__init__()

        self.weights = weights    

        audio_path = config['audio']['path'] #only to get embedding dim
        llm_path = config['llm']['path']

        self.audio_embedding_dim = AutoConfig.from_pretrained(audio_path).d_model # Audio embedding dimension (Whisper)
        self.llm_embedding_dim = AutoConfig.from_pretrained(llm_path).hidden_size # LLM embedding dimension (EuroLLM-1.7B-Instruct)
        logger.info(f"Audio embedding dimension: {self.audio_embedding_dim}")
        logger.info(f"LLM embedding dimension: {self.llm_embedding_dim}")

        if is_infer: 
            ### load Audio Embedder ###
            self.audio_embedder = Embedder(config['audio'])
            self.audio_embedder.to(device=device, dtype=dtype)
            self.audio_embedder.freeze() # audio embedder always freezed
        else:
            self.audio_embedder = None

        ### load Projector ###
        self.projector = Projector(config['projector'], audio_embedding_dim=self.audio_embedding_dim, llm_embedding_dim=self.llm_embedding_dim)
        self.projector.to(device=device, dtype=dtype)
        if not is_infer:
            self.projector.unfreeze() # ensure projector is in train mode (unfrozen) by default
        else:
            self.projector.freeze() # freeze projector for inference

        # load only the LLM embedding layer during training and when CE loss is not used, to save GPU memory, otherwise load the full LLM
        self.llm = LLM(
            config['llm'], 
            config_lora=None, 
            load_only_embedding_layer=not is_infer and weights.get('CE', 0.) == 0.
        ) 
        self.llm.to(device=device, dtype=dtype)
        if not is_infer and False: #unfreeze also if lora is activated
            self.llm.unfreeze_lora()
        else:
            self.llm.freeze()

        self.audio_token = config["llm"]["audio_token"]
        self.audio_token_id = self.llm.tokenizer.convert_tokens_to_ids(self.audio_token)
        assert self.audio_token_id is not None, "audio_token_id is None"
        assert isinstance(self.audio_token_id, int), type(self.audio_token_id)
        logger.info(f"Audio token: '{self.audio_token}' -> ID: {self.audio_token_id}")

        self.summary()

    # ========================================================
    # Forward (training)
    # ========================================================
    def forward(self, target_ids, pt_paths, offsets, prompt_ids=None):
        pad_id = self.llm.tokenizer.pad_token_id
        eos_id = self.llm.tokenizer.eos_token_id

        # prompt_ids and reference_ids are only needed if CE loss over LLM output embeddings is used (weights['CE'] > 0)
        # otherwise, they can be ignored and set to None to save memory and computation during training
        
        # target_ids already in device and dtype of projector, no need to move here
        input_embeds = self.read_cache_embs(pt_paths, offsets) #input_embes already in device and dtype of projector, no need to move here

        assert input_embeds.dim() == 3, f"Expected input_embeds to have 3 dimensions [B, T, D], got {input_embeds.shape}"
        assert target_ids.dim() == 2, f"Expected target_ids to have 2 dimensions [B, T], got {target_ids.shape}"
        assert input_embeds.size(0) == target_ids.size(0), f"Batch size mismatch between input_embeds and target_ids: {input_embeds.size(0)} vs {target_ids.size(0)}"
 
        # Count txt tokens + 1 (result cannot be more than sequence length of target_ids)
        n_txt_tokens = (target_ids != self.llm.tokenizer.pad_token_id).sum(dim=1) + 1 
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
        proj_norm = proj_embs.norm(dim=-1).mean()

        # LLM Embedder forward
        text_embs = self.llm.embedder(target_ids) # [B, T_max, D_llm]
        B, T_max, D = text_embs.shape
        assert T_max == proj_embs.size(1), f"Expected T_max={proj_embs.size(1)}, got {T_max}"
        assert D == self.llm_embedding_dim, f"Expected D={self.llm_embedding_dim}, got {D}"
        text_norm = text_embs.norm(dim=-1).mean()

        dout = {}
        dout['proj_norm'] = proj_norm.item()
        dout['text_norm'] = text_norm.item()

        # --------------
        # --- losses ---
        # --------------
        loss = torch.tensor(0.0, device=proj_embs.device)

        # ----- MSE: absolute alignment (scale + direction) -----
        loss_mse_txt = F.mse_loss(text_embs[txt_mask], proj_embs[txt_mask], reduction="mean")
        loss_mse_pad = F.mse_loss(text_embs[pad_mask], proj_embs[pad_mask], reduction="mean")
        dout['loss_mse_txt'] = loss_mse_txt.item()
        dout['loss_mse_pad'] = loss_mse_pad.item()
        if self.weights.get('mse', 0) > 0:
            loss_mse = self.weights.get('alpha', 0.5) * loss_mse_txt + (1 - self.weights.get('alpha', 0.5)) * loss_mse_pad
            loss += self.weights.get('mse', 0) * loss_mse

        # ----- Cosine loss: directional alignment -----
        loss_cos = 1.0 - F.cosine_similarity(text_embs[txt_mask], proj_embs[txt_mask], dim=-1).mean()
        dout['loss_cos'] = loss_cos.item()
        if self.weights.get('cos', 0) > 0:
            loss += self.weights.get('cos', 0) * loss_cos

        # ----- Contrastive loss: directional alignment with temperature scaling -----
        proj_embs_norm = F.normalize(proj_embs, dim=-1)
        text_embs_norm = F.normalize(text_embs, dim=-1)
        B, T, D = proj_embs_norm.shape
        # [B, T, T]
        logits = torch.matmul(proj_embs_norm, text_embs_norm.transpose(1, 2))
        logits = logits / self.weights.get('temp_contrast', 1.0)
        # Build per-token targets
        targets = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, T)
        # Flatten
        logits = logits.reshape(B * T, T)
        targets = targets.reshape(B * T)
        # Flatten mask
        valid_mask = txt_mask.reshape(B * T)
        # Keep only valid tokens
        logits = logits[valid_mask]
        targets = targets[valid_mask]
        loss_contrast = F.cross_entropy(logits, targets)
        dout['loss_contrast'] = loss_contrast.item()
        if self.weights.get('contrast', 0) > 0:
            loss += self.weights.get('contrast', 0) * loss_contrast

        # ----- scale loss: handles scale differences -----
        loss_scale = ((proj_embs.norm(dim=-1) - text_embs.norm(dim=-1))**2)[txt_mask].mean()
        dout['loss_scale'] = loss_scale.item()
        if self.weights.get('scale', 0) > 0:
            loss += self.weights.get('scale', 0) * loss_scale

        # --- Cross-entropy loss: handles token-level embedding prediction ---
        logits_txt = torch.matmul(proj_embs[txt_mask], self.llm.embedder.weight.t()) / self.weights.get('temp_ce', 1.0) # logits: [N_txt, D] x [D, V] -> [N_txt, V]
        loss_ce_txt = F.cross_entropy(logits_txt, target_ids[txt_mask], reduction="mean")
        dout['loss_ce_txt'] = loss_ce_txt.item()
        logits_pad = torch.matmul(proj_embs[pad_mask], self.llm.embedder.weight.t()) / self.weights.get('temp_ce', 1.0) # logits: [N_pad, D] x [D, V] -> [N_pad, V]
        loss_ce_pad = F.cross_entropy(logits_pad, target_ids[pad_mask], reduction="mean")
        dout['loss_ce_pad'] = loss_ce_pad.item()
        if self.weights.get('ce', 0) > 0: # only txt is used
            loss += self.weights.get('ce', 0) * loss_ce_txt

        # ----- Accuracy metric for pad predictions ---
        logits_all = torch.matmul(proj_embs, self.llm.embedder.weight.t())

        # indices of the first predicted pad token in each sequence
        first_pad_pos_pre = (logits_all.argmax(dim=-1) == pad_id).float().argmax(dim=1) # [B]
        # indices of the first reference pad token in each sequence
        first_pad_pos_ref = (target_ids == pad_id).float().argmax(dim=1) # [B]
        # average distance between predicted and reference pad positions
        pad_pos_distance = (first_pad_pos_pre - first_pad_pos_ref).abs().float().mean()
        dout['pos_pad'] = pad_pos_distance.item()

        # accuracy of pad prediction over all tokens: percentage of tokens where the model correctly predicts whether it's a pad token or not, averaged over the batch
        pred_pad_all = logits_all.argmax(dim=-1) == pad_id # [B, T], True where model predicts pad token
        ref_pad_all = target_ids == pad_id # [B, T], True where reference has pad token
        n_correct = (pred_pad_all == ref_pad_all).float().sum() # count correct predictions over all tokens
        acc_pad_all = n_correct / ref_pad_all.numel() # normalize by total number of tokens (at least one pad token per sequence, so numel > 0)
        dout['acc_pad'] = acc_pad_all.item()

        # --- Cross-entropy loss over LLM output embeddings: handles token-level prediction at LLM output level (after generation) ---
        if self.weights.get('CE', 0) > 0:
            # in target_ids sequences replace the left-most <pad> token by an <eos> token
            # to ensure the model learns to predict the end of the sequence and does not keep generating padding tokens
            # the remaining <pad> tokens after the first one can be left as they will be ignored
            target_ids_llm = target_ids.clone()
            pad_mask = (target_ids_llm == pad_id)
            assert pad_mask.any(dim=1).all(), "Each sequence in target_ids must contain at least one pad token to replace with eos token"
            first_pad_pos = pad_mask.float().argmax(dim=1)
            rows = torch.arange(B, device=target_ids.device)
            target_ids_llm[rows, first_pad_pos] = eos_id

            formatted_batch = self.format_batch(proj_embs, prompt_ids, target_ids=target_ids_llm)
            # outputs = self.llm.model(
            outputs = self.llm(
                inputs_embeds=formatted_batch['inputs_embeds'],
                attention_mask=formatted_batch['attention_mask'],
                labels=formatted_batch['labels'],
                return_dict=True,
            )
            dout['loss_CE'] = outputs.loss.item()
            loss += self.weights.get('CE', 0) * outputs.loss

        dout['loss'] = loss
        return dout


    def format_batch(self, proj_embs, prompt_ids, target_ids=None):
        device = next(self.llm.parameters()).device 
        dtype = next(self.llm.parameters()).dtype
        B = prompt_ids.size(0)

        pad_id = self.llm.tokenizer.pad_token_id

        # Split prompt into pre-audio and post-audio parts based on the position of the audio token
        prompt_pre_list = []
        prompt_pos_list = []
        for pids in prompt_ids:
            # find index of self.audio_token_id
            idx = (pids == self.audio_token_id).nonzero(as_tuple=True)[0]
            if len(idx) != 1:
                raise ValueError("Each prompt must contain exactly one audio token")
            idx = idx.item()
            prompt_pre_list.append(pids[:idx])     # tokens before audio token
            prompt_pos_list.append(pids[idx+1:])   # tokens after audio token

        # Pad prompt_pre and prompt_pos separately
        prompt_pre_ids = pad_sequence(prompt_pre_list, batch_first=True, padding_value=pad_id).to(device)
        prompt_pos_ids = pad_sequence(prompt_pos_list, batch_first=True, padding_value=pad_id).to(device)

        if target_ids is not None:
            target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_id).to(device)

        # Embed prompt and target
        prompt_pre_embs = self.llm.embedder(prompt_pre_ids)   # (B, Lpre_max, D)
        prompt_pos_embs = self.llm.embedder(prompt_pos_ids) # (B, Lpost_max, D)
        if target_ids is not None:
            target_embs = self.llm.embedder(target_ids)       # (B, Ltgt_max, D)

        # concat sequences
        if target_ids is not None:
            inputs_embeds = torch.cat([prompt_pre_embs, proj_embs, prompt_pos_embs, target_embs], dim=1)
        else:
            inputs_embeds = torch.cat([prompt_pre_embs, proj_embs, prompt_pos_embs], dim=1)

        # build attention mask
        pre_mask = (prompt_pre_ids != pad_id)
        pos_mask = (prompt_pos_ids != pad_id)
        proj_mask = torch.ones(B, proj_embs.size(1), device=device).bool()

        if target_ids is not None:
            tgt_mask = (target_ids != pad_id)
            attention_mask = torch.cat([pre_mask, proj_mask, pos_mask, tgt_mask], dim=1)
        else:
            attention_mask = torch.cat([pre_mask, proj_mask, pos_mask], dim=1)

        # -------------------------
        # Build FULL labels tensor
        # -------------------------
        labels = None
        if target_ids is not None:
            B, T, _ = inputs_embeds.shape
            labels = torch.full((B, T), -100, device=device)

            # Compute where targets start
            start = ( prompt_pre_ids.size(1) + proj_embs.size(1) + prompt_pos_ids.size(1) )
            labels[:, start:start + target_ids.size(1)] = target_ids

            # Ignore padding in targets
            labels[labels == pad_id] = -100

        return {
            "inputs_embeds": inputs_embeds.to(dtype),
            "attention_mask": attention_mask,
            "labels": labels,
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

        pad_id = self.llm.tokenizer.pad_token_id
        eos_id = self.llm.tokenizer.eos_token_id
        
        input_embeds, _ = self.audio_embedder(audio_paths) # [B, T_max, D_audio]
        if input_embeds.dim() != 3:
            raise ValueError(f"Expected input_embeds to have 3 dimensions [B, T_max, D_audio], got {input_embeds.shape}")
 
        proj_embeds, _ = self.projector(input_embeds) # [B, S_max, D_llm]
        if proj_embeds.dim() != 3:
            raise ValueError(f"Expected proj_embeds to have 3 dimensions [B, S_max, D_llm], got {proj_embeds.shape}")

        if proj_embeds.shape[2] != self.llm_embedding_dim:
            raise ValueError(f"Expected D={self.llm_embedding_dim}, got {proj_embeds.shape[2]}")


        prompt_ids = self.llm.tokenizer(prompt, return_tensors="pt").input_ids
        formatted_batch = self.format_batch(proj_embeds, prompt_ids)

        outputs = self.llm.generate(
            inputs_embeds=formatted_batch["inputs_embeds"],
            attention_mask=formatted_batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            no_repeat_ngram_size = no_repeat_ngram_size, 
            repetition_penalty = repetition_penalty,
            pad_token_id = self.llm.tokenizer.pad_token_id,
            eos_token_id = self.llm.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        return self.llm.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

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

    def save(self, ckpt_path):
        self.projector.save(ckpt_path)
        self.llm.save(ckpt_path)

    def summary(self):
        aud_total, aud_trainable = self.audio_embedder.count_pars() if self.audio_embedder is not None else (0, 0)
        pro_total, pro_trainable = self.projector.count_pars() if self.projector is not None else (0, 0)
        llm_total, llm_trainable = self.llm.count_pars() if self.llm is not None else (0, 0)
        
        all_total = llm_total + pro_total + aud_total
        all_trainable = llm_trainable + pro_trainable + aud_trainable

        # -----------------------------
        # Logging
        # -----------------------------
        logger.info("=" * 100)
        logger.info("AudioLLM PARS               TOTAL |       TRAINABLE |          FROZEN")
        logger.info("=" * 100)
        logger.info(f"Audio           : {aud_total:>15,} | {aud_trainable:>15,} | {aud_total - aud_trainable:>15,}")
        logger.info(f"Projector       : {pro_total:>15,} | {pro_trainable:>15,} | {pro_total - pro_trainable:>15,}")
        logger.info(f"LLM             : {llm_total:>15,} | {llm_trainable:>15,} | {llm_total - llm_trainable:>15,}")
        logger.info("-" * 100)
        logger.info(f"TOTAL           : {all_total:>15,} | {all_trainable:>15,} | {all_total - all_trainable:>15,}")
        logger.info(f"Trainable %     : {100 * all_trainable / all_total:.2f}%")
        logger.info("=" * 100)
