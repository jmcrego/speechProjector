# Trainer.py

import os
import re
import glob
import json
import torch
import shutil
import random
import logging
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from Dataset import BatchedBucketSampler, collate_fn
from scripts.utils import compute_grad_norm

logger = logging.getLogger("Trainer")


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_dataset,
        eval_dataset=None,
        batch_size=4,
        max_steps=10000,
        max_epochs=10,
        save_best_n=3,
        eval_every=1000,
        log_every=50,
        accum_steps=1,
        output_dir="./output_dir",
        json_logger=None,
        seed=42,
        resume=False,
    ):
        
        # meta = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and not k.endswith('dataset') and not k == "model"}
        # logger.info(f"Initializing {meta}")        

        self.seed_everything(seed)

        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.save_best_n = save_best_n
        self.eval_every = eval_every
        self.log_every = log_every
        self.accum_steps = accum_steps
        self.output_dir = output_dir
        self.json_logger = json_logger
        os.makedirs(output_dir, exist_ok=True)

        self.tokenizer = self.model.tokenizer

        param = next(self.model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        logger.info(f"Model parameters are on device={self.device} with dtype={self.dtype}")

        # -----------------------
        # Sampler & DataLoader
        # -----------------------
        
        self.train_sampler = BatchedBucketSampler(train_dataset, batch_size=batch_size, shuffle=not train_dataset.is_cached)
        self.train_loader = DataLoader(train_dataset, batch_sampler=self.train_sampler, collate_fn=collate_fn)
        logger.info(f"Initialized Sampler and DataLoader for train with batch_size={batch_size} with {len(self.train_dataset)} samples")

        self.eval_sampler = BatchedBucketSampler(eval_dataset, batch_size=batch_size, shuffle=not train_dataset.is_cached)
        self.eval_loader = DataLoader(eval_dataset, batch_sampler=self.eval_sampler, collate_fn=collate_fn)
        logger.info(f"Initialized Sampler and DataLoader for eval with batch_size={batch_size} with {len(self.eval_dataset)} samples")

        if max_epochs:
            self.max_steps = min(self.max_steps, int(len(train_dataset) / (batch_size * accum_steps)))
            logger.info(f"max_steps set to {self.max_steps}")

        # -----------------------
        # Optimizer & Scheduler
        # -----------------------
        lr_proj= config['optim']['lr_proj']
        self.optimizer = torch.optim.AdamW([{"params": self.model.projector.parameters(), "lr": lr_proj}])
        logger.info(f"Initialized AdamW optimizer with lr_proj={lr_proj}")

        if resume:
            state = torch.load(config['projector']['path'].replace(".proj.pt",".optim.pt"))
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.step = state["step"]
            logger.info(f"Resume training from {config}, loaded optimizer/step={self.step}")
        else:
            self.step = 0

        self.batch = 0 # microbatch step
        self.epoch = 0 # number of epochs completed
        self.sample = 0 # number of samples processed
        self.start_time = datetime.now()

        warmup_steps = config['optim']['warmup_steps']
        # cosine scheduler with warmup
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=int(warmup_steps), num_training_steps=self.max_steps)
        logger.info(f"Initialized Cosine scheduler with warmup. {self.max_steps} steps, ({warmup_steps}) warmup steps)")


    # -----------------------------
    # Seed everything
    # -----------------------------
    @staticmethod
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -----------------------
    # Save checkpoint
    # -----------------------
    def save_checkpoint(self, step=None, prefix="checkpoint"):
        step_str = f"_step{step}" if step is not None else ""
        ckpt_path = os.path.join(self.output_dir, f"{prefix}{step_str}")

        # save model
        self.model.save(ckpt_path)

        # save optimizer
        state = {"optimizer_state_dict": self.optimizer.state_dict(), "step": self.step}
        torch.save(state, f"{ckpt_path}.optim.pt")
        logger.info(f"Saved optimizer to {ckpt_path}.optim.pt")

        # Save config file after updating lora path
        self.config['projector']['path'] = ckpt_path + ".proj.pt"
        with open(f"{ckpt_path}.config.json", "w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=4)
        logger.info(f"Saved config to {ckpt_path}.config.json")

        # remove older checkpoints, keep only top N
        remove_old_checkpoints(step, self.output_dir, prefix, self.save_best_n)

    # -----------------------
    # Training loop
    # -----------------------
    def train(self):
        logger.info("Start training")

        self.model.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_audio_norm = 0.0
        accum_text_norm = 0.0
        total_pads = 0
        total_samples = 0

        scaler = torch.amp.GradScaler()  # initialize GradScaler

        while self.max_steps and self.step < self.max_steps:
            self.epoch += 1

            for batch in self.train_loader:
                pt_paths = batch["pt_paths"] # list of paths to .pt files containing audio embeddings (tensor of shape [T', D])
                offsets = batch["offsets"] # list of (start, end) frame offsets for each sample in the batch (for slicing audio embeddings)
                target_ids = batch["target_ids"] # [B, L_max] torch.long token ids for ASR transcription (padded to seq_len)
                target_ids = target_ids.to(self.device)

                self.batch += 1
                self.sample += batch["target_ids"].size(0)
                total_pads += (target_ids == self.tokenizer.pad_token_id).sum().item() # Number of pad tokens in the batch (for logging pad)
                total_samples += batch["target_ids"].size(0) # Number of samples (for logging pad)

                # Forward pass
                # this with disables automatic mixed precision for everything inside that context.
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    # Pass input_embeds instead of input_ids to the model, along with target_ids and attention_mask for loss computation
                    outputs = self.model(target_ids, pt_paths, offsets) 
                    raw_loss = outputs["loss"]
                    loss = raw_loss / self.accum_steps                    
                    accum_loss += raw_loss.detach()

                    with torch.no_grad():
                        accum_audio_norm += outputs["audio_norm"].detach()
                        accum_text_norm += outputs["text_norm"].detach()

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation
                if self.batch % self.accum_steps == 0:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)

                    # --- Compute grad norms ---
                    proj_grad_norm = compute_grad_norm(self.model.projector.parameters())
                    lora_grad_norm = compute_grad_norm(self.model.llm.lora_parameters())

                    scale_val = getattr(self.model.projector, "scale", None)
                    if scale_val is not None and isinstance(scale_val, torch.Tensor):
                        scale_val = scale_val.item()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer step via scaler
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad() #jmcc

                    # Scheduler step
                    self.scheduler.step()

                    # Logging (log_every must be multiple of accum_steps)
                    self.step += 1 
                    if self.step % self.log_every == 0:
                        avg_loss = accum_loss / self.accum_steps
                        avg_audio_norm = accum_audio_norm / self.accum_steps
                        avg_text_norm = accum_text_norm / self.accum_steps
                        self.log_fn( #training log
                            avg_loss.item(),
                            audio_norm=avg_audio_norm.item(),
                            text_norm=avg_text_norm.item(),
                            scale=scale_val,
                            proj_grad_norm=proj_grad_norm.item(),
                            lora_grad_norm=lora_grad_norm.item(),
                            total_pads=total_pads,
                            total_samples=total_samples,
                        )
                        self.json_logger.log(
                            type="train", step=self.step, loss=avg_loss.item(), 
                            audio_norm=avg_audio_norm.item(), text_norm=avg_text_norm.item(),
                            scale=scale_val,
                            proj_grad_norm=proj_grad_norm.item(), lora_grad_norm=lora_grad_norm.item(),
                            total_pads=total_pads, total_samples=total_samples,
                        )

                        total_pads = 0
                        total_samples = 0

                    accum_loss = 0.0
                    accum_audio_norm = 0.0
                    accum_text_norm = 0.0

                    # Evaluation + checkpoint
                    if self.eval_loader is not None and self.step % self.eval_every == 0:
                        self.evaluate(
                            max_new_tokens=256,
                            temperature=0.0,
                            top_p=1.0,
                            no_repeat_ngram_size = 0,
                            repetition_penalty = 1.1,
                        )
                        self.save_checkpoint(self.step)

                    if self.max_steps and self.step >= self.max_steps:
                        logger.info(f"Reached max steps {self.max_steps}, stopping training after "
                              f"{self.sample} samples, "
                              f"{self.step} steps, "
                              f"{self.batch} batches, "
                              f"{self.sample/len(self.train_dataset):.3f} epochs.")
                        break

            if self.max_epochs and self.epoch >= self.max_epochs:
                logger.info(f"Reached max epochs {self.max_epochs}, stopping training after "
                      f"{self.sample} samples, "
                      f"{self.step} steps, "
                      f"{self.batch} batches, "
                      f"{self.sample/len(self.train_dataset):.3f} epochs.")
                break

        logger.info("End training")

    # -----------------------
    # Evaluation
    # -----------------------
    @torch.no_grad()
    def evaluate(
        self,
    ):
        """
        Evaluation with:
        1) standard forward loss
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        
        for batch in self.eval_loader:
            # ----------------------------
            # Move tensors to device
            # ----------------------------
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # ----------------------------
            # Forward pass (loss)
            # ----------------------------
            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.dtype,
                enabled=(self.device.type == "cuda"),
            ):
                outputs = self.model(**batch)

            loss = outputs["loss"].item()
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # valid log
        self.log_fn(avg_loss, is_eval=True) 
        self.json_logger.log(type="eval", step=self.step, loss=avg_loss)                        

        self.model.train()
        return avg_loss

    # -----------------------
    # Logging 
    # -----------------------
    def log_fn(self, loss, audio_norm=None, text_norm=None, scale=None, proj_grad_norm=None, lora_grad_norm=None, is_eval=False, bleu=None, wer=None, cer=None, total_pads=0, total_samples=0, acc=None):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        log_str =  f"{'VAL ' if is_eval else 'TRN'} | "
        log_str += f"step={self.step:0>6d}/{self.max_steps} | "
        log_str += f"epoch={self.sample/len(self.train_dataset):.3f}/{self.max_epochs} | "
        log_str += f"loss={loss:.3f} | "
        log_str += f"lr_proj={self.optimizer.param_groups[0]['lr']:.3e} | "
        log_str += f"lr_lora={self.optimizer.param_groups[1]['lr']:.3e} | "

        if proj_grad_norm is not None:
            log_str += f"‖proj_grad‖={proj_grad_norm:.2f} | "
        if lora_grad_norm is not None:
            log_str += f"‖lora_grad‖={lora_grad_norm:.2f} | "
        if scale is not None:
            log_str += f"scale={scale:.2f} | "
        if audio_norm is not None:
            log_str += f"‖audio‖={audio_norm:.2f} | "
        if text_norm is not None:
            log_str += f"‖text‖={text_norm:.2f} | "
        if total_samples:
            log_str += f"pads_per_sample={total_pads/total_samples:.2f} | "
        
        log_str += f"elapsed={h:02d}h:{m:02d}m:{s:02d}s"
        logger.info(log_str)


    def read_cache_embs(self, pt_paths, offsets):
        """
        Reads the batch embeddings cached in disk as indicated by pt_paths and offsets. 
        Saves the last buckets in an LRU cache to speed up access for samples in the same bucket.
        Args:
            pt_paths (List[str]): bucket filenames
            offsets (List[int] or Tensor): index inside each bucket

        Returns:
            audio_embs: Tensor [B, T, D] (on CPU)
        """

        # Simple in-memory cache to avoid redundant disk reads for samples in the same bucket (pt_path)
        if not hasattr(self, "_bucket_cache"):
            self._bucket_cache = OrderedDict()  # pt_path → bucket dict with "audio_embs" key
            self._buffer_size = 2  # max number of buckets to keep in memory

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
                if len(self._bucket_cache) >= self._buffer_size:
                    self._bucket_cache.popitem(last=False)

            self._bucket_cache[pt_path] = bucket
            bucket_embs = bucket["audio_embs"]  # [B_bucket, T, D]

            # ---- Extract needed embeddings ----
            for batch_idx, offset in items:
                batch_embs[batch_idx] = bucket_embs[offset]

        # Stack in original batch order
        audio_embs = torch.stack(batch_embs, dim=0)

        return audio_embs


# x: [B, T, D] embeddings (B=batch size, T=time steps, D=embedding dim)
def batch_embedding_norm(x, mask=None):
    # Compute norm per vector along last dim
    norm = torch.norm(x, dim=-1)  # [B, T]

    if mask is not None:
        norm = norm * mask.float()  # zero out masked positions

    # Return average over batch and sequence
    return norm.sum() / (mask.sum() if mask is not None else x.numel() / x.size(-1))


def remove_old_checkpoints(step, output_dir, prefix, save_best_n):
    if step is None:
        return

    #Ex: checkpoint_step20000.proj.pt
    existing_steps = []
    for fname in os.listdir(output_dir):
        if fname.startswith(prefix) and fname.endswith(".proj.pt"):
            m = re.search(r"_step(\d+).proj.pt", fname)
            if m:
                existing_steps.append(int(m.group(1)))

    for old_step in sorted(existing_steps, reverse=True)[save_best_n:]:
        old_ckpt_path = os.path.join(output_dir, f"{prefix}_step{old_step}")

        try:
            for path in glob.glob(f"{old_ckpt_path}.*"):
                logger.info(f"Removing {path}.*")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        except Exception as e:
            print(f"Error removing old checkpoint {old_ckpt_path}: {e}")


