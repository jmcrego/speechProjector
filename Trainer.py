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
from collections import OrderedDict, defaultdict

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
        
        self.train_sampler = BatchedBucketSampler(train_dataset, batch_size=batch_size, shuffle=True)
        self.train_loader = DataLoader(train_dataset, batch_sampler=self.train_sampler, collate_fn=collate_fn)
        logger.info(f"Initialized Sampler and DataLoader for train with batch_size={batch_size} with {len(self.train_dataset)} samples")

        self.eval_sampler = BatchedBucketSampler(eval_dataset, batch_size=batch_size, shuffle=False)
        self.eval_loader = DataLoader(eval_dataset, batch_sampler=self.eval_sampler, collate_fn=collate_fn)
        logger.info(f"Initialized Sampler and DataLoader for eval with batch_size={batch_size} with {len(self.eval_dataset)} samples")

        if max_epochs:
            self.max_steps = min(self.max_steps, int(len(train_dataset) / (batch_size * accum_steps)))
            logger.info(f"converted max_epochs={max_epochs} to max_steps={self.max_steps}")

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

        # Save config file after updating path
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

        accum = defaultdict(float)

        # total_pads = 0
        # total_samples = 0

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

                # total_pads += (target_ids == self.tokenizer.pad_token_id).sum().item() # Number of pad tokens in the batch (for logging pad)
                # total_samples += batch["target_ids"].size(0) # Number of samples (for logging pad)

                # Forward pass
                # this with disables automatic mixed precision for everything inside that context.
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    # Pass input_embeds instead of input_ids to the model, along with target_ids and attention_mask for loss computation
                    outputs = self.model(target_ids, pt_paths, offsets) 
                    raw_loss = outputs["loss"]
                    loss = raw_loss / self.accum_steps                    

                    accum['loss'] += raw_loss.item()
                    accum['loss_cos'] += outputs["loss_cos"].item()
                    accum['loss_ce'] += outputs["loss_ce"].item()
                    accum['loss_scale'] += outputs["loss_scale"].item()
                    accum['loss_mse_txt'] += outputs["loss_mse_txt"].item()
                    accum['loss_mse_pad'] += outputs["loss_mse_pad"].item()
                    accum['audio_norm'] += outputs["audio_norm"].item()
                    accum['text_norm'] += outputs["text_norm"].item()
                    accum['n_pads'] += (target_ids == self.tokenizer.pad_token_id).sum().item()
                    accum['n_samples'] += target_ids.size(0)
                    accum['n_batchs'] += 1
    
                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation
                if self.batch % self.accum_steps == 0:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)

                    # --- Compute grad norms ---
                    proj_grad_norm = compute_grad_norm(self.model.projector.parameters())
                    accum['proj_grad_norm'] = proj_grad_norm.item()

                    scale_val = getattr(self.model.projector, "scale", None)
                    if scale_val is not None and isinstance(scale_val, torch.Tensor):
                        accum['scale'] = scale_val.item()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer step via scaler
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad() # reset gradients

                    # Scheduler step
                    self.scheduler.step() # update learning rate

                    # Logging (log_every must be multiple of accum_steps)
                    self.step += 1 
                    if self.step % self.log_every == 0:
                        self.log_fn(accum, is_eval=False)
                        accum = defaultdict(float)

                    # Evaluation + checkpoint
                    if self.step % self.eval_every == 0:
                        if self.eval_loader is not None and len(self.eval_loader) > 0:
                            self.evaluate()
                        self.save_checkpoint(self.step)

                    if self.max_steps and self.step >= self.max_steps:
                        logger.info(f"Reached max steps {self.max_steps}, stopping training after "
                              f"{self.sample} samples, "
                              f"{self.step} steps, "
                              f"{self.batch} batches, "
                              f"{self.sample/len(self.train_dataset):.3f} epochs.")
                        break

            # if self.max_epochs and self.epoch >= self.max_epochs:
            #     logger.info(f"Reached max epochs {self.max_epochs}, stopping training after "
            #           f"{self.sample} samples, "
            #           f"{self.step} steps, "
            #           f"{self.batch} batches, "
            #           f"{self.sample/len(self.train_dataset):.3f} epochs.")
            #     break

        logger.info("End training")

    # -----------------------
    # Evaluation
    # -----------------------
    @torch.no_grad()
    def evaluate(
        self,
    ):
        """
        Evaluation with standard forward loss
        """
        self.model.eval()

        accum = defaultdict(float)

        for n, batch in enumerate(self.eval_loader, start=1):
            pt_paths = batch["pt_paths"] # list of paths to .pt files containing audio embeddings (tensor of shape [T', D])
            offsets = batch["offsets"] # list of (start, end) frame offsets for each sample in the batch (for slicing audio embeddings)
            target_ids = batch["target_ids"] # [B, L_max] torch.long token ids for ASR transcription (padded to seq_len)
            target_ids = target_ids.to(self.device)

            # Forward pass
            # this with disables automatic mixed precision for everything inside that context.
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == "cuda")):
                # Pass input_embeds instead of input_ids to the model, along with target_ids and attention_mask for loss computation
                outputs = self.model(target_ids, pt_paths, offsets) 

            accum['loss'] += outputs["loss"].item()
            accum['loss_cos'] += outputs["loss_cos"].item()
            accum['loss_ce'] += outputs["loss_ce"].item()
            accum['loss_scale'] += outputs["loss_scale"].item()
            accum['loss_mse_txt'] += outputs["loss_mse_txt"].item()
            accum['loss_mse_pad'] += outputs["loss_mse_pad"].item()
            accum['audio_norm'] += outputs["audio_norm"].item()
            accum['text_norm'] += outputs["text_norm"].item()
            accum['n_pads'] += (target_ids == self.tokenizer.pad_token_id).sum().item()
            accum['n_samples'] += target_ids.size(0)
            accum['n_batchs'] += 1


        # valid log
        self.log_fn(accum, is_eval=True)

        self.model.train()


    # -----------------------
    # Logging 
    # -----------------------
    def log_fn(self, accum, is_eval=False):

        elapsed = (datetime.now() - self.start_time).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        loss = accum['loss'] / max(1, accum['n_batchs'])
        audio_norm = accum['audio_norm'] / max(1, accum['n_batchs'])
        text_norm = accum['text_norm'] / max(1, accum['n_batchs'])
        loss_cos = accum['loss_cos'] / max(1, accum['n_batchs'])
        loss_ce = accum['loss_ce'] / max(1, accum['n_batchs'])
        loss_scale = accum['loss_scale'] / max(1, accum['n_batchs'])
        loss_mse_txt = accum['loss_mse_txt'] / max(1, accum['n_batchs'])
        loss_mse_pad = accum['loss_mse_pad'] / max(1, accum['n_batchs'])
        scale_val = accum.get('scale', None)
        proj_grad_norm = accum.get('proj_grad_norm', None)
        total_pads = accum['n_pads']
        total_samples = accum['n_samples']

        log_str =  f"{'VAL ' if is_eval else 'TRN'} | "
        log_str += f"step={self.step:0>6d}/{self.max_steps} | "
        # log_str += f"epoch={self.sample/len(self.train_dataset):.3f}/{self.max_epochs} | "
        log_str += f"epoch={self.sample/len(self.train_dataset):.3f} | "
        log_str += f"loss={loss:.4f} | "
        log_str += f"ℒ_cos={loss_cos:.4f} | " if loss_cos is not None else ""
        log_str += f"ℒ_ce={loss_ce:.4f} | " if loss_ce is not None else ""
        log_str += f"ℒ_scale={loss_scale:.4f} | " if loss_scale is not None else ""
        log_str += f"ℒ_mse_txt={loss_mse_txt:.4f} | " if loss_mse_txt is not None else ""
        log_str += f"ℒ_mse_pad={loss_mse_pad:.4f} | " if loss_mse_pad is not None else ""
        log_str += f"lr_proj={self.optimizer.param_groups[0]['lr']:.3e} | "

        if proj_grad_norm is not None:
            log_str += f"‖proj_grad‖={proj_grad_norm:.2f} | "
        if scale_val is not None:
            log_str += f"scale={scale_val:.2f} | "
        if audio_norm is not None:
            log_str += f"‖audio‖={audio_norm:.2f} | "
        if text_norm is not None:
            log_str += f"‖text‖={text_norm:.2f} | "
        if total_samples:
            log_str += f"pads_per_sample={total_pads/total_samples:.2f} | "
        
        log_str += f"elapsed={h:02d}:{m:02d}:{s:02d}"
        logger.info(log_str)

        if self.json_logger is not None:
            self.json_logger.log(
                split="eval" if is_eval else "train", 
                step=self.step, 
                loss=loss, 
                audio_norm=audio_norm, 
                text_norm=text_norm,
                loss_cos=loss_cos,
                loss_ce=loss_ce,
                loss_scale=loss_scale,
                loss_mse_txt=loss_mse_txt,
                loss_mse_pad=loss_mse_pad,
                proj_grad_norm=proj_grad_norm,
                scale=scale_val,
                lr_proj=self.optimizer.param_groups[0]['lr'],
                pads_per_sample=(total_pads/total_samples) if total_samples else None,
                elapsed=f"{h:02d}:{m:02d}:{s:02d}",
            )



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


