# Trainer.py

import os
import re
import glob
import json
import torch
import shutil
import random
import logging
import sacrebleu
import jiwer
import unicodedata
import numpy as np
from datetime import datetime
import time

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from Dataset import BatchedLengthSampler

logger = logging.getLogger("Trainer")

class UnicodeNormalize(jiwer.AbstractTransform):
    def process_string(self, s: str):
        return unicodedata.normalize("NFKC", s)

class RemoveTags(jiwer.AbstractTransform):
    def process_string(self, s: str):
        # handles nested brackets, empty tags                                                                                                                                                                                                                                    
        s = re.sub(r"\<[^>]*\>", "", s)  # Remove <anything>                                                                                                                                                                                                                                  
        s = re.sub(r"\[[^\]]*\]", "", s)  # Remove [anything]                                                                                                                                                                                                                                 
        return s
    
class NormalizeApostrophes(jiwer.AbstractTransform):
    def process_string(self, s: str):
        return re.sub(r"[’']", " ", s) # Handle straight and curly apostrophes (do not delete the space as it does RemovePunctuation)

transform = jiwer.Compose([ 
    UnicodeNormalize(), 
    RemoveTags(), 
    jiwer.ToLowerCase(), 
    NormalizeApostrophes(),
    jiwer.RemovePunctuation(), 
    jiwer.RemoveWhiteSpace(replace_by_space=True), 
    jiwer.Strip(), 
    jiwer.RemoveEmptyStrings() 
])

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

class Trainer:
    def __init__(
        self,
        config,
        model,
        train_dataset,
        eval_dataset=None,
        batch_size=4,
        lr_proj=5e-4,
        lr_lora=1e-4,
        max_steps=10000,
        max_epochs=10,
        warmup_steps=0,
        save_best_n=3,
        eval_every=1000,
        log_every=50,
        accum_steps=1,
        output_dir="./output_dir",
        json_logger=None,
        seed=42,
        resume=False,
    ):
        
        meta = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and not k.endswith('dataset') and not k == "model"}
        logger.info(f"Initializing {meta}")        

        self.seed_everything(seed)

        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.lr_proj = lr_proj
        self.lr_lora = lr_lora
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.save_best_n = save_best_n
        self.eval_every = eval_every
        self.log_every = log_every
        self.accum_steps = accum_steps
        self.output_dir = output_dir
        self.json_logger = json_logger
        self.tokenizer = self.model.llm.tokenizer
        os.makedirs(output_dir, exist_ok=True)


        param = next(self.model.llm.model.parameters())
        self.device = param.device
        self.dtype = param.dtype

        train_batch_size = batch_size
        eval_batch_size = 1

        # -----------------------
        # Sampler & DataLoader
        # -----------------------
        
        self.train_sampler = BatchedLengthSampler(train_dataset, batch_size=train_batch_size, shuffle=not train_dataset.is_cached)
        self.train_loader = DataLoader(train_dataset, batch_sampler=self.train_sampler, collate_fn=self.collate_fn)
        logger.info(f"Initialized Sampler and DataLoader for train with batch_size={train_batch_size} with {len(self.train_dataset)} samples")

        self.eval_sampler = BatchedLengthSampler(eval_dataset, batch_size=eval_batch_size, shuffle=not train_dataset.is_cached)
        self.eval_loader = DataLoader(eval_dataset, batch_sampler=self.eval_sampler, collate_fn=self.collate_fn)
        logger.info(f"Initialized Sampler and DataLoader for eval with batch_size={eval_batch_size} with {len(self.eval_dataset)} samples")

        if max_epochs:
            self.max_steps = min(self.max_steps, int(len(train_dataset) / (batch_size * accum_steps)))
            logger.info(f"max_steps set to {self.max_steps}")

        # -----------------------
        # Optimizer & Scheduler
        # -----------------------

        self.optimizer = torch.optim.AdamW([
            {"params": self.model.projector.parameters(), "lr": lr_proj},
            {"params": [p for n, p in self.model.llm.model.named_parameters() if p.requires_grad], "lr": lr_lora},
        ])
        logger.info(f"Initialized AdamW optimizer with lr_proj={lr_proj} lr_lora={lr_lora}")

        if resume:
            state = torch.load(config['projector']['path'].replace(".proj.pt",".optim.pt"))
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.step = state["step"]
            logger.info(f"Resume training from {config}, loaded optimizer/step={self.step}")
        else:
            self.step = 0

        self.batch = 0 # microbatch step
        self.epoch = 0
        self.sample = 0
        self.start_time = datetime.now()

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.warmup_steps), num_training_steps=self.max_steps)
        logger.info(f"Initialized Linear scheduler with warmup. {self.max_steps} steps, ({self.warmup_steps}) warmup steps)")


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
        self.config['lora']['path'] = ckpt_path + ".lora"
        self.config['embeddings']['path'] = ckpt_path + ".embs.pt"
        with open(f"{ckpt_path}.config.json", "w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=4)
        logger.info(f"Saved config to {ckpt_path}.config.json")

        # remove older checkpoints, keep only top N
        remove_old_checkpoints(step, self.output_dir, prefix, self.save_best_n)

    # -----------------------
    # Collator function
    # -----------------------
    def collate_fn(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        audio_paths = [x["audio_path"] for x in batch]
        def ensure_tensor(x):
            return x.detach().clone() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
        prompt_ids = pad_sequence([ensure_tensor(x["prompt_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        target_ids = pad_sequence([ensure_tensor(x["target_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)

        has_audio_cache = "pt_path" in batch[0] and "offset" in batch[0]
        if has_audio_cache:
            pt_paths = [x["pt_path"] for x in batch]
            offsets = torch.tensor([x["offset"] for x in batch], dtype=torch.long)
        else:
            pt_paths = None
            offsets = None

        return {
            "audio_paths": audio_paths,
            "pt_paths": pt_paths,         # List[str]
            "offsets": offsets,           # (B,)
            "prompt_ids": prompt_ids,
            "target_ids": target_ids
        }

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
                self.batch += 1
                self.sample += batch["prompt_ids"].size(0)
                # Move tensors to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Number of pad tokens in the batch (for logging pad)
                total_pads += (batch["prompt_ids"] == self.tokenizer.pad_token_id).sum().item() + (batch["target_ids"] == self.tokenizer.pad_token_id).sum().item()
                # Number of samples (for logging pad)
                total_samples += batch["prompt_ids"].size(0)

                # Forward pass
                # this with disables automatic mixed precision for everything inside that context.
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    outputs = self.model(**batch)

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
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        no_repeat_ngram_size = 0,
        repetition_penalty = 1.1,
    ):
        """
        Evaluation with:
        1) standard forward loss
        2) autoregressive generation for qualitative inspection
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        logged_samples = 0

        predictions = []
        references = []
        
        tic = time.time()

        for batch in self.eval_loader:
            # ----------------------------
            # Move tensors to device
            # ----------------------------
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # ----------------------------
            # 1) Forward pass (loss)
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

            # ----------------------------
            # 2) Generation
            # ----------------------------
            audio_paths = batch["audio_paths"]
            prompt_ids = batch["prompt_ids"]
            target_ids = batch["target_ids"]

            # Run generation
            gen_texts = self.model.generate(
                audio_paths=audio_paths,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size = no_repeat_ngram_size,
                repetition_penalty = repetition_penalty,
            )

            # Decode prompt text (for logging only)
            prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)
            # Decode targets (ground truth)
            target_texts = self.tokenizer.batch_decode(target_ids, skip_special_tokens=False)

            predictions.extend(gen_texts)
            references.extend(target_texts)

            for i in range(len(audio_paths)):
                logger.info(f"[Eval sample {logged_samples}]")
                logger.info(f"AUDIO: {audio_paths[i]}")
                logger.info(f"PROMPT: {prompt_texts[i].replace("\n","↵")}")
                logger.info(f"TARGET: {target_texts[i].replace("\n","↵")}")
                logger.info(f"PREDIC: {gen_texts[i].replace("\n","↵")}")
                logger.info("=" * 80)

                logged_samples += 1

        bleu_score, wer_score, cer_score, acc = eval_test_set(references, predictions, self.tokenizer.eos_token)
        avg_loss = total_loss / max(1, n_batches)

        # valid log
        self.log_fn(avg_loss, is_eval=True, bleu=bleu_score, wer=wer_score, cer=cer_score, acc=acc) 
        self.json_logger.log(type="eval", step=self.step, loss=avg_loss, bleu=bleu_score, wer=wer_score, cer=cer_score, acc=acc)                        

        logger.info(f"Generation took {time.time()-tic:.2f}s for {len(predictions)} samples")

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
        if bleu is not None:
            log_str += f"bleu={bleu:.2f} | "
        if wer is not None:
            log_str += f"wer={wer:.2f} | "
        if cer is not None:
            log_str += f"cer={cer:.2f} | "
        if acc is not None:
            log_str += f"acc={acc:.2f} | " #lang tag accuracy
        if total_samples:
            log_str += f"pads_per_sample={total_pads/total_samples:.2f} | "
        
        log_str += f"elapsed={h:02d}h:{m:02d}m:{s:02d}s"
        logger.info(log_str)


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


def eval_test_set(references, predictions, eos_token):
    # get initial lang tag
    hyp_lang = [re.match(r"^(<[^>]+>)", x).group(1) if re.match(r"^(<[^>]+>)", x) else None for x in predictions]
    ref_lang = [re.match(r"^(<[^>]+>)", x).group(1) if re.match(r"^(<[^>]+>)", x) else None for x in references]

    # remove ending </s>
    predictions = [x.replace(eos_token, "").strip() for x in predictions]
    references = [x.replace(eos_token, "").strip() for x in references]

    # remove initial lang tag
    predictions = [re.sub(r"^<[^>]+>\s*", "", x) for x in predictions]
    references = [re.sub(r"^<[^>]+>\s*", "", x) for x in references]

    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score

    # Pre-transform both
    # refs_transformed = transform(references)
    # hyps_transformed = transform(predictions)
    refs_transformed = [ transform(x) or "EMPTY" for x in references]
    hyps_transformed = [ transform(x) or "EMPTY" for x in predictions]

    # def transform_one(x):
    #     out = transform([x])
    #     return out[0] if out else "EMPTY"

    # refs_transformed = [ transform_one(x) for x in references ]
    # hyps_transformed = [ transform_one(x) for x in predictions ]

    if len(refs_transformed) != len(hyps_transformed):
        logger.info(f"Reference / hypothesis length mismatch after transform {len(refs_transformed)} != {len(hyps_transformed)}")
        return bleu_score, 0., 0., 0.


    # Word-level metrics                                                                                                                                                                                                                                                                          
    word_output = jiwer.process_words(refs_transformed, hyps_transformed)
    logger.info("\n" + jiwer.visualize_alignment(word_output, show_measures=True))
    logger.info(f"WER: {word_output.wer:.4f}")

    # Character-level metrics                                                                                                                                                                                                                                                                     
    char_output = jiwer.process_characters(refs_transformed, hyps_transformed)
    logger.info(f"CER: {char_output.cer:.4f}")

    lang_acc = evaluate_lang_tags(hyp_lang, ref_lang)

    return bleu_score, 100*word_output.wer, 100*char_output.cer, lang_acc


def evaluate_lang_tags(hyp_lang, ref_lang):
    """
    Evaluate language tag predictions
    
    Args:
        hyp_lang: List of predicted language tags (can contain None)
        ref_lang: List of reference language tags (can contain None)
    """
    # Filter out None values
    valid_pairs = [(h, r) for h, r in zip(hyp_lang, ref_lang) if h is not None and r is not None]
    
    if not valid_pairs:
        logger.info("=" * 70)
        logger.info("LANGUAGE TAG EVALUATION")
        logger.info("=" * 70)
        logger.info(f"Total samples: {len(hyp_lang)}")
        logger.info(f"Missing reference tags: {sum(1 for x in ref_lang if x is None)}")
        logger.info(f"Missing hypothesis tags: {sum(1 for x in hyp_lang if x is None)}")
        logger.info("ERROR: No valid language tag pairs found")
        logger.info("=" * 70)
        return
    
    hyp_valid = [h for h, r in valid_pairs]
    ref_valid = [r for h, r in valid_pairs]
    
    # Calculate accuracy
    accuracy = accuracy_score(ref_valid, hyp_valid)
    
    # Get unique labels
    labels = sorted(list(set(ref_valid + hyp_valid)))
    
    # Confusion matrix
    cm = confusion_matrix(ref_valid, hyp_valid, labels=labels)
    
    # Print report
    logger.info("=" * 70)
    logger.info("LANGUAGE TAG EVALUATION")
    logger.info("=" * 70)
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total samples: {len(hyp_lang)}")
    logger.info(f"  Valid pairs: {len(valid_pairs)}")
    logger.info(f"  Missing reference tags: {sum(1 for x in ref_lang if x is None)}")
    logger.info(f"  Missing hypothesis tags: {sum(1 for x in hyp_lang if x is None)}")
    
    logger.info(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    logger.info(f"\nDetailed Classification Report:")
    logger.info("\n" + classification_report(ref_valid, hyp_valid, labels=labels, zero_division=0))
    
    logger.info("Confusion Matrix:")
    header = "Ref \ Pred" + "".join(f"{label:>10}" for label in labels)
    logger.info(header)
    logger.info("-" * (10 + 10 * len(labels)))
    for i, label in enumerate(labels):
        row = f"{label:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        logger.info(row)
    logger.info("=" * 70)
    return accuracy



