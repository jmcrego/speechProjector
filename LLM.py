# LLM.py

import torch
import logging
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, PeftModel, LoraConfig
import gc

logger = logging.getLogger("LLM")

class LLM(torch.nn.Module):
    def __init__(self, config, config_lora, load_only_embedding_layer=False):
        """
        Wrapper for the base LLM
        """
        super().__init__()

        llm_path = config['path']

        # -------------------------
        # Tokenizer
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            raise ValueError(
                "Tokenizer does not define pad_token. "
                "PAD token is required for alignment training."
            )
        logger.info(f"Loaded Tokenizer from {llm_path}")
        logger.info(f"bos_token = {self.tokenizer.bos_token} {self.tokenizer.bos_token_id}")
        logger.info(f"eos_token = {self.tokenizer.eos_token} {self.tokenizer.eos_token_id}")
        logger.info(f"pad_token = {self.tokenizer.pad_token} {self.tokenizer.pad_token_id}")

        # =====================================================
        # MODE 1 — Embedding-only mode
        # =====================================================
        if load_only_embedding_layer:

            # Load BASE model only (no LM head) on CPU
            tmp_model = AutoModel.from_pretrained(llm_path, low_cpu_mem_usage=True, device_map="cpu")
            embedding_weight = (tmp_model.get_input_embeddings().weight.detach().clone())
            self.embedder = torch.nn.Embedding.from_pretrained(embedding_weight, freeze=True)

            # Clean up full model
            del tmp_model
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Loaded embedding layer only (LLM not kept in memory).")

        # =====================================================
        # MODE 2 — Full LLM (+ optional LoRA)
        # =====================================================
        else:

            # Load full causal LM on CPU
            self.model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True, device_map="cpu")
            self.embedder = self.model.get_input_embeddings() # reference to the embedding layer for alignment losses
            logger.info("Loaded full LLM on CPU.")

            # Optional LoRA
            if config_lora is not None:

                lora_path = config_lora.get("path", None)
                if lora_path is not None:
                    # Load existing adapters
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    logger.info(f"Loaded LoRA adapters from {lora_path}")

                else:
                    # Create new adapters
                    lora_cfg = LoraConfig(
                        r=config_lora["r"],
                        lora_alpha=config_lora["lora_alpha"],
                        target_modules=config_lora["target_modules"],
                        lora_dropout=config_lora["lora_dropout"],
                        bias=config_lora["bias"],
                        task_type=config_lora["task_type"],
                    )

                    self.model = get_peft_model(self.model, lora_cfg)
                    logger.info("Initialized new LoRA adapters.")


    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, *args, **kwargs):
        if hasattr(self, "model"):
            return self.model(*args, **kwargs)
        else:
            raise RuntimeError(
                "LLM was loaded with load_only_embedding_layer=True. "
                "Full forward pass is unavailable."
            )

    def generate(self, *args, **kwargs):
        if hasattr(self, "model"):
            return self.model.generate(*args, **kwargs)
        else:
            raise RuntimeError(
                "LLM was loaded with load_only_embedding_layer=True. "
                "Generation is unavailable."
            )

    # -------------------------------------------------
    # Freezing logic
    # -------------------------------------------------
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False # all LLM parameters are frozen
        self.eval()
        logger.info("LLM (LoRA) frozen (eval mode)")

    def unfreeze_lora(self):
        for n, p in self.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True # LoRA adapters are trainable
            else:
                p.requires_grad = False # rest of LLM parameters are frozen
        self.train()
        logger.info("LLM (LoRA) unfrozen (train mode)")

    # -------------------------------------------------
    # Save LoRA adapters
    # -------------------------------------------------
    def save(self, ckpt_path):
        # save lora parameters if they exist
        if hasattr(self, "model") and isinstance(self.model, PeftModel):
            self.model.save_pretrained(ckpt_path + ".lora")
            logger.info(f"Saved LoRA adapters to {ckpt_path}.lora")

    def count_pars(self):
        if hasattr(self, "model"):
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            total = sum(p.numel() for p in self.embedder.parameters())
            trainable = sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)
        return total, trainable

    # @property
    # def embedding_dim(self):
    #     if hasattr(self, "model"):
    #         return self.model.config.hidden_size
    #     else:
    #         return self.embedder.embedding_dim
        
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    parser = argparse.ArgumentParser(description="Instantiate LLM backbone")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    # Load JSON config
    with open(args.config, "r") as f:
        config = json.load(f)

    llm = LLM(config['llm'], config['lora'])
    logger.info("LLM successfully initialized")