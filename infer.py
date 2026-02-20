# infer.py

import torch
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.utils import get_device_dtype
from AudioLLM import AudioLLM

logger = logging.getLogger("infer")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transcribe and/or translate audio using AudioLLM (Hugging Face).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--audio_path", type=str, required=True, help="Audio file/s")
    parser.add_argument("--lang", type=str, required=True, help="Audio language")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    # parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    # Inference params
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No repeat ngram size (dangerous for ASR/STT, speech allow repetitions)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (good for ASR/STT, but bad for QA)")
    # Task params
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            # logging.FileHandler(Path(args.output_dir) / "infer.log" , mode='a', encoding='utf-8'),  # append mode
            logging.StreamHandler()  # and log to console
        ]
    )
    logging.getLogger("transformers.infer").setLevel(logging.WARNING)

    # --------------------------------------------------
    # Config file
    # --------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    logger.info(f"CONFIG FILE: {args.config}\n" + json.dumps(config, indent=2) + "\n")
    logger.info(f"CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    prompt = (f"Input:\n<extra_id_0>\nRepeat the above {args.lang} Input text:\n")
    prompt = (f"Input:\n<extra_id_0>\nTranslate the above {args.lang} Input text into French:\n")
    prompt = (f"Input:\n<extra_id_0>\nWhere are gathering the group of people?:\n")
    prompt = (f"Input:\n<extra_id_0>\nQuestion:\nWhere are people?:\nAnswer:\n")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    t = time.time()
    model = AudioLLM(config=config, device=device, dtype=dtype, is_infer=True)
    logger.debug(f"Loading model took {time.time() - t:.2f} sec")


    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    t = time.time()
    model.eval()
    with torch.no_grad():
        output = model.generate(
            audio_paths=[args.audio_path],
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            repetition_penalty = args.repetition_penalty, 
        )

    logger.debug(f"Generation took {time.time() - t:.2f} sec")
    logger.info(f"SRC: {args.audio_path}")
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"HYP: {output[0]}")
