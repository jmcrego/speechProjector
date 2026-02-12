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
        description="Transcribe and/or translate audio using AudioToLLM (Hugging Face).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--audio_path", type=str, required=True, help="Audio file/s")
    parser.add_argument("--slang", type=str, required=True, help="Audio language")
    parser.add_argument("--tlang", type=str, required=True, help="Target language")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    # parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    # Inference params
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No repeat ngram size")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    # Task params
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    prompt = (f'<|im_start|>system'
        '<|im_end|>'
        '<|im_start|>user'
        f'Translate the following {args.slang} speech utterance into {args.tlang}:'
        f'{args.slang}: <extra_id_0>'
        f'{args.tlang}: <|im_end|>'
        '<|im_start|>assistant'
    )

    # Create output directory if needed
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # json_logger = JSONMetricsLogger(Path(args.output_dir) / "metrics.jsonl")

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
    logger.info(f"Config: {config}")

    logger.info("=" * 80)
    logger.info(f"Starting new run @ {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"CONFIG FILE: {args.config}\n" + json.dumps(config, indent=2) + "\n")
    logger.info("=" * 80)

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    t = time.time()
    model = AudioLLM(
        config=config,
        device=device,
        dtype=dtype,
        is_infer=True, 
    )
    model.eval()
    logger.info(f"Loading model took {time.time() - t:.2f} sec")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    t = time.time()

    with torch.no_grad():
        output = model.generate(
            audio_paths=[args.audio_path], 
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            no_repeat_ngram_size = args.no_repeat_ngram_size, #dangerous for ASR/STT, speech allow repetitions
            repetition_penalty = args.repetition_penalty, #good for ASR/STT, but bad for QA
        )

        logger.info(f"AUDIO: {args.audio_path}")
        def replace_CR(text):
            return text.replace("\n", "↵") if text is not None else None
        logger.info(f"PREDIC: {replace_CR(output[0])}")

    logger.info(f"Generation took {time.time() - t:.2f} sec")
