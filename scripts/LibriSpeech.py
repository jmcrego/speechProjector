import argparse
from pathlib import Path
#from collections import defaultdict
from tqdm import tqdm
#import numpy as np
#from pydub import AudioSegment
#import soundfile as sf
#import subprocess
#import shutil
import json
#import soxr 
#import os
import sys
#from scipy.io.wavfile import write
from utils import duration #load_audio_ffmpeg, extract_fragments, 
from text_normalize import normalize_text


def main():
    parser = argparse.ArgumentParser(description="Extract LibriSpeech audio fragments and build JSONL.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--idir", type=str, default="/lustre/fsmisc/dataset/LibriSpeech", help="Input path")
    parser.add_argument("--odir", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets", help="Output path")
    args = parser.parse_args()

    base_path = Path(args.idir)
    out_path = Path(args.odir)

    json_file = out_path / f"LibriSpeech.jsonl"
    with json_file.open("w", encoding="utf-8") as f_json:

        data_sets = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"]

        n_in, n_out = 0, 0
        t_out = 0.0
        for data_set in data_sets:

            data_set_path = base_path / data_set            

            files = list(data_set_path.glob("**/*.trans.txt")) # find recursively all files in data_set_path ended by *.trans.txt

            for file in tqdm(files, desc=f"Processing {data_set}", unit=" file"):
                
                with file.open("r", encoding="utf-8") as f: 

                    for line in f:

                        n_in += 1
                        audio_stem, text = line.strip().split(" ", 1)
                        text = normalize_text(text)
                        audio_path = file.parent / f"{audio_stem}.flac"

                        if not audio_path.exists():
                            # print(f"Audio file {audio_path} not found, skipping", file=sys.stderr)
                            continue

                        d = duration(audio_path)
                        if d > 30.0:
                            # print(f"Audio file {audio_path} is too long ({duration(audio_path):.1f} sec), skipping", file=sys.stderr)
                            continue

                        f_json.write(
                            json.dumps({
                                "audio_file": str(audio_path),
                                "split": data_set,
                                "transcription": { "lang": "en", "text": text },
                            }, ensure_ascii=False) + "\n"
                        )
                        n_out += 1
                        t_out += d

            h = t_out // 3600
            m = (t_out % 3600) // 60
            s = t_out % 60
            print(f"Created {n_out} out of {n_in} audio segments ({100*n_out/n_in:.2f}%). Duration is {h}:{m}:{s:05.2f} ({t_out/n_out:.1f} secs/file)\n")


if __name__ == "__main__":
    main()
