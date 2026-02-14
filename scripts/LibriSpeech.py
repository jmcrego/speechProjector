import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import subprocess
import numpy as np
import shutil
import json
import soxr 
import os
import sys
from scipy.io.wavfile import write
from utils import load_audio_ffmpeg, extract_fragments


def build_segments_dict(segments_path, source_path, target_path):
    """Read segments, source, target files and group by audio_name."""
    segments_dict = defaultdict(list)

    with segments_path.open("r", encoding="utf-8") as f_seg, \
         source_path.open("r", encoding="utf-8") as f_src, \
         target_path.open("r", encoding="utf-8") as f_tgt:

        n_segments = 0
        for seg, src, tgt in zip(f_seg, f_src, f_tgt):
            _, audio_name, beg, end = seg.strip().split(" ")
            segments_dict[audio_name].append({
                "beg": float(beg),
                "end": float(end),
                "src": src.strip(),
                "tgt": tgt.strip()
            })
            n_segments += 1

        print(f"Found {n_segments} segments in {len(segments_dict)} audio files")
        
    return segments_dict


def get_audio_dict(base_path):
    print(base_path)
    flac_stem2path = {}
    for audio_name in base_path.glob("*.flac"):
        audio_stem = Path(audio_name).stem
        if audio_stem in flac_stem2path:
            print(f"repeated entry {audio_stem}")
        flac_stem2path[audio_stem] = base_path / audio_name
    print(f"Found {len(set(flac_stem2path.keys()))} flac files")

    return flac_stem2path


def main():
    parser = argparse.ArgumentParser(description="Extract MultilingualTEDx audio fragments and build TSV.")
    parser.add_argument("--idir", type=str, default="/lustre/fsmisc/dataset/LibriSpeech", help="Input path")
    parser.add_argument("--odir", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets", help="Output path")
    args = parser.parse_args()

    base_path = Path(args.idir)
    out_path = Path(args.odir)

    din = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"]
    
    lang_pairs = {tuple(p.name.split("-")) for p in base_path.iterdir() if p.is_dir() and len(p.name.split("-")) == 2 and all(len(x) == 2 for x in p.name.split("-"))}
    data_sets = ["valid", "test", "train"]

    # tsv_file = out_path / f"MultilingualTEDx.tsv"
    # with tsv_file.open("w", encoding="utf-8") as f_tsv:
    json_file = out_path / f"MultilingualTEDx.jsonl"
    with json_file.open("w", encoding="utf-8") as f_json:

        n_entries = 0
        t_entries = 0
        for lsrc, ltgt in lang_pairs:
            if lsrc == ltgt:
                continue

            for data_set in data_sets:
                print(f"---------- {lsrc}-{ltgt}:{data_set} ----------")
                source_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{lsrc}"
                target_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{ltgt}"
                segments_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"segments"

                flac_stem2path = get_audio_dict(base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "wav")

                n_created = 0
                n_exist = 0
                t_audio = 0
                n_skipped = 0

                segments_dict = build_segments_dict(segments_path, source_path, target_path)

                for audio_stem, segments in tqdm(segments_dict.items(), desc=f"Processing {lsrc}-{ltgt}:{data_set}", unit="file"):

                    results, n, m, duration, s = extract_fragments(flac_stem2path[audio_stem], segments, out_path / "audios"/ "MultilingualTEDx")
                    n_created += n
                    n_exist += m
                    t_audio += duration
                    n_skipped += s

                    for ofile_name, seg in results:
                        out_file = str(out_path / "audios" / "MultilingualTEDx" / ofile_name)
                        # f_tsv.write(f"{out_file}\t{lsrc}\t{seg['src']}\t{ltgt}\t{seg['tgt']}\t{data_set}\n")
                        f_json.write(
                            json.dumps({
                                "audio_file": out_file,
                                "split": data_set,
                                "transcription": {
                                    "lang": lsrc, 
                                    "text": seg['src']
                                },
                                "translation": {
                                    "lang": ltgt,
                                    "text": seg['tgt']
                                }
                            }, ensure_ascii=False) + "\n"
                        )
                print(f"Created {n_created} files ({n_exist} existing), total duration {t_audio:.1f} secs, skipped {n_skipped} segments")

            n_entries += n_created + n_exist
            t_entries += t_audio

    print(f"Total entries {n_entries}")
    print(f"Total duration {t_entries:.1f} secs ({t_entries/n_entries:.1f} secs/file)")

if __name__ == "__main__":
    main()
