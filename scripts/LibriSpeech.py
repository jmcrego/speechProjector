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
#import sys
#from scipy.io.wavfile import write
from utils import duration #load_audio_ffmpeg, extract_fragments, 
from text_normalize import normalize_text


def main():
    parser = argparse.ArgumentParser(description="Extract LibriSpeech audio fragments and build JSONL.")
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
                            print(f"Audio file {audio_path} not found, skipping")
                            continue

                        d = duration(audio_path)
                        if d > 30.0:
                            print(f"Audio file {audio_path} is too long ({duration(audio_path):.1f} sec), skipping")
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

        print(f"Created {n_out} audio segments out of {n_in} available for {data_set}. Total duration is {t_out:.1f} secs ({t_out/n_out:.1f} secs/file)\n")


    # json_file = out_path / f"MultilingualTEDx.jsonl"
    # with json_file.open("w", encoding="utf-8") as f_json:

    #     n_entries = 0
    #     t_entries = 0
    #     for lsrc, ltgt in lang_pairs:
    #         if lsrc == ltgt:
    #             continue

    #         for data_set in data_sets:
    #             print(f"---------- {lsrc}-{ltgt}:{data_set} ----------")
    #             source_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{lsrc}"
    #             target_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{ltgt}"
    #             segments_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"segments"

    #             flac_stem2path = get_audio_dict(base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "wav")

    #             n_created = 0
    #             n_exist = 0
    #             t_audio = 0
    #             n_skipped = 0

    #             segments_dict = build_segments_dict(segments_path, source_path, target_path)

    #             for audio_stem, segments in tqdm(segments_dict.items(), desc=f"Processing {lsrc}-{ltgt}:{data_set}", unit="file"):

    #                 results, n, m, duration, s = extract_fragments(flac_stem2path[audio_stem], segments, out_path / "audios"/ "MultilingualTEDx")
    #                 n_created += n
    #                 n_exist += m
    #                 t_audio += duration
    #                 n_skipped += s

    #                 for ofile_name, seg in results:
    #                     out_file = str(out_path / "audios" / "MultilingualTEDx" / ofile_name)
    #                     # f_tsv.write(f"{out_file}\t{lsrc}\t{seg['src']}\t{ltgt}\t{seg['tgt']}\t{data_set}\n")
    #                     f_json.write(
    #                         json.dumps({
    #                             "audio_file": out_file,
    #                             "split": data_set,
    #                             "transcription": {
    #                                 "lang": lsrc, 
    #                                 "text": seg['src']
    #                             },
    #                             "translation": {
    #                                 "lang": ltgt,
    #                                 "text": seg['tgt']
    #                             }
    #                         }, ensure_ascii=False) + "\n"
    #                     )
    #             print(f"Created {n_created} files ({n_exist} existing), total duration {t_audio:.1f} secs, skipped {n_skipped} segments")

    #         n_entries += n_created + n_exist
    #         t_entries += t_audio

    # print(f"Total entries {n_entries}")
    # print(f"Total duration {t_entries:.1f} secs ({t_entries/n_entries:.1f} secs/file)")

if __name__ == "__main__":
    main()
