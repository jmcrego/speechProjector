import json
import csv
import sys

def jsonl_to_tsv(json_path: str, keys: list[str] = []):

    writer = csv.writer(sys.stdout, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    with open(json_path, "r", encoding="utf-8") as f:
        header = None
        for idx, line in enumerate(f):
            sample = json.loads(line)

            if header is None:
                header = keys if keys else sample.keys()
                writer.writerow(header)

            row = [sample.get(col, "") for col in header]
            writer.writerow(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSONL dataset to TSV format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--keys", type=str, nargs="+", help="List of jsonl keys to include OR include all keys if not given")
    args = parser.parse_args()
    # example usage: python json2tsv.py --jsonl_path data/samples.jsonl --keys audio_file transcription translation
    jsonl_to_tsv(args.jsonl_path, args.keys)