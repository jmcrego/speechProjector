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
    parser = argparse.ArgumentParser(description="Convert JSON dataset to TSV format")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--keys", type=str, nargs="+", help="List of keys to keep when reading the jsonl file")
    args = parser.parse_args()

    jsonl_to_tsv(args.json_path, args.keys)