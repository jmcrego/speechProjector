import re
import html
import unicodedata


# Carefully chosen punctuation set (Unicode categories starting with P)
# We remove characters classified as punctuation by Unicode
def remove_punctuation(text: str) -> str:
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercasing
    text = text.lower()

    # 3. Remove HTML tags (e.g., <br>, <p>, <span>)
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Unescape HTML entities (&gt;, &nbsp;, etc.)
    text = html.unescape(text)

    # Some HTML entities turn into non-breaking spaces
    text = text.replace("\u00a0", " ")

    # 5. Remove punctuation (Unicode-aware)
    text = remove_punctuation(text)

    # 6. Remove string within brackets (e.g., [noise], (laughter), {music})
    text = re.sub(r"[\[\(\{][^\]\)\}]*[\]\)\}]", " ", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test text normalization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", type=str, help="Input file to normalize OR use stdin if not given")
    args = parser.parse_args()

    f = open(args.input_file, "r", encoding="utf-8") if args.input_file else sys.stdin
    for l in f:
        l = l.strip()
        print(f"Orig: {l}")
        print(f"Norm: {normalize_text(l)}")
    f.close() if args.input_file else None
    