import re
import html
import unicodedata
import logging

logger = logging.getLogger(__name__)

pattern_brackets = re.compile(r"[\(\[\{].*?[\)\]\}]")    

# Carefully chosen punctuation set (Unicode categories starting with P)
# We remove characters classified as punctuation by Unicode
def remove_punctuation(text: str) -> str:
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )

# café → cafe, naïve → naive, coöperate → cooperate
def remove_diacritics(text):
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )

def remove_brackets(text):
    # Remove content within (), [], {}, and the brackets themselves, log removed content for debugging
    def replacer(match):
        content = match.group(0)
        logger.debug(f"Removing bracketed content: {content}")
        return " "
    # This regex matches any content within (), [], {}, including nested ones (non-greedy match)
    text = pattern_brackets.sub(replacer, text)
    return text

def remove_html(text):
    # Remove HTML tags (e.g., <br>, <p>, <span>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities (&gt;, &nbsp;, etc.)
    text = html.unescape(text)
    # Some HTML entities turn into non-breaking spaces
    text = text.replace("\u00a0", " ")
    return text

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercasing
    text = text.lower()

    # remove html tags and unescape entities
    text = remove_html(text)

    # keep only valid Unicode characters (remove invalid byte sequences)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Remove string within brackets (e.g., [noise], (laughter), {music})
    text = remove_brackets(text)

    # Remove punctuation (Unicode-aware)
    text = remove_punctuation(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test text normalization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", type=str, help="Input file to normalize OR use stdin if not given")
    args = parser.parse_args()

    f = open(args.input_file, "r", encoding="utf-8") if args.input_file else sys.stdin
    for idx, l in enumerate(f):
        l = l.strip()
        print(f"\n--- Sample {idx} ---")
        print(f"[src] {l}")
        print(f"[tgt] {normalize_text(l)}")
    f.close() if args.input_file else None
