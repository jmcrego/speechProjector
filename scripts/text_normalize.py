import re
import html
import unicodedata
import logging

logger = logging.getLogger(__name__)

pattern_brackets = re.compile(r"[\(\[\{].*?[\)\]\}]")    

currency_map = {
    "$": "dollars",
    "€": "euros",
    "£": "pounds",
    "¥": "yen",
    "₹": "rupees",
    "₩": "won",
    "₽": "rubles",
    "₺": "lira",
    "₫": "dong",
    "₴": "hryvnia",
    "₦": "naira",
    "₱": "peso",
    "₲": "guarani",
    "₳": "austral",
    "₵": "cedi",
    "₸": "tenge",
    "₼": "manat",
    "₽": "ruble",
    "₾": "lari",
    "₿": "bitcoin",
}

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
    def replacer(match):
        content = match.group(0)
        logger.debug(f"Removing bracketed content: {content}")
        return " "
    # This regex matches any content within (), [], {}, including nested ones (non-greedy match). It will remove the brackets and their content.
    # Note: This will not handle nested brackets of the same type correctly (e.g., "This is (a test (with nested) brackets) example"), 
    # but it will handle different types of brackets nested within each other (e.g., "This is [a test (with nested) brackets] example").
    # For structures with multiple non-overlapped labels like "This is (a test) and [another test] example", it will remove both "(a test)" and "[another test]" correctly.
    text = pattern_brackets.sub(replacer, text)
    return text

def remove_unescape_html(text):
    # Unescape HTML entities (&gt;, &nbsp;, etc.) and log the unescaped entities
    def unescape_replacer(match):
        entity = match.group(0)
        unescaped = html.unescape(entity)
        logger.debug(f"Unescaping HTML entity: {entity} → {unescaped}")
        return unescaped
    text = re.sub(r"&[a-zA-Z]+?;", unescape_replacer, text)

    # Remove HTML tags (e.g., <br>, <p>, <span>) and log the removed tags
    def replacer(match):
        content = match.group(0)
        logger.debug(f"Removing HTML tag: {content}")
        return " "

    text = re.sub(r"<[^>]+>", replacer, text)

    # Some HTML entities turn into non-breaking spaces
    text = text.replace("\u00a0", " ")
    return text

def replace_currency(text: str) -> str:
    # Replace currency symbols with their names (e.g., $ → dollars, € → euros) and log the replacements
    for symbol, name in currency_map.items():
        if symbol in text:
            logger.debug(f"Replacing currency symbol: {symbol} → {name}")
        text = text.replace(symbol, f" {name} ")
    return text

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercasing
    text = text.lower()

    # remove html tags and unescape entities
    text = remove_unescape_html(text)

    # keep only valid Unicode characters (remove invalid byte sequences)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Remove string within brackets (e.g., [noise], (laughter), {music})
    text = remove_brackets(text)

    # Replace currency symbols with their names (e.g., $ → dollars, € → euros)
    # text = replace_currency(text)

    # Remove diacritics (e.g., café → cafe, naïve → naive, coöperate → cooperate)
    # text = remove_diacritics(text)

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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    f = open(args.input_file, "r", encoding="utf-8") if args.input_file else sys.stdin
    for idx, l in enumerate(f):
        l = l.strip()
        logger.debug(f"[src{idx}] {l}")
        n = normalize_text(l)
        print(f"{n}")
        logger.debug(f"[tgt{idx}] {n}")
    f.close() if args.input_file else None
