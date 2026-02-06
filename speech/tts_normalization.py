import re

# ---------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------

_money_re = re.compile(
    r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?"
)

def _spell_int(n: int) -> str:
    small = [
        "zero","one","two","three","four","five","six","seven","eight","nine",
        "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
        "seventeen","eighteen","nineteen"
    ]
    tens = ["","", "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

    if n < 20:
        return small[n]
    if n < 100:
        t, r = divmod(n, 10)
        return tens[t] + ("" if r == 0 else f"-{small[r]}")
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{small[h]} hundred" + ("" if r == 0 else f" {_spell_int(r)}")
    if n < 10000:
        th, r = divmod(n, 1000)
        return f"{small[th]} thousand" + ("" if r == 0 else f" {_spell_int(r)}")
    return str(n)

def normalize_currency(text: str) -> str:
    """Convert $1.25 → 'one dollar and twenty-five cents'."""
    def repl(m):
        dollars_str = m.group(1).replace(",", "")
        cents_str = m.group(2)

        try:
            dollars = int(dollars_str)
            cents = int((cents_str or "0").ljust(2, "0")) if cents_str else 0
        except ValueError:
            return m.group(0)

        if dollars == 0 and cents > 0:
            c = _spell_int(cents)
            return f"{c} cent{'s' if cents != 1 else ''}"

        parts = []
        if dollars > 0:
            d = _spell_int(dollars)
            parts.append(f"{d} dollar{'s' if dollars != 1 else ''}")
        if cents > 0:
            c = _spell_int(cents)
            conj = " and " if dollars > 0 else ""
            parts.append(f"{conj}{c} cent{'s' if cents != 1 else ''}")

        return "".join(parts) if parts else "zero dollars"

    return _money_re.sub(repl, text)

# ---------------------------------------------------------------------
# General pronunciation normalization
# ---------------------------------------------------------------------

def normalize_pronunciation(text: str) -> str:
    """
    Normalize text so it is spoken clearly by TTS engines.
    """

    # Math-like patterns: 4x -> 4 x, 2y -> 2 y
    text = re.sub(r"\b(\d+)\s*([a-zA-Z])\b", r"\1 \2", text)

    # Prevent vowel fusion (AI → ay eye)
    text = re.sub(r"\bAI\b", "ay eye", text, flags=re.I)

    # Acronyms: speak letters individually
    ACR_SPACE = {
        r"\bLLM\b":  "L L M",
        r"\bAPI\b":  "A P I",
        r"\bCPU\b":  "C P U",
        r"\bGPU\b":  "G P U",
        r"\bNPU\b":  "N P U",
        r"\bSSD\b":  "S S D",
        r"\bFPS\b":  "F P S",
        r"\bYOLO\b": "Y O L O",
        r"\bRTSP\b": "R T S P",
        r"\bHTTP\b": "H T T P",
        r"\bUSB\b":  "U S B",
    }
    for pat, repl in ACR_SPACE.items():
        text = re.sub(pat, repl, text, flags=re.I)

    # Punctuation spacing fixes
    text = re.sub(r"([.?!])(?=\S)", r"\1 ", text)
    text = re.sub(r"(?<=\w)\.(?=\w)", ". ", text)
    text = re.sub(r"(?<=\w)/(?=\w)", "/ ", text)

    return re.sub(r"\s{2,}", " ", text).strip()

# ---------------------------------------------------------------------
# Canonical TTS pipeline
# ---------------------------------------------------------------------

def normalize_for_tts(text: str) -> str:
    text = normalize_currency(text)
    text = normalize_pronunciation(text)
    return text
