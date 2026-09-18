"""Text side of the prompt: tokenizer (``tokenizers``, no transformers), the
official string assembly and the non-verbal tag handling.

Mirrors ``omnivoice/models/omnivoice.py`` (``_combine_text``,
``_tokenize_with_nonverbal_tags``, ``_prepare_inference_inputs``) and
``omnivoice/utils/text.py`` (``add_punctuation``).
"""
from __future__ import annotations

import re
from pathlib import Path

from tokenizers import Tokenizer

NONVERBAL = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

END_PUNCTUATION = set(';:,.!?…)]}"\'“”‘’；：，。！？、）】') | {"……"}

_CJK = r"[一-鿿]"


class TextTokenizer:
    def __init__(self, model_dir: str | Path):
        path = Path(model_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (k2-fsa/OmniVoice ships it)")
        self.tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids


def combine_text(text: str, ref_text: str | None = None) -> str:
    full = (ref_text.strip() + " " + text.strip()) if ref_text else text.strip()
    full = re.sub(r"[\r\n]+", "", full)
    full = full.replace("（", "(").replace("）", ")")
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(rf"(?<={_CJK})\s+|\s+(?={_CJK})", "", full)
    return full


def add_punctuation(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in END_PUNCTUATION:
        text += "。" if any("一" <= c <= "鿿" for c in text) else "."
    return text


def tokenize_with_nonverbal_tags(text: str, tok: TextTokenizer) -> list[int]:
    parts: list[int] = []
    last = 0
    for m in NONVERBAL.finditer(text):
        if m.start() > last:
            parts.extend(tok.encode(text[last:m.start()]))
        parts.extend(tok.encode(m.group()))
        last = m.end()
    if last < len(text):
        parts.extend(tok.encode(text[last:]))
    return parts or tok.encode(text)


def prompt_text_ids(tok: TextTokenizer, text: str, *, ref_text: str | None, language: str | None,
                    instruct: str | None, denoise: bool) -> tuple[list[int], list[int]]:
    """(style ids, text ids) exactly as the official ``_prepare_inference_inputs`` builds them."""
    style = "<|denoise|>" if denoise else ""
    style += f"<|lang_start|>{language or 'None'}<|lang_end|>"
    style += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
    style_ids = tok.encode(style)
    wrapped = f"<|text_start|>{combine_text(text, ref_text)}<|text_end|>"
    return style_ids, tokenize_with_nonverbal_tags(wrapped, tok)


SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set("\"'“”‘’）]》>」】")
ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.", "Capt.", "Gen.",
    "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.",
    "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.",
    "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
}


def chunk_text_punctuation(text: str, chunk_len: int, min_chunk_len: int | None = None) -> list[str]:
    """Official ``chunk_text_punctuation``: split at punctuation (abbreviation-aware), merge to ``chunk_len``."""
    sentences: list[list[str]] = []
    cur: list[str] = []
    for ch in text:
        if not cur and sentences and (ch in SPLIT_PUNCTUATION or ch in CLOSING_MARKS):
            sentences[-1].append(ch)
            continue
        cur.append(ch)
        if ch in SPLIT_PUNCTUATION:
            abbrev = False
            if ch == ".":
                tmp = "".join(cur).strip()
                if tmp and tmp.split()[-1] in ABBREVIATIONS:
                    abbrev = True
            if not abbrev:
                sentences.append(cur)
                cur = []
    if cur:
        sentences.append(cur)

    merged: list[list[str]] = []
    chunk: list[str] = []
    for sent in sentences:
        if len(chunk) + len(sent) <= chunk_len:
            chunk.extend(sent)
        else:
            if chunk:
                merged.append(chunk)
            chunk = sent
    if chunk:
        merged.append(chunk)

    if min_chunk_len is not None:
        first_short = bool(merged) and len(merged[0]) < min_chunk_len
        final: list[list[str]] = []
        for i, c in enumerate(merged):
            if i == 1 and first_short:
                final[-1].extend(c)
            elif len(c) >= min_chunk_len or not final:
                final.append(c)
            else:
                final[-1].extend(c)
    else:
        final = merged
    return [s for s in ("".join(c).strip() for c in final) if s]
