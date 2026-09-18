"""Rewrite a chunk of text into what should actually be spoken.

Two off-the-shelf pieces do the work; everything here is glue and patches.

* ``markdown-it`` parses the text as CommonMark and only its text tokens are
  kept, so list markers, emphasis, code fences, link targets and heading
  hashes never reach the voice.
* ``wetext`` (WeTextProcessing's Chinese TN rules, in a runtime that needs no
  pynini) rewrites numbers for the ear: 35% → 百分之三十五, 3.5 → 三点五,
  10:30 → 十点三十分, 1/3 → 三分之一, ¥199 → 一百九十九元.

``wetext`` still gets a handful of shapes wrong, so they are rewritten before
it runs (each one measured — see the README's section on text normalisation):

    2024 年        cardinal instead of a year, when a space splits the digits
                   from 年/月/日/时/分   → 两千零二十四 年
    1,299美元      the thousands comma survives → 一,两百九十九美元
    2024-09-10     read as two subtractions → 两千零二十四到零九…
    6-02           read as a subtraction → 六减零二
    3-5 天         so is a range → 三减五天
    GPT-4          letter-hyphen-digit is a minus → GPT负四
    -5 度          → 负五度, where 零下五度 is the idiom
    +3%            the unary plus is spoken → 百分之正三
    010-88886666   an area code hyphen is a minus

and the spaces wetext leaves between Chinese characters are removed after it.

Parentheses become a short pause rather than vanishing, because they hold real
content as often as stage directions; ``strip_stage=True`` drops the short ones
（轻叹一声）for models that write those.

``normalize`` returns ``""`` when nothing sayable is left (a table rule, a line
of emoji) — the caller's signal to skip the chunk.
"""
from __future__ import annotations

import re

from markdown_it import MarkdownIt
from wetext import Normalizer

_md = MarkdownIt("commonmark")
_tn = Normalizer(lang="zh", operator="tn")

_CJK = r"一-鿿㐀-䶿"

# --- markdown -> plain text ------------------------------------------------

def _plain(text: str) -> str:
    """Only what a reader would say out loud, block by block."""
    out: list[str] = []
    for token in _md.parse(text):
        if token.type == "inline":
            said = "".join(c.content for c in (token.children or []) if c.type in ("text", "code_inline"))
            if said.strip():
                out.append(said.strip())
        elif token.type == "fence":
            continue  # a code block is not speech
    return "\n".join(out)


# --- patches applied before wetext ----------------------------------------

_DATE_UNIT = "年月日时分"
_SPACE_BEFORE_UNIT = re.compile(rf"(\d)\s+(?=[{_DATE_UNIT}])")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# A hyphen between two numbers is a date only when the day is zero-padded
# (6-02): 3-5 天 and 版本 1-2 are ranges, and guessing wrong is worse than
# reading a range. Everything else numeric-hyphen-numeric is a phone number
# (area code + 7-8 digits, read digit by digit) or a range ("到").
_ISO_DATE = re.compile(r"(?<![\d\-/])(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])(?![\d\-/])")
_MONTH_DAY = re.compile(r"(?<![\d.\-/])([1-9]|1[0-2])-(0[1-9])(?![\d\-/])")
_PHONE = re.compile(r"(?<![\d\-])(\d{3,4})-(\d{7,8})(?![\d\-])")
_NUM_RANGE = re.compile(r"(?<=\d)-(?=\d)")
_ALPHA_HYPHEN_NUM = re.compile(r"(?<=[A-Za-z])-(?=\d)")
_MINUS_DEGREE = re.compile(r"[-−]\s*(\d+(?:\.\d+)?)\s*(?=度|℃|°C)")
_UNARY_PLUS = re.compile(r"(?<![\w.])\+(?=\d)")
_SPACE_AROUND_CJK = re.compile(rf"(?<=[{_CJK}])[ \t]+|[ \t]+(?=[{_CJK}])")

_DIGITS = "零一二三四五六七八九"


def _han(n: int) -> str:
    """1-31 the way a date is read: 二日, never 两日."""
    if n < 10:
        return _DIGITS[n]
    if n < 20:
        return "十" + (_DIGITS[n % 10] if n % 10 else "")
    return _DIGITS[n // 10] + "十" + (_DIGITS[n % 10] if n % 10 else "")


def _prefix_fixes(text: str) -> str:
    text = _SPACE_BEFORE_UNIT.sub(r"\1", text)          # 2024 年 -> 2024年
    text = _THOUSANDS.sub("", text)                     # 1,299 -> 1299
    # 2024-09-10 and 6-02 are dates, not subtractions. The day is written out
    # so wetext cannot turn 2日 into 两日; the year is left to wetext, which
    # reads 2024年 digit by digit the way a year is read.
    text = _ISO_DATE.sub(lambda m: f"{m.group(1)}年{_han(int(m.group(2)))}月{_han(int(m.group(3)))}日", text)
    text = _MONTH_DAY.sub(lambda m: f"{_han(int(m.group(1)))}月{_han(int(m.group(2)))}日", text)
    text = _ALPHA_HYPHEN_NUM.sub(" ", text)             # GPT-4 -> GPT 4
    text = _PHONE.sub(r"\1\2", text)                    # 010-88886666 -> 01088886666
    text = _NUM_RANGE.sub("到", text)                    # 3-5 天 -> 3到5 天
    text = _MINUS_DEGREE.sub(r"零下\1", text)            # -5 度 -> 零下5度
    text = _UNARY_PLUS.sub("", text)                    # +3% -> 3%
    return text


# --- parentheses -----------------------------------------------------------

_PAREN = re.compile(r"[（(]([^（()）]{0,12})[）)]")
_ANY_PAREN = re.compile(r"[（(]([^（()）]*)[）)]")


def _parentheses(text: str, strip_stage: bool) -> str:
    if strip_stage:
        text = _PAREN.sub("", text)
    return _ANY_PAREN.sub(lambda m: f"，{m.group(1)}，" if m.group(1).strip() else "", text)


# --- public ----------------------------------------------------------------

_SAYABLE = re.compile(rf"[{_CJK}A-Za-z0-9]")
_REPEATED_PAUSE = re.compile(r"[，,]\s*(?=[，,。！？!?；;])")
_LEADING_PAUSE = re.compile(r"^[\s，,]+")
_TRAILING_PAUSE = re.compile(r"[\s，,]+$")


def normalize(text: str, *, strip_stage: bool = False) -> str:
    """Markdown in, speech out. Empty string when nothing is left to say."""
    if not text or not text.strip():
        return ""
    said = _plain(text)
    said = _parentheses(said, strip_stage)
    if not _SAYABLE.search(said):
        return ""
    said = _tn.normalize(_prefix_fixes(said))
    said = _SPACE_AROUND_CJK.sub("", said)
    said = _REPEATED_PAUSE.sub("", said)
    return _TRAILING_PAUSE.sub("", _LEADING_PAUSE.sub("", said))
