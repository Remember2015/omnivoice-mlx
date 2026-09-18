"""Chinese text normalisation for OmniVoice: what the model can read aloud.

The rewriting itself is :mod:`omnivoice_mlx.ttstext` — markdown-it to throw
away what a reader would not say, wetext for the numbers, plus patches for the
shapes that normaliser still gets wrong. It needs two packages this one does
not (``pip install markdown-it-py wetext``), so it is imported lazily and
``normalize`` is off by default everywhere. README section 16 has the
34-sentence CER measurements that say what the layer is worth (7.09 % raw →
0.33 % normalised, which is the TTS+ASR noise floor).

What this module adds is OmniVoice's inline control syntax, which that layer
knows nothing about and would destroy:

  * bracketed tags — ``[laughter]``, ``[sigh]``, ``[question-en]`` and the CMU
    pronunciation overrides ``[B EY1 S]``;
  * pinyin tone markers — ``HAO3``, whose tone digit ``zh_normalization``
    happily reads as a number (``念 HAO3 这个音`` -> ``念 HAO 三这个音``).

Both are held out of the rewrite and put back verbatim, the way the official
``omnivoice.utils.text.normalize_text`` does it (``_apply_with_protection``).
Two deliberate differences from the official code:

  * the spans are masked with private-use characters and the *whole* string is
    normalised in one pass, instead of normalising the gaps separately. The
    layer above trims a trailing pause off whatever it is handed, so
    per-gap normalisation would silently eat the comma in ``你好，[laughter]``.
  * the official pinyin pattern ``[A-Z]+[1-5]`` also matches inside ordinary
    tokens: it protects the ``PM2`` of ``PM2.5`` and the ``A1`` of ``A100``,
    which leaves half a token unread. A tone digit here has to end the token
    (``[A-Z]+[1-5](?![a-z0-9.])``), so ``MP3`` and ``NI2HAO3`` are still safe
    while ``PM2.5`` goes through as ``PM 二点五``.

``normalize`` is idempotent on prose and on its own output.
"""
from __future__ import annotations

import re
import sys

__all__ = ["normalize"]


def _load_tts_text():
    """The rewriting layer, imported lazily: it pulls in markdown-it and wetext,
    which the rest of the package does not need."""
    try:
        from . import ttstext  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"text normalisation needs its own two dependencies ({e}): "
            "pip install markdown-it-py wetext  (requirements-textnorm.txt). "
            "generate(normalize=True) is the only caller."
        ) from e
    return ttstext


_tts_text = None


def _layer():
    global _tts_text
    if _tts_text is None:
        _tts_text = _load_tts_text()
    return _tts_text

# Both copied from omnivoice/utils/text.py (the reference package, v0.2.1);
# the pinyin one is tightened, see the module docstring.
_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")
_PINYIN_TONE_RE = re.compile(r"[A-Z]+[1-5](?![a-z0-9.])")

# Placeholders for the protected spans. A different private-use block from the
# one tts_text.read_numbers uses for its Latin runs (U+E000..U+E0FF), so its
# own unmasking pass cannot touch ours.
_MASK_BASE = 0xE200
_MASK_RE = re.compile(r"[\ue200-\ue2ff]")
_ANY_PUA = re.compile(r"[\ue000-\ue2ff]")


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in _BRACKET_TAG_RE.finditer(text)]
    spans += [m.span() for m in _PINYIN_TONE_RE.finditer(text)]
    if not spans:
        return []
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        # Overlapping spans merge, and so do spans separated by whitespace
        # alone: "NI3 HAO3" is one control span, and the space between the two
        # syllables is part of it. Masking them separately would hand the
        # normaliser a lone space between two placeholders, and it deletes it.
        if merged and (start <= merged[-1][1] or not text[merged[-1][1]:start].strip()):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def normalize(text: str, *, strip_stage: bool = False) -> str:
    """Rewrite one chunk of TTS input into what should be spoken.

    ``strip_stage`` drops short parentheticals such as ``（轻叹一声）`` instead
    of turning them into a pause; that is the external layer's option for chat models
    that write stage directions, off by default.

    Returns ``""`` when nothing sayable is left (a table rule, a line of emoji),
    which is the caller's signal to skip the chunk entirely.
    """
    if not text or not text.strip():
        return text
    text = _ANY_PUA.sub("", text)  # ours to use as placeholders

    spans = _protected_spans(text)
    if not spans:
        return _layer().normalize(text, strip_stage=strip_stage)

    if len(spans) > 0xFF:
        spans = spans[:0xFF]
    held: list[str] = []
    out: list[str] = []
    last = 0
    for start, end in spans:
        out.append(text[last:start])
        out.append(chr(_MASK_BASE + len(held)))
        held.append(text[start:end])
        last = end
    out.append(text[last:])
    masked = "".join(out)

    said = _layer().normalize(masked, strip_stage=strip_stage)
    if not said:
        # Everything sayable was a protected span, so tts_text's "is there
        # anything left to say" guard saw only placeholders and gave up.
        said = re.sub(r"\s+", " ", masked).strip()
    return _MASK_RE.sub(lambda m: held[ord(m.group(0)) - _MASK_BASE], said)


if __name__ == "__main__":  # quick probe: python -m omnivoice_mlx.textnorm "文本"
    for arg in sys.argv[1:] or [line.rstrip("\n") for line in sys.stdin]:
        print(f"{arg}\n  -> {normalize(arg)}")
