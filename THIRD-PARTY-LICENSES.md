# Third-party code in this repository

## omnivoice_mlx/higgs/ — mlx-audio

The Higgs-audio v2 tokenizer (`tokenizer.py`, `dac.py`, `semantic.py`,
`wav2vec.py`, `config.py`, `codec_ops.py`) is vendored from
[mlx-audio](https://github.com/Blaizzy/mlx-audio) so that the codec needs
neither mlx-audio nor transformers at runtime. Local edits are marked
`# vendored:` and the encoded tokens / decoded waveforms are bit-identical to
mlx-audio's.

```
MIT License

Copyright (c) 2024 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## The rest of omnivoice_mlx/ — derived from k2-fsa/OmniVoice

Apache-2.0, see LICENSE and NOTICE. Upstream:
<https://github.com/k2-fsa/OmniVoice>.

## Not in this repository

* **Model weights** (`k2-fsa/OmniVoice` on Hugging Face) — CC-BY-NC, and so is
  anything `scripts/convert.py` writes from them. Non-commercial, attribution.
* **The Higgs-audio v2 tokenizer weights** (`eustlb/higgs-audio-v2-tokenizer`)
  — see that repository.
* **The evaluation models** (Fun-ASR-Nano, CAM++, UTMOS22-strong) — what the
  README's CER / speaker-similarity / UTMOS numbers were measured with. Neither
  the models nor the harness that drove them is part of this repository.
* **The reference voice clip** — you supply your own, see `bench/cases.py`.
