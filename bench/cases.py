"""Shared test material: the sentences every bench uses, and the reference clip
they clone from.

**The reference clip is not in this repository.** A recorded voice belongs to
whoever spoke it, so bring your own and point the benches at it:

    export OMNIVOICE_REF_WAV=assets/my-voice.wav
    export OMNIVOICE_REF_TEXT="它念的那句话，标点照写。"

Any clean mono recording works. Every number in the README was measured with a
3.7 s clip (92 audio tokens); a longer one only makes the prompt longer, and
the prompt is on the critical path of every denoising step. The transcript has
to be what the clip actually says — it goes into the prompt, it is not a label.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REF_WAV_ENV = "OMNIVOICE_REF_WAV"
REF_TEXT_ENV = "OMNIVOICE_REF_TEXT"

_MISSING = f"""no reference clip configured.

Every bench clones one voice. Record or pick a clean 3-4 s mono clip, write down
exactly what it says, and set both:

    export {REF_WAV_ENV}=/path/to/clip.wav
    export {REF_TEXT_ENV}="what the clip says, punctuation included."
"""


def reference() -> tuple[Path, str]:
    """(clip, transcript) from the environment, or exit with instructions."""
    wav, text = os.environ.get(REF_WAV_ENV), os.environ.get(REF_TEXT_ENV)
    if not wav or not text:
        raise SystemExit(_MISSING)
    path = Path(wav).expanduser()
    if not path.is_file():
        raise SystemExit(f"{REF_WAV_ENV}={wav} is not a file")
    return path, text


def __getattr__(name: str):  # PEP 562: keep `from cases import REF_WAV, REF_TEXT` working
    if name == "REF_WAV":
        return reference()[0]
    if name == "REF_TEXT":
        return reference()[1]
    raise AttributeError(name)

CASES = [
    ("opener", "今天天气不错，"),
    ("sentence", "我们出去走走吧，顺便买点东西回来。"),
    ("long", "北京今天多云转晴，气温十八到二十五度，风力三级，出门建议带一件外套。"),
]

ASR_SET = {
    "00": "明天下午三点半提醒我去趟银行，顺便把水电费交了。",
    "01": "帮我查一下从北京南站到上海虹桥最早的一班高铁。",
    "02": "这个月工资是一万两千三百块，比上个月多了八百。",
    "03": "我昨天在中关村碰到了老王，他说他家闺女考上了复旦。",
    "04": "把空调调到二十六度，风速调小一点，别对着人吹。",
    "05": "你能不能用英文说一遍 good morning everyone？",
    "06": "那家川菜馆的水煮鱼太辣了，下次咱们换个粤菜试试。",
    "07": "我手机快没电了，附近哪里有共享充电宝？",
    "08": "上周的季度报告发给张经理和李总了吗？",
    "09": "把这首歌的音量调低，然后放一首周杰伦的。",
    "10": "这道题的答案是三分之二还是四分之三？",
    "11": "帮我订一张后天飞深圳的机票，靠窗，经济舱就行。",
    "12": "我妈让我周末回家吃饭，说要包饺子。",
    "13": "这台电脑的内存是十六个 G，硬盘是一个 T 的固态。",
    "14": "会议改到星期四上午十点，地点还在三楼会议室。",
    "15": "你觉得今年的房价还会不会继续跌？",
    "16": "快递说下午五点前送到，我不在家，放在门口就行。",
    "17": "把 WiFi 密码发我一下，我手机连不上。",
    "18": "他今天有点感冒，嗓子哑了，说话声音很小。",
    "19": "刚才那个电话是谁打来的？我没听清楚。",
}


def sentence_set(which: str) -> list[tuple[str, str]]:
    if which == "cases":
        return list(CASES)
    if which == "asr":
        return list(ASR_SET.items())
    if which == "all":
        return list(CASES) + list(ASR_SET.items())
    raise ValueError(which)


def all_texts() -> dict[str, str]:
    return dict(CASES) | ASR_SET


if __name__ == "__main__":
    print(json.dumps(all_texts(), ensure_ascii=False, indent=1))
