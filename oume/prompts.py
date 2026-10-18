"""Deck-agnostic study prompts for the three OUME modalities."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Iterable

_TAGS = re.compile(r"<[^>]+>")


def plain(value: str) -> str:
    text = html.unescape(_TAGS.sub(" ", value or ""))
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(card: dict, modality, guided: bool, distractors: Iterable[str]) -> dict:
    front = plain(card.get("front", ""))
    back = plain(card.get("back", ""))
    reading = plain(card.get("reading", ""))
    gloss = plain(card.get("gloss", ""))
    example = plain(card.get("example", "")) or plain(card.get("context", ""))
    cue = gloss or back
    name = modality.name if hasattr(modality, "name") else str(modality)
    if name == "VISUAL_CHOICE":
        shown = front if not reading or reading == front else f"{front}（{reading}）"
        return _choice(front, back, shown, guided, list(distractors))
    if name == "AUDIO_SPELL":
        return {
            "kind": "spell",
            "instruction": "看释义，输入目标词或读音。题面不会给出要写的词。",
            "shown": cue,
            "audio": card.get("audio") or "",
            "expected_hidden": True,
        }
    cloze_target = front if front and front in example else ""
    return {
        "kind": "generate",
        "instruction": "用目标词补全语境。没有可靠人工评分时，只按对/错记为 Good 或 Again。",
        "shown": _cloze(example, cloze_target) if example else f"____ — {cue}",
        "expected_hidden": True,
    }


def _choice(front: str, back: str, context: str, guided: bool, distractors: list[str]) -> dict:
    options = [back] + [item for item in distractors if plain(item) and plain(item) != back]
    options = list(dict.fromkeys(plain(item) for item in options if plain(item)))[:4]
    while len(options) < 2:
        options.append(f"（牌组选项不足 {len(options) + 1}）")
    seed = int(hashlib.sha256(f"{front}|{back}".encode()).hexdigest()[:8], 16)
    ordered = options[:]
    for index in range(len(ordered) - 1, 0, -1):
        swap = seed % (index + 1)
        ordered[index], ordered[swap] = ordered[swap], ordered[index]
        seed //= index + 1
    return {
        "kind": "choice",
        "instruction": "先看答案并确认一次，这次不计为困难回忆。" if guided else "选择与提示匹配的一项。",
        "shown": context,
        "options": ordered,
        "reveal_answer": back if guided else "",
    }


def _cloze(context: str, target: str) -> str:
    if target and target in context:
        return context.replace(target, "____", 1)
    return f"____ — {context}" if context else "用你记得的表达补全：____"
