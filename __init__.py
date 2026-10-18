"""OUME study window for Anki 26.

The window owns scheduling for one deck. Anki's reviewer and FSRS queue are
not asked to advance those cards, which avoids two schedulers writing two
different intervals for the same exposure.
"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

from aqt import gui_hooks, mw
from aqt.qt import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QTimer,
    QVBoxLayout,
    qconnect,
)
from aqt.utils import tooltip

_ADDON = Path(__file__).resolve().parent
if str(_ADDON) not in sys.path:
    sys.path.insert(0, str(_ADDON))

from oume.persistence import JsonStore
from oume.session import StudyEngine


def _profile_path() -> Path:
    configured = mw.addonManager.getConfig(__name__) or {}
    raw = configured.get("profile_path") or os.environ.get("OUME_PROFILE", "")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local/share/oume/anki-profile.json"


def _engine() -> StudyEngine:
    config = mw.addonManager.getConfig(__name__) or {}
    engine = StudyEngine(JsonStore(_profile_path()))
    engine.cfg.max_daily_new_cards = int(config.get("daily_new_cap", 8))
    return engine


def _cards_in_deck(deck_name: str) -> list[dict]:
    deck_id = mw.col.decks.id_for_name(deck_name)
    if deck_id is None:
        return []
    items = []
    for card_id in mw.col.find_cards(f'deck:"{deck_name}"'):
        card = mw.col.get_card(card_id)
        note = card.note()
        fields = list(note.values())
        names = [field["name"] for field in note.note_type()["flds"]]
        values = dict(zip(names, fields))
        front = values.get("Word") or (fields[0] if fields else "")
        back = values.get("Meaning") or (fields[1] if len(fields) > 1 else "")
        reading = values.get("Reading", "")
        gloss = values.get("Gloss") or back
        example = values.get("Example") or values.get("Context") or (fields[2] if len(fields) > 2 else "")
        items.append({
            "card_id": str(card.id),
            "note_id": str(note.id),
            "deck_id": int(deck_id),
            "front": front,
            "back": back,
            "reading": reading,
            "gloss": gloss,
            "example": example,
            "context": example,
            "level": values.get("Level", ""),
        })
    return items


class StudyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent or mw)
        self.setWindowTitle("OUME 学习")
        self.resize(760, 560)
        self.engine = _engine()
        self.started = 0.0
        self.current = None
        config = mw.addonManager.getConfig(__name__) or {}

        self.deck = QLineEdit(config.get("deck_name", "OUME"))
        self.budget = QSpinBox()
        self.budget.setRange(5, 120)
        self.budget.setValue(int(config.get("session_minutes", 25)))
        self.view = QTextBrowser()
        self.answer = QLineEdit()
        self.answer.setPlaceholderText("输入拼写、生成内容，或选项原文")
        self.status = QLabel("选择牌组后开始。其它牌组不会被这个调度器改写。")

        start = QPushButton("开始 / 下一题")
        submit = QPushButton("提交")
        hint = QPushButton("看了提示（记为 Again）")
        qconnect(start.clicked, self.next_item)
        qconnect(submit.clicked, lambda: self.submit(False))
        qconnect(hint.clicked, lambda: self.submit(True))
        qconnect(self.answer.returnPressed, lambda: self.submit(False))

        top = QHBoxLayout()
        top.addWidget(QLabel("牌组"))
        top.addWidget(self.deck, 1)
        top.addWidget(QLabel("分钟"))
        top.addWidget(self.budget)
        top.addWidget(start)
        buttons = QHBoxLayout()
        buttons.addWidget(self.answer, 1)
        buttons.addWidget(submit)
        buttons.addWidget(hint)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.view, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        qconnect(self.timer.timeout, self.next_item)

    def next_item(self):
        deck_name = self.deck.text().strip()
        if mw.col.decks.id_for_name(deck_name) is None:
            QMessageBox.warning(self, "OUME", f"找不到牌组：{deck_name}")
            return
        self.timer.stop()
        added = self.engine.enroll(_cards_in_deck(deck_name))
        item = self.engine.next_item(budget_min=float(self.budget.value()))
        self.current = item
        resume = resume_policy(item)
        if item["action"] == "suspend":
            minutes = max(1, round(item["wait_min"]))
            self.view.setHtml(f"<h2>离开屏幕</h2><p>{html.escape(item['message'])}</p><p>到点后点「开始 / 下一题」，不要停在这个窗口。</p>")
            self.status.setText(f"约 {minutes} 分钟后可继续。新登记 {added} 张。")
            self.started = 0.0
            return
        if item["action"] == "done":
            self.view.setHtml("<h2>这一预算内没有待学卡片</h2>")
            self.status.setText(self._status_line())
            self.started = 0.0
            return
        self.started = time.time() if resume["clock_running"] else 0.0
        self.view.setHtml(_render(item))
        self.answer.clear()
        self.status.setText(self._status_line())

    def submit(self, used_hint: bool):
        if not self.current or self.current.get("action") != "review":
            return
        typed = self.answer.text().strip()
        prompt = self.current["prompt"]
        accepted = _accepted(self.current)
        scored = score_response(prompt, typed, accepted, used_hint)
        if scored["status"] != "scored":
            self.status.setText("选择题需要点选或完整输入其中一个选项。")
            return
        elapsed = int((time.time() - self.started) * 1000) if self.started else 500
        result = self.engine.submit(
            self.current["token"],
            scored["correct"],
            max(500, elapsed),
            typed=typed,
            choice=scored["choice"],
            used_hint=used_hint,
            confidence=3,
        )
        shown = " / ".join(accepted)
        self.status.setText(f"{scored['verdict']} · {result['grade']} · 下一状态 {result['state']} · 答案：{shown}")
        self.current = None
        QTimer.singleShot(700, self.next_item)

    def _status_line(self) -> str:
        state = self.engine.status()
        return (
            f"已复习 {state['reviews']} · 校准 θ {state['theta_effective']} "
            f"(不确定度 {state['posterior_sd']}) · 疲劳 {state['fatigue']}"
        )


def resume_policy(item: dict) -> dict:
    """Incubation is a break. Only a visible prompt starts the response clock."""
    if item.get("action") == "review":
        return {"timer_ms": None, "clock_running": True}
    return {"timer_ms": None, "clock_running": False}


def score_response(prompt: dict, typed: str, accepted: list[str], used_hint: bool) -> dict:
    """Score one visible response without opening Anki."""
    options = prompt.get("options", [])
    choice = typed if typed in options else ""
    if options and not choice:
        return {"status": "need-choice"}
    revealed = bool(prompt.get("reveal_answer"))
    correct = (not used_hint) and (
        revealed or any(_matches(choice or typed, answer) for answer in accepted)
    )
    if used_hint or not correct:
        verdict = "记为失败"
    elif revealed:
        verdict = "已看答案并确认，不计为回忆"
    else:
        verdict = "记对"
    return {"correct": correct, "choice": choice, "verdict": verdict, "status": "scored"}


def _expected(item: dict) -> str:
    card = _engine().data["cards"][item["card_id"]]
    if item.get("modality") in ("AUDIO_SPELL", "CONTEXT_GEN"):
        return _plain(card.get("front", ""))
    return _plain(card.get("back", ""))


def _accepted(item: dict) -> list[str]:
    card = _engine().data["cards"][item["card_id"]]
    if item.get("modality") in ("AUDIO_SPELL", "CONTEXT_GEN"):
        values = [card.get("front", ""), card.get("reading", "")]
    else:
        values = [card.get("back", "")]
    answers = []
    for value in values:
        text = _plain(value)
        if text and text not in answers:
            answers.append(text)
    return answers


def _matches(typed: str, expected: str) -> bool:
    return typed.casefold().strip() == expected.casefold().strip() and bool(expected)


def _plain(value: str) -> str:
    from oume.prompts import plain
    return plain(value)


def _render(item: dict) -> str:
    prompt = item["prompt"]
    options = "".join(f"<li>{html.escape(option)}</li>" for option in prompt.get("options", []))
    reveal = f"<p><b>先看答案：</b>{html.escape(prompt.get('reveal_answer', ''))}</p>" if prompt.get("reveal_answer") else ""
    return (
        f"<p>{html.escape(prompt['instruction'])}</p>"
        f"<h2>{html.escape(prompt.get('shown', ''))}</h2>{reveal}<ol>{options}</ol>"
        f"<p>模态 {html.escape(item['modality'])} · 状态 {html.escape(item['state'])}</p>"
    )


def open_study() -> None:
    if not mw.col:
        tooltip("先打开一个 Anki 配置档。")
        return
    StudyDialog().exec()


def setup_menu() -> None:
    action = mw.form.menuTools.addAction("OUME Study")
    qconnect(action.triggered, open_study)


gui_hooks.main_window_did_init.append(setup_menu)
