"""Replace FSRS intervals with OUME while keeping Anki's reviewer.

The card, template and Again/Hard/Good/Easy buttons stay Anki's. Before
those buttons are drawn, their next intervals are rewritten from the
OUME-Pragmatic transition. The same transition is recorded when the button
is pressed, so the next review does not drift back to FSRS.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from aqt import gui_hooks, mw
from aqt.qt import qconnect
from aqt.utils import showInfo, tooltip

_ADDON = Path(__file__).resolve().parent
if str(_ADDON) not in sys.path:
    sys.path.insert(0, str(_ADDON))

from oume.persistence import JsonStore
from oume.schedule import scheduling_intervals
from oume.scheduler_core import Grade, Modality, execute_review_transition
from oume.session import StudyEngine

_RATINGS = {1: "again", 2: "hard", 3: "good", 4: "easy"}


def config() -> dict:
    return mw.addonManager.getConfig(__name__) or {}


def profile_path() -> Path:
    raw = config().get("profile_path") or os.environ.get("OUME_PROFILE", "")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local/share/oume/anki-profile.json"


def engine() -> StudyEngine:
    study = StudyEngine(JsonStore(profile_path()))
    study.cfg.max_daily_new_cards = int(config().get("daily_new_cap", 8))
    return study


def deck_enabled(card) -> bool:
    if not config().get("enabled", True) or card is None:
        return False
    wanted = str(config().get("deck_name", "")).strip()
    if not wanted:
        return True
    return mw.col.decks.name(card.current_deck_id()) == wanted


def card_content(card) -> dict:
    note = card.note()
    fields = list(note.values())
    names = [field["name"] for field in note.note_type()["flds"]]
    values = dict(zip(names, fields))
    front = values.get("Word") or (fields[0] if fields else "")
    back = values.get("Meaning") or (fields[1] if len(fields) > 1 else "")
    example = values.get("Example") or values.get("Context") or (fields[2] if len(fields) > 2 else "")
    return {
        "card_id": str(card.id),
        "note_id": str(note.id),
        "deck_id": int(card.current_deck_id()),
        "front": front,
        "back": back,
        "reading": values.get("Reading", ""),
        "gloss": values.get("Gloss") or back,
        "example": example,
        "context": example,
        "level": values.get("Level", ""),
    }


def apply_intervals(states, intervals: dict[str, int]) -> None:
    """Write one delay into each of Anki's four scheduling states."""
    for rating, seconds in intervals.items():
        _set_delay(getattr(states, rating), seconds)


def _set_delay(state, seconds: int) -> None:
    normal = state.normal
    if seconds < 86400:
        learning = normal.learning
        learning.scheduled_secs = max(60, int(seconds))
        learning.remaining_steps = 1
        return
    review = normal.review
    review.scheduled_days = max(1, int(round(seconds / 86400)))


def prepare_answer_buttons(buttons, reviewer, card):
    """Rewrite Anki's four next states, leaving the buttons themselves unchanged."""
    if reviewer is None or card is None or not deck_enabled(card):
        return buttons
    try:
        states = reviewer.get_scheduling_states()
        study = engine()
        study.enroll([card_content(card)])
        now = time.time()
        study.ensure_session(now)
        intervals = scheduling_intervals(
            study._card(str(card.id)),
            study._user(),
            study.now_minutes(now),
            study.learning_day(now),
            Modality.VISUAL_CHOICE,
            study.cfg,
        )
        apply_intervals(states, intervals)
    except Exception:
        return buttons
    return buttons


def record_answer(reviewer, card, ease: int) -> None:
    if not deck_enabled(card) or ease not in _RATINGS:
        return
    study = engine()
    study.enroll([card_content(card)])
    now = time.time()
    study.ensure_session(now)
    oume_card = study._card(str(card.id))
    user = study._user()
    grade = Grade(ease)
    taken = max(500, int(card.time_taken(capped=False)))
    execute_review_transition(
        oume_card,
        study.now_minutes(now),
        study.learning_day(now),
        grade,
        Modality.VISUAL_CHOICE,
        user,
        study.cfg,
        "OUME-Pragmatic",
    )
    study._fatigue(user, Modality.VISUAL_CHOICE, grade, taken, study.now_minutes(now))
    study._write_user(user)
    study._write_card(oume_card)
    study.data["stats"]["reviews"] += 1
    study.data["stats"]["correct"] += int(grade != Grade.AGAIN)
    study.save()


def show_status() -> None:
    if not mw.col:
        tooltip("先打开一个 Anki 配置档。")
        return
    wanted = str(config().get("deck_name", "")).strip() or "全部牌组"
    state = engine().status()
    showInfo(
        "OUME 已接管所选牌组的下次间隔。复习仍使用 Anki 的卡片和四个按钮。\n\n"
        f"范围：{wanted}\n"
        f"已记录复习：{state['reviews']}\n"
        f"有效记忆尺度：{state['theta_effective']}\n"
        f"疲劳：{state['fatigue']}"
    )


def setup_menu() -> None:
    action = mw.form.menuTools.addAction("OUME 调度状态")
    qconnect(action.triggered, show_status)


gui_hooks.main_window_did_init.append(setup_menu)
gui_hooks.reviewer_will_init_answer_buttons.append(prepare_answer_buttons)
gui_hooks.reviewer_did_answer_card.append(record_answer)
