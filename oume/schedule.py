"""Map OUME's next interval onto Anki's four answer buttons.

Anki still shows the card and owns the queue. This module only answers:
given the button the learner presses, when should that card return?
"""

from __future__ import annotations

import copy
from typing import Optional

from oume.scheduler_core import (
    Card,
    CardState,
    EngineConfig,
    Grade,
    Modality,
    UserState,
    execute_review_transition,
)

PROFILE = "OUME-Pragmatic"
DAY_MINUTES = 1440.0


def scheduling_intervals(
    card: Card,
    user: UserState,
    now_minutes: float,
    day: int,
    modality: Modality = Modality.VISUAL_CHOICE,
    cfg: Optional[EngineConfig] = None,
) -> dict[str, int]:
    """Seconds until the card is due again, for Again, Hard, Good and Easy.

    A same-day learning pulse stays in seconds. A cross-day review is at
    least one day, because that is the resolution of OUME's scheduled day.
    """
    cfg = cfg or EngineConfig()
    intervals = {}
    for grade in Grade:
        trial_card = copy.deepcopy(card)
        trial_user = copy.deepcopy(user)
        execute_review_transition(
            trial_card, now_minutes, day, grade, modality, trial_user, cfg, PROFILE
        )
        intervals[grade.name.lower()] = _seconds_until_due(trial_card, now_minutes, day)
    return intervals


def _seconds_until_due(card: Card, now_minutes: float, day: int) -> int:
    if card.state == CardState.MICRO_PULSING:
        wait_min = max(1.0, card.optimal_due_at - now_minutes)
        return int(round(wait_min * 60.0))
    wait_days = max(1, int(card.scheduled_due_day) - int(day))
    return int(wait_days * DAY_MINUTES * 60.0)
