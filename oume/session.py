"""Persistence-backed study session for the OUME-Pragmatic scheduler."""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

from oume.persistence import JsonStore
from oume.prompts import build_prompt
from oume.scheduler_core import (
    Card,
    CardState,
    EngineConfig,
    Grade,
    Modality,
    UserState,
    calculate_initial_stability,
    compute_learning_day,
    evaluate_response_grade,
    execute_review_transition,
    get_effective_theta_calib,
    get_scaffolding_modality,
    handle_session_transition,
    memory_retrievability,
    pick_next_card,
    predict_p_correct,
    update_theta_calib_cumulative_map,
)

PROFILE = "OUME-Pragmatic"
_TAG = re.compile(r"<[^>]+>")


def _strip(text: str) -> str:
    return _TAG.sub(" ", text or "").replace("&nbsp;", " ").strip()


def grade_for_study(
    is_correct: bool,
    used_hint: bool,
    rt_ms: int,
    modality: Modality,
    confidence: int,
    is_guided_primer: bool,
    user: UserState,
    cfg: EngineConfig,
    record_rt: bool,
) -> Grade:
    """Section 4.5 thresholds for a live session.

    The simulator keeps its tested 0.85 / confidence-3 operator so published
    v1.15 numbers stay reproducible. A study session uses the written rule:
    slow means z > 1, Easy needs confidence of at least 4, and a revealed
    primer is not scored as a slow retrieval.
    """
    grade = evaluate_response_grade(
        is_correct=is_correct,
        used_hint=used_hint,
        rt_ms=rt_ms,
        modality=modality,
        confidence=confidence,
        is_guided_primer=is_guided_primer,
        user=user,
        cfg=cfg,
        rubric_kappa=None if int(modality) == int(Modality.CONTEXT_GEN) else 1.0,
        record_rt=record_rt,
    )
    if not is_correct or used_hint:
        return grade
    if grade == Grade.EASY and confidence < 4:
        grade = Grade.GOOD
    if grade != Grade.HARD:
        return grade
    if is_guided_primer or _response_z(user, modality, rt_ms) <= 1.0:
        return Grade.GOOD
    return grade


def _response_z(user: UserState, modality: Modality, rt_ms: int) -> float:
    default_mu = {1: math.log(8.0), 2: math.log(14.0), 3: math.log(32.0)}[int(modality)]
    history = user.rt_stats_by_mod.get(int(modality), [])
    mu, sigma = default_mu, 0.35
    if len(history) >= 5:
        mean = sum(history) / len(history)
        variance = sum((item - mean) ** 2 for item in history) / max(1, len(history) - 1)
        weight = min(1.0, len(history) / 15.0)
        mu = (1.0 - weight) * default_mu + weight * mean
        sigma = (1.0 - weight) * 0.35 + weight * max(0.15, math.sqrt(variance))
    return (math.log(max(0.5, rt_ms / 1000.0)) - mu) / max(0.15, sigma)


class StudyEngine:
    """One learner, one deck, one learning day at a time.

    Time is measured in minutes from a fixed epoch so the reference scheduler
    can be called without changing its units. Cards that Anki already owns
    stay untouched until this engine is asked to enroll them.
    """

    def __init__(self, store: JsonStore, cfg: Optional[EngineConfig] = None):
        self.store = store
        self.cfg = cfg or EngineConfig()
        self.data = store.load()
        self.data.setdefault("schema", 1)
        self.data.setdefault("epoch_unix", int(time.time()))
        self.data.setdefault("cards", {})
        self.data.setdefault("user", {})
        self.data.setdefault("log", [])
        self.data.setdefault("queue", [])
        self.data.setdefault("probes", [])
        self.data.setdefault("notices", [])
        self.data.setdefault("stats", {"reviews": 0, "correct": 0, "introduced": 0})
        if not self.data["user"]:
            self.data["user"] = {
                "user_id": "local",
                "F": 0.0,
                "V": 1.0,
                "theta_calib": 0.0,
                "theta_calib_prev": 0.0,
                "psi_enc": 1.0,
                "last_action_t": 0.0,
                "cum_fisher_info": 0.0,
                "theta_posterior_sd": 1.0,
                "rt_stats_by_mod": {"1": [], "2": [], "3": []},
                "total_ans_count": 0,
                "last_session_day": None,
                "cum_probe_obs": [],
                "last_calibrated_day": None,
                "tz_offset_hours": _local_offset_hours(),
            }
        self._transitioned_day: Optional[int] = self.data["user"].get("last_transition_day")

    def save(self) -> None:
        self.store.save(self.data)

    def now_minutes(self, unix_sec: Optional[float] = None) -> float:
        stamp = time.time() if unix_sec is None else unix_sec
        return max(0.0, (stamp - self.data["epoch_unix"]) / 60.0)

    def learning_day(self, unix_sec: Optional[float] = None) -> int:
        stamp = time.time() if unix_sec is None else unix_sec
        user = self.data["user"]
        return compute_learning_day(
            stamp,
            tz_offset_hours=float(user.get("tz_offset_hours", _local_offset_hours())),
            prev_learning_day=user.get("last_session_day"),
            day_boundary_hour_local=self.cfg.day_boundary_hour_local,
        )

    def enroll(self, items: Sequence[dict]) -> int:
        """Register notes that are not already scheduled by this engine."""
        added = 0
        for item in items:
            card_id = str(item["card_id"])
            if card_id in self.data["cards"]:
                row = self.data["cards"][card_id]
                for key, value in _content_fields(item).items():
                    if value or key not in row:
                        row[key] = value
                continue
            card = Card(card_id=card_id)
            row = _card_to_row(card)
            row.update(_content_fields(item))
            self.data["cards"][card_id] = row
            added += 1
        self.save()
        return added

    def ensure_session(self, unix_sec: Optional[float] = None) -> dict:
        day = self.learning_day(unix_sec)
        if self._transitioned_day == day:
            return {"learning_day": day, "already_open": True}
        user = self._user()
        now = self.now_minutes(unix_sec)
        stats = handle_session_transition(self._cards(), user, now, day, self.cfg)
        self._write_user(user)
        self._write_cards()
        self.data["user"]["last_session_day"] = day
        self.data["user"]["last_transition_day"] = day
        self._transitioned_day = day
        self.save()
        return {"learning_day": day, **{k: int(v) for k, v in stats.items()}}

    def calibrate_previous_day(self) -> Optional[float]:
        """Apply yesterday's review probes once, then clear the daily buffer."""
        user_row = self.data["user"]
        day = user_row.get("last_session_day")
        if not self.data["probes"] or user_row.get("last_calibrated_day") == day:
            return user_row.get("theta_calib")
        user = self._user()
        user.cum_probe_obs = [tuple(x) for x in user_row.get("cum_probe_obs", [])]
        theta = update_theta_calib_cumulative_map(user, [tuple(x) for x in self.data["probes"]], self.cfg)
        user_row["cum_probe_obs"] = [list(x) for x in user.cum_probe_obs]
        user_row["theta_calib"] = user.theta_calib
        user_row["cum_fisher_info"] = user.cum_fisher_info
        user_row["theta_posterior_sd"] = user.theta_posterior_sd
        user_row["last_calibrated_day"] = day
        self.data["probes"] = []
        self.save()
        return theta

    def next_item(
        self,
        budget_min: float = 25.0,
        unix_sec: Optional[float] = None,
        introduced_today: Optional[int] = None,
    ) -> dict:
        self.ensure_session(unix_sec)
        self.calibrate_previous_day()
        now = self.now_minutes(unix_sec)
        day = self.learning_day(unix_sec)
        cards = self._cards()
        user = self._user()
        introduced = self._introduced_today(day) if introduced_today is None else introduced_today
        picked, wake, rescued = pick_next_card(
            cards,
            user,
            now,
            day,
            now + budget_min,
            budget_min,
            self.cfg,
            self._warmup_anchor(day, now),
            PROFILE,
            introduced,
            return_rescue_flag=True,
        )
        if picked is None:
            wait = max(0.0, (wake or now) - now)
            if wait >= self.cfg.micro_session_suspend_threshold_min:
                notice = {
                    "kind": "leave-and-notify",
                    "wait_min": round(wait, 2),
                    "ready_at_unix": self.data["epoch_unix"] + (wake or now) * 60.0,
                    "message": "这一轮微脉冲还在间隔中。离开屏幕休息，到点再回来即可。",
                }
                self.data["notices"].append(notice)
                self.data["notices"] = self.data["notices"][-30:]
                self.save()
                return {"action": "suspend", **notice}
            return {"action": "done", "wait_min": round(wait, 2)}

        row = self.data["cards"][picked.card_id]
        modality = get_scaffolding_modality(picked, PROFILE)
        guided = picked.state in (CardState.NEW, CardState.MICRO_PULSING) and picked.pulse_count == 0
        token = uuid.uuid4().hex
        self.data["queue"] = [{
            "token": token,
            "card_id": picked.card_id,
            "modality": int(modality),
            "guided": guided,
            "soft_rescue": bool(rescued),
            "shown_unix": time.time(),
            "state_before": picked.state.value,
            "s_before": picked.S,
            "day": day,
        }]
        self.save()
        return {
            "action": "review",
            "token": token,
            "card_id": picked.card_id,
            "modality": modality.name,
            "modality_id": int(modality),
            "guided": guided,
            "soft_rescue": bool(rescued),
            "state": picked.state.value,
            "pulse_count": picked.pulse_count,
            "prompt": build_prompt(row, modality, guided, self._distractors(picked.card_id)),
            "prediction": self._prediction(picked, modality, now),
        }

    def submit(
        self,
        token: str,
        correct: bool,
        rt_ms: int,
        typed: str = "",
        choice: str = "",
        used_hint: bool = False,
        confidence: int = 3,
    ) -> dict:
        pending = next((q for q in self.data["queue"] if q["token"] == token), None)
        if pending is None:
            raise KeyError("review token is not pending")
        card = self._card(pending["card_id"])
        user = self._user()
        modality = Modality(int(pending["modality"]))
        revealed = bool(pending["guided"]) and modality == Modality.VISUAL_CHOICE
        grade = grade_for_study(
            is_correct=correct,
            used_hint=used_hint,
            rt_ms=max(500, int(rt_ms)),
            modality=modality,
            confidence=confidence,
            is_guided_primer=bool(pending["guided"]),
            user=user,
            cfg=self.cfg,
            record_rt=not revealed and not used_hint and correct,
        )
        was_review = card.state == CardState.REVIEW
        reviewed_at = card.last_review_t
        now = self.now_minutes()
        day = int(pending["day"])
        delta_e = execute_review_transition(card, now, day, grade, modality, user, self.cfg, PROFILE)
        if was_review and not used_hint:
            delta_days = max(1e-3, (now - (reviewed_at if reviewed_at is not None else now)) / 1440.0)
            outcome = 1.0 if correct else 0.0
            self.data["probes"].append([delta_days, max(0.1, card.S or 0.1), int(modality), outcome])
        self._fatigue(user, modality, grade, rt_ms, now)
        self._write_user(user)
        self._write_card(card)
        self.data["queue"] = [q for q in self.data["queue"] if q["token"] != token]
        self.data["stats"]["reviews"] += 1
        self.data["stats"]["correct"] += int(bool(correct) and not used_hint)
        if pending["state_before"] == CardState.NEW.value:
            self.data["stats"]["introduced"] += 1
        event = {
            "id": hashlib.sha256(token.encode()).hexdigest()[:16],
            "card_id": card.card_id,
            "day": day,
            "modality": int(modality),
            "grade": int(grade),
            "correct": bool(correct),
            "rt_ms": int(rt_ms),
            "typed": typed[:500],
            "choice": choice[:200],
            "state_after": card.state.value,
            "delta_e": delta_e,
            "unix": time.time(),
        }
        self.data["log"].append(event)
        self.data["log"] = self.data["log"][-5000:]
        self.save()
        return {
            "grade": grade.name,
            "state": card.state.value,
            "stability_days": card.S,
            "next_pulse_min": None if card.state != CardState.MICRO_PULSING else self.cfg.optimal_interval_min,
            "due_day": card.scheduled_due_day,
            "answer": self.data["cards"][card.card_id].get("back", ""),
        }

    def status(self) -> dict:
        cards = list(self.data["cards"].values())
        by_state: Dict[str, int] = {}
        for row in cards:
            by_state[row["state"]] = by_state.get(row["state"], 0) + 1
        user = self.data["user"]
        reviews = max(1, int(self.data["stats"]["reviews"]))
        return {
            "profile": PROFILE,
            "cards": len(cards),
            "by_state": by_state,
            "theta_calib": round(float(user.get("theta_calib", 0.0)), 4),
            "theta_effective": round(get_effective_theta_calib(self._user(), self.cfg, PROFILE), 4),
            "posterior_sd": round(float(user.get("theta_posterior_sd", 1.0)), 4),
            "fatigue": round(float(user.get("F", 0.0)), 4),
            "accuracy": round(self.data["stats"]["correct"] / reviews, 4) if self.data["stats"]["reviews"] else None,
            "reviews": self.data["stats"]["reviews"],
            "introduced": self.data["stats"]["introduced"],
            "pending_probes": len(self.data["probes"]),
            "last_notice": self.data["notices"][-1] if self.data["notices"] else None,
        }

    def _prediction(self, card: Card, modality: Modality, now: float) -> Optional[float]:
        if card.S is None or card.last_review_t is None:
            return None
        theta = get_effective_theta_calib(self._user(), self.cfg, PROFILE)
        r_mem = memory_retrievability((now - card.last_review_t) / 1440.0, card.S, math.exp(theta))
        return round(predict_p_correct(r_mem, int(modality), self.cfg), 4)

    def _fatigue(self, user: UserState, modality: Modality, grade: Grade, rt_ms: int, now: float) -> None:
        idle = max(0.0, now - user.last_action_t - rt_ms / 60000.0)
        user.F *= math.exp(-idle / self.cfg.tau_rest)
        load = self.cfg.c_mod.get(int(modality), 1.0) * self.cfg.c_grade.get(int(grade), 1.0)
        user.F = min(0.80, user.F + self.cfg.delta_F * load * ((rt_ms / 1000.0) / self.cfg.rt_ref))
        arousal = {1: 0.65, 2: 0.85, 3: 1.00, 4: 1.15}[int(grade)]
        user.V = max(0.60, min(1.30, (1.0 - self.cfg.omega_v) * user.V + self.cfg.omega_v * arousal))
        user.last_action_t = now

    def _distractors(self, card_id: str) -> List[str]:
        pool = [
            _strip(row.get("back", ""))
            for key, row in self.data["cards"].items()
            if key != card_id and _strip(row.get("back", ""))
        ]
        pool = list(dict.fromkeys(pool))
        digest = hashlib.sha256(card_id.encode()).digest()
        start = digest[0] % max(1, len(pool))
        return [pool[(start + i) % len(pool)] for i in range(min(3, len(pool)))]

    def _warmup_anchor(self, day: int, now: float) -> float:
        anchors = self.data.setdefault("warmup_anchor_min", {})
        key = str(day)
        if key not in anchors:
            anchors[key] = now
        return float(anchors[key])

    def _introduced_today(self, day: int) -> int:
        return sum(1 for row in self.data["cards"].values() if row.get("intro_day") == day)

    def _cards(self) -> List[Card]:
        self._live = {key: _row_to_card(row) for key, row in self.data["cards"].items()}
        return list(self._live.values())

    def _card(self, card_id: str) -> Card:
        self._cards()
        return self._live[card_id]

    def _write_cards(self) -> None:
        for card in self._live.values():
            self._write_card(card)

    def _write_card(self, card: Card) -> None:
        preserved = {
            k: self.data["cards"][card.card_id].get(k)
            for k in ("front", "back", "reading", "gloss", "example", "context", "level", "audio", "note_id", "deck_id")
        }
        row = _card_to_row(card)
        row.update(preserved)
        if card.intro_day is None and card.state != CardState.NEW:
            row["intro_day"] = self.data["user"].get("last_session_day")
        self.data["cards"][card.card_id] = row

    def _user(self) -> UserState:
        raw = dict(self.data["user"])
        stats = {int(k): list(v) for k, v in raw.get("rt_stats_by_mod", {}).items()}
        return UserState(
            user_id=raw.get("user_id", "local"),
            F=float(raw.get("F", 0.0)),
            V=float(raw.get("V", 1.0)),
            theta_calib=float(raw.get("theta_calib", 0.0)),
            theta_calib_prev=float(raw.get("theta_calib_prev", 0.0)),
            psi_enc=float(raw.get("psi_enc", 1.0)),
            last_action_t=float(raw.get("last_action_t", 0.0)),
            cum_fisher_info=float(raw.get("cum_fisher_info", 0.0)),
            theta_posterior_sd=float(raw.get("theta_posterior_sd", 1.0)),
            rt_stats_by_mod=stats,
            total_ans_count=int(raw.get("total_ans_count", 0)),
            last_session_day=raw.get("last_session_day"),
            cum_probe_obs=[tuple(item) for item in raw.get("cum_probe_obs", [])],
        )

    def _write_user(self, user: UserState) -> None:
        row = self.data["user"]
        row.update({
            "F": user.F,
            "V": user.V,
            "theta_calib": user.theta_calib,
            "theta_calib_prev": user.theta_calib_prev,
            "psi_enc": user.psi_enc,
            "last_action_t": user.last_action_t,
            "cum_fisher_info": user.cum_fisher_info,
            "theta_posterior_sd": user.theta_posterior_sd,
            "rt_stats_by_mod": {str(k): v for k, v in user.rt_stats_by_mod.items()},
            "total_ans_count": user.total_ans_count,
            "last_session_day": user.last_session_day,
            "cum_probe_obs": [list(item) for item in user.cum_probe_obs],
        })


def _content_fields(item: dict) -> dict:
    return {
        "front": item.get("front", ""),
        "back": item.get("back", ""),
        "reading": item.get("reading", ""),
        "gloss": item.get("gloss", ""),
        "example": item.get("example", item.get("context", "")),
        "context": item.get("context", ""),
        "level": item.get("level", ""),
        "audio": item.get("audio", ""),
        "note_id": item.get("note_id"),
        "deck_id": item.get("deck_id"),
    }


def _card_to_row(card: Card) -> dict:
    row = asdict(card)
    row["state"] = card.state.value
    return row


def _row_to_card(row: dict) -> Card:
    state = CardState(row["state"])
    return Card(
        card_id=row["card_id"],
        state=state,
        A=float(row.get("A", 0.0)),
        R_des=float(row.get("R_des", 0.0)),
        E=float(row.get("E", 0.0)),
        m_last=row.get("m_last"),
        micro_streak=int(row.get("micro_streak", 0)),
        pulse_count=int(row.get("pulse_count", 0)),
        S=row.get("S"),
        D=float(row.get("D", 5.0)),
        refractory_until=float(row.get("refractory_until", 0.0)),
        optimal_due_at=float(row.get("optimal_due_at", 0.0)),
        scheduled_due_day=int(row.get("scheduled_due_day", 0)),
        lapse_count=int(row.get("lapse_count", 0)),
        last_review_t=row.get("last_review_t"),
        intro_day=row.get("intro_day"),
        graduated_day=row.get("graduated_day"),
        graduated_at_pulse=row.get("graduated_at_pulse"),
        days_in_flight=int(row.get("days_in_flight", 0)),
        S_pre_lapse=row.get("S_pre_lapse"),
        is_session_deferred=bool(row.get("is_session_deferred", False)),
    )


def _local_offset_hours() -> float:
    return -time.altzone / 3600.0 if time.localtime().tm_isdst else -time.timezone / 3600.0


def initial_stability_report(card: Card, cfg: EngineConfig) -> Optional[float]:
    if card.state != CardState.SLEEP_PENDING:
        return card.S
    return calculate_initial_stability(card.D, card.E, cfg)
