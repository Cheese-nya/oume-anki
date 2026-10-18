"""Scheduler core used by the OUME Anki add-on.

Extracted from the v1.15 reference simulator. Experiment runners, synthetic
students and benchmark statistics stay in the research tree and are not part
of the installable add-on.
"""

import copy
import math
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union

# ==============================================================================
# 1. Enums and Data Models
# ==============================================================================

class CardState(str, Enum):
    NEW = "NEW"
    MICRO_PULSING = "MICRO_PULSING"
    SLEEP_PENDING = "SLEEP_PENDING"
    REVIEW = "REVIEW"
    RELEARNING = "RELEARNING"
    LEECH_SUSPENDED = "LEECH_SUSPENDED"

class Modality(int, Enum):
    VISUAL_CHOICE = 1  # Guided Primer / 4-choice recognition (guess ~25%, slip ~2%)
    AUDIO_SPELL = 2    # Typing / spelling (guess ~0%, slip ~3%)
    CONTEXT_GEN = 3    # Cloze / free generation (guess ~0%, slip ~5%)

class Grade(int, Enum):
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4

@dataclass
class Card:
    card_id: str
    state: CardState = CardState.NEW
    # Fast & slow intra-day variables
    A: float = 0.0          # Kinase activation [0, 1]
    R_des: float = 0.0      # Receptor desensitization [0, 1]
    E: float = 0.0          # CREB transcription accumulation
    m_last: Optional[int] = None
    micro_streak: int = 0   # Consecutive successful micro-pulses
    pulse_count: int = 0    # Total pulses in current cycle
    # Long-term cortical variables (S is None for NEW cards to prevent zero division)
    S: Optional[float] = None # Stability (days), None until graduated
    D: float = 5.0          # Difficulty [1, 10]
    # Scheduling timestamps (minutes from simulation t=0)
    refractory_until: float = 0.0
    optimal_due_at: float = 0.0
    scheduled_due_day: int = 0
    # Counters & stats
    lapse_count: int = 0
    total_reviews: int = 0
    total_study_seconds: float = 0.0
    last_review_t: Optional[float] = None
    intro_day: Optional[int] = None
    intro_t: Optional[float] = None
    graduated_at_pulse: Optional[int] = None
    graduated_day: Optional[int] = None
    days_in_flight: int = 0
    had_opportunity_for_3_pulses: bool = False
    is_session_deferred: bool = False
    # Pre-registered blind delayed-test assignment (v1.13, C-82)
    blind_test_horizon: Optional[str] = None
    blind_test_modality: Optional[int] = None
    # Read-only telemetry mirrors / legacy test initialization hooks (C-77)
    true_D: Optional[float] = None
    true_A: float = 0.0
    true_R_des: float = 0.0
    true_E: float = 0.0
    true_S: Optional[float] = None
    true_pre_lapse_S: Optional[float] = None
    true_engram_quality: float = 0.35
    true_has_gen_engram: bool = False
    true_last_t: Optional[float] = None
    true_m_last: Optional[int] = None
    S_pre_lapse: Optional[float] = None
    min_isi_min: float = 10.0
    overdue_applied: bool = False

    def __post_init__(self):
        if self.true_D is None:
            self.true_D = self.D

@dataclass
class UserState:
    user_id: str
    F: float = 0.0               # Continuous Fatigue [0, 0.8]
    V: float = 1.0               # Arousal [0.6, 1.3]
    theta_calib: float = 0.0     # Logit calibration offset [-1.2, 1.2]
    theta_calib_prev: float = 0.0 # d-1 logit calibration for non-circular probes
    psi_enc: float = 1.0         # Individual encoding rate prior [0.70, 1.40]
    last_action_t: float = 0.0
    Q_calib: float = 1.0         # Backward compatible exponential scale
    Q_calib_prev: float = 1.0    # Backward compatible exponential scale
    # v1.15 practical production engine fields (§4.5, §4.6, §11.1, G6)
    cum_probe_obs: List[Tuple[float, float, int, float]] = field(default_factory=list)
    cum_fisher_info: float = 0.0
    theta_posterior_sd: float = 1.0
    rt_stats_by_mod: Dict[int, List[float]] = field(default_factory=lambda: {1: [], 2: [], 3: []})
    total_ans_count: int = 0
    last_session_day: Optional[int] = None

    def __post_init__(self):
        if self.theta_calib != 0.0 and self.Q_calib == 1.0:
            self.Q_calib = math.exp(self.theta_calib)
        elif self.Q_calib != 1.0 and self.theta_calib == 0.0:
            self.theta_calib = math.log(max(0.1, self.Q_calib))
        if self.theta_calib_prev != 0.0 and self.Q_calib_prev == 1.0:
            self.Q_calib_prev = math.exp(self.theta_calib_prev)
        elif self.Q_calib_prev != 1.0 and self.theta_calib_prev == 0.0:
            self.theta_calib_prev = math.log(max(0.1, self.Q_calib_prev))

@dataclass
class ReviewEvent:
    event_id: str
    card_id: str
    user_id: str
    timestamp: float        # minutes from t=0
    day: int
    modality: int
    grade: int
    rt_ms: int
    state_before: str
    state_after: str
    S_before: Optional[float]
    S_after: Optional[float]
    D_before: float
    D_after: float
    E_before: float
    E_after: float
    R_memory: float
    P_correct: float
    delta_E: float = 0.0

# ==============================================================================
# 2. Mathematical Engine Configuration (v1.11 / v1.15)
# ==============================================================================

@dataclass
class EngineConfig:
    # Kinase & desensitization
    tau_des: float = 8.0     # min
    tau_A: float = 22.0      # min
    gamma_A: float = 0.75
    gamma_des: float = 0.85
    # Transcription & Elastic Pulses
    tau_E: float = 240.0     # min
    Theta_CREB: float = 1.00
    alpha_base: float = 0.255 # recalibrated for dynamic fatigue
    alpha_res: float = 0.45
    optimal_interval_min: float = 10.0 # elastic baseline recommendation
    delta_t_soft: float = 3.0          # min (soft refractory threshold)
    phi_soft_damping: float = 0.416    # derived: 1 - 0.85 * exp(-3/8)
    max_wave_concurrency: int = 8      # Little's Law 10m window wave cap (C-75)
    max_daily_new_cards: int = 8       # Uniform daily new-card introduction quota across all arms (Section 5.1)
    # Modality & grades
    lambda_shift: float = 1.30
    lambda_gen: float = 1.55
    q_weights: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.80, 3: 1.00, 4: 1.15})
    # Long-term FSRS parameters
    D_init: float = 5.0
    w0: float = 3.2
    w1: float = 0.18
    beta_E: float = 0.80     # Concave transcription elasticity
    theta_gain: float = 2.10
    w2: float = 0.65
    w3: float = 0.22
    w4: float = 1.10
    # Normalized PE Gate G(1)=1 parity (C-66)
    w4b: float = 12.0
    R_gate: float = 0.35
    w_d: float = 0.10
    delta_d: float = 0.85
    w_lapse: float = 0.35
    w5: float = 0.20
    eta_lapse: float = 0.40
    R_target: float = 0.90
    S_min: float = 0.25
    S_max: float = 36500.0
    # Continuous Fatigue & Arousal (C-35)
    tau_rest: float = 45.0   # min
    delta_F: float = 0.005   # base fatigue increment per reference unit
    rt_ref: float = 15.0     # reference reaction time in seconds
    omega_v: float = 0.25
    c_mod: Dict[int, float] = field(default_factory=lambda: {1: 0.75, 2: 1.00, 3: 1.35})
    c_grade: Dict[int, float] = field(default_factory=lambda: {1: 1.40, 2: 1.20, 3: 1.00, 4: 0.70})
    # Observation Model with Modality-Disjoint Offsets (Scheme A single-path: S_eff carries calibration)
    modality_offsets: Dict[int, float] = field(default_factory=lambda: {1: 0.70, 2: 0.35, 3: 0.0})
    gamma_logit: float = 1.0
    enable_logit_calib: bool = False # False when Formula 7a already applies Q_calib = exp(theta_calib) (C-74)
    guess_rates: Dict[int, float] = field(default_factory=lambda: {1: 0.25, 2: 0.00, 3: 0.00})
    slip_rates: Dict[int, float] = field(default_factory=lambda: {1: 0.02, 2: 0.03, 3: 0.05})
    # Morning Probes & Bayesian Shrinkage (C-38, C-53, C-67)
    probe_shrinkage_n0: float = 3.0
    probe_kappa: float = 0.35
    # Relative MER Optimal Stopping (v1.8, C-54)
    lambda_stop_rel: float = 0.45 # relative marginal stability threshold (dimensionless)
    dt_cost: float = 0.50         # expected review time in minutes
    omega_F: float = 1.0          # fatigue cost magnification
    rho_overnight: float = 0.90   # overnight slow-variable freeze retention rate (prior)
    stale_timeout_new: int = 3
    stale_timeout_relearn: int = 7
    # v1.15 Practical Production Engine Parameters (§4.6, §11.1, G3, G6)
    theta_crlb_sd_gate: float = 0.20            # G6: CRLB SD threshold (I_total >= 25 for full weight)
    micro_session_suspend_threshold_min: float = 2.0 # G3: suspend into Leave-and-Notify when pulse wait > 2.0m
    day_boundary_hour_local: float = 4.0        # §4.6: 04:00 AM local day boundary
    student_enable_tap: bool = True             # §5.2 S1/S2/S3 switch: transfer-appropriate processing penalty at test
    student_enable_gen_bonus: bool = True       # §5.2 S2/S3 switch: generation engram quality bonus


# ==============================================================================
# 2B. Official 21-Parameter FSRS-6 Reference Engine (v1.13, C-84 / v1.15 Fit)
# ==============================================================================

@dataclass
class FSRS6OfficialParams:
    """
    Official 21-Parameter FSRS-6 Reference Parameterization (open-spaced-repetition / Jarrett Ye, C-84):
      w[0..3]:   Initial stability S_0(G) for grades G in {1: Again, 2: Hard, 3: Good, 4: Easy}
      w[4..5]:   Initial difficulty D_0(G) = clip(w4 - exp(w5 * (G - 1)) + 1, 1, 10)
      w[6..7]:   Difficulty linear damping & mean reversion to D_0(4)
      w[8..10]:  Long-term recall stability growth (exp(w8), S^(-w9), exp(w10*(1-R))-1)
      w[11..14]: Post-lapse stability (w11 * D^(-w12) * ((S+1)^w13 - 1) * exp(w14*(1-R)))
      w[15..16]: Hard penalty (w15) and Easy bonus (w16)
      w[17..19]: Same-day short-term stability S' = S * exp(w17 * (G - 3 + w18)) * S^(-w19)
      w[20]:     Trainable power-law forgetting curve decay exponent (R(t,S) = (1 + factor * t/S)^(-w20))
    """
    w: Tuple[float, ...] = (
        0.212, 1.2931, 2.3065, 8.2956,
        6.4133, 0.8334, 3.0194, 0.001,
        1.8722, 0.1666, 0.796, 1.4835,
        0.0614, 0.2629, 1.6483, 0.6014,
        1.8729, 0.5425, 0.0912, 0.0658,
        0.1542
    )
    request_retention: float = 0.90
    s_min: float = 0.10
    s_max: float = 36500.0


def fsrs6_forgetting_curve(
    t_days: float,
    S: float,
    params: Optional[FSRS6OfficialParams] = None
) -> float:
    """Official FSRS-6 trainable power-law forgetting curve: R(t, S) = (1 + factor * t / S)^(-w20)."""
    if S is None or S <= 0.0:
        return 0.0
    p = params or FSRS6OfficialParams()
    w20 = max(0.05, p.w[20])
    factor = (0.9 ** (-1.0 / w20)) - 1.0
    return (1.0 + factor * (max(0.0, t_days) / max(p.s_min, S))) ** (-w20)


def fsrs6_init_stability(grade: int, params: Optional[FSRS6OfficialParams] = None) -> float:
    """Official FSRS-6 initial stability S_0(G) = w[G - 1] for G in {1, 2, 3, 4}."""
    p = params or FSRS6OfficialParams()
    g_idx = max(0, min(3, int(grade) - 1))
    return max(p.s_min, float(p.w[g_idx]))


def fsrs6_init_difficulty(grade: int, params: Optional[FSRS6OfficialParams] = None) -> float:
    """Official FSRS-6 initial difficulty D_0(G) = clip(w4 - exp(w5 * (G - 1)) + 1, 1, 10)."""
    p = params or FSRS6OfficialParams()
    g = max(1, min(4, int(grade)))
    d0 = p.w[4] - math.exp(p.w[5] * (g - 1.0)) + 1.0
    return max(1.0, min(10.0, d0))


def fsrs6_next_difficulty(D: float, grade: int, params: Optional[FSRS6OfficialParams] = None) -> float:
    """Official FSRS-6 difficulty update with linear damping and mean reversion toward D_0(4)."""
    p = params or FSRS6OfficialParams()
    g = max(1, min(4, int(grade)))
    delta_d = -p.w[6] * (g - 3.0)
    d_prime = D + delta_d * ((10.0 - D) / 9.0)
    d0_easy = fsrs6_init_difficulty(4, p)
    d_next = p.w[7] * d0_easy + (1.0 - p.w[7]) * d_prime
    return max(1.0, min(10.0, d_next))


def fsrs6_short_term_stability(S: float, grade: int, params: Optional[FSRS6OfficialParams] = None) -> float:
    """Official FSRS-6 same-day short-term stability transition using w[17..19]."""
    p = params or FSRS6OfficialParams()
    g = max(1, min(4, int(grade)))
    s_safe = max(p.s_min, S)
    s_inc = math.exp(p.w[17] * (g - 3.0 + p.w[18])) * (s_safe ** (-p.w[19]))
    if g >= 3:
        s_inc = max(1.0, s_inc)
    return max(p.s_min, min(p.s_max, s_safe * s_inc))


def fsrs6_next_recall_stability(
    D: float,
    S: float,
    R: float,
    grade: int,
    params: Optional[FSRS6OfficialParams] = None
) -> float:
    """Official FSRS-6 long-term recall stability update S'_r(D, S, R, G) using w[8..10, 15..16]."""
    p = params or FSRS6OfficialParams()
    g = max(2, min(4, int(grade)))
    s_safe = max(p.s_min, S)
    d_safe = max(1.0, min(10.0, D))
    hard_penalty = p.w[15] if g == 2 else 1.0
    easy_bonus = p.w[16] if g == 4 else 1.0
    inc = (
        math.exp(p.w[8])
        * (11.0 - d_safe)
        * (s_safe ** (-p.w[9]))
        * (math.exp(p.w[10] * (1.0 - R)) - 1.0)
        * hard_penalty
        * easy_bonus
    )
    return max(p.s_min, min(p.s_max, s_safe * (1.0 + inc)))


def fsrs6_next_forget_stability(
    D: float,
    S: float,
    R: float,
    params: Optional[FSRS6OfficialParams] = None
) -> float:
    """Official FSRS-6 post-lapse stability S'_f(D, S, R) using w[11..14]."""
    p = params or FSRS6OfficialParams()
    s_safe = max(p.s_min, S)
    d_safe = max(1.0, min(10.0, D))
    s_f = (
        p.w[11]
        * (d_safe ** (-p.w[12]))
        * (((s_safe + 1.0) ** p.w[13]) - 1.0)
        * math.exp(p.w[14] * (1.0 - R))
    )
    return max(p.s_min, min(s_safe, s_f))


def fsrs6_next_interval(S: float, params: Optional[FSRS6OfficialParams] = None) -> int:
    """Official FSRS-6 next optimal interval in days for target retention R_req."""
    p = params or FSRS6OfficialParams()
    w20 = max(0.05, p.w[20])
    factor = (0.9 ** (-1.0 / w20)) - 1.0
    r_req = max(0.70, min(0.99, p.request_retention))
    ivl = (max(p.s_min, S) / factor) * ((r_req ** (-1.0 / w20)) - 1.0)
    return max(1, int(round(ivl)))


def fit_fsrs6_params_from_logs(
    logs: List[ReviewEvent],
    request_retention: float = 0.90,
    cfg: Optional[EngineConfig] = None
) -> FSRS6OfficialParams:
    """
    v1.15 (§5.1.2 `A-FSRS6-fit` / `A++-FSRS6-fit`):
    Fits key FSRS-6 parameters (initial stability scale alpha_s0 on w[0..3], recall growth w[8],
    and power-law exponent w[20]) by minimizing prequential binary cross-entropy (LogLoss) on
    cross-day review transitions from longitudinal review logs.
    Eliminates the baseline handicap caused by mismatched default FSRS-6 parameters (e.g., w[3]=8.296d).
    """
    c_cfg = cfg or EngineConfig()
    base_w = list(FSRS6OfficialParams().w)
    if not logs:
        return FSRS6OfficialParams(w=tuple(base_w), request_retention=request_retention)

    # Group events by card_id in chronological order
    by_card: Dict[str, List[ReviewEvent]] = {}
    for ev in logs:
        by_card.setdefault(ev.card_id, []).append(ev)

    def _eval_loss(scale_s0: float, easy_cap_mult: float, w8_val: float, w20_val: float) -> float:
        w_cand = list(base_w)
        w_cand[0] = max(0.15, base_w[0] * scale_s0)
        w_cand[1] = max(0.30, base_w[1] * scale_s0)
        w_cand[2] = max(0.50, base_w[2] * scale_s0)
        # Cap w[3] (Easy initial stability) relative to w[2] to avoid the 8.3d default overshoot
        w_cand[3] = max(w_cand[2] * 1.15, w_cand[2] * easy_cap_mult)
        w_cand[8] = w8_val
        w_cand[20] = w20_val
        p_cand = FSRS6OfficialParams(w=tuple(w_cand), request_retention=request_retention)

        ll_sum = 0.0
        n_obs = 0
        # Weak L2 regularization toward sensible domain priors
        reg = 0.02 * ((math.log(max(0.1, scale_s0))) ** 2 + (w8_val - base_w[8]) ** 2)
        for cid, evs in by_card.items():
            s_cur: Optional[float] = None
            d_cur: float = evs[0].D_before if evs else 5.0
            last_t: Optional[float] = None
            last_day: Optional[int] = None
            for ev in evs:
                g_val = int(ev.grade)
                if s_cur is None:
                    d0_g = fsrs6_init_difficulty(g_val, p_cand)
                    d_cur = max(1.0, min(10.0, 0.5 * d_cur + 0.5 * d0_g))
                    s0_g = fsrs6_init_stability(g_val, p_cand)
                    s_cur = max(p_cand.s_min, s0_g * math.exp(-0.12 * (d_cur - 5.0)))
                elif last_day == ev.day and last_t is not None and (ev.timestamp - last_t) < 360.0:
                    d_cur = fsrs6_next_difficulty(d_cur, g_val, p_cand)
                    s_cur = fsrs6_short_term_stability(s_cur, g_val, p_cand)
                else:
                    dt_days = max(1e-3, (ev.timestamp - (last_t or ev.timestamp)) / 1440.0)
                    r_pred = fsrs6_forgetting_curve(dt_days, s_cur, p_cand)
                    p_obs = max(1e-5, min(1.0 - 1e-5, predict_p_correct(r_pred, int(ev.modality), c_cfg)))
                    y_obs = 1.0 if g_val >= 2 else 0.0
                    ll_sum += -(y_obs * math.log(p_obs) + (1.0 - y_obs) * math.log(1.0 - p_obs))
                    n_obs += 1
                    d_cur = fsrs6_next_difficulty(d_cur, g_val, p_cand)
                    if g_val >= 2:
                        s_cur = fsrs6_next_recall_stability(d_cur, s_cur, r_pred, g_val, p_cand)
                    else:
                        s_cur = fsrs6_next_forget_stability(d_cur, s_cur, r_pred, p_cand)
                last_t = ev.timestamp
                last_day = ev.day
        if n_obs == 0:
            return 1e9
        return (ll_sum / n_obs) + reg

    # Coordinate grid + local refinement over (scale_s0, easy_cap_mult, w8, w20)
    best_params = (0.65, 1.35, base_w[8], base_w[20])
    best_loss = _eval_loss(*best_params)

    for s_scale in (0.35, 0.50, 0.65, 0.80, 1.00, 1.25):
        for e_mult in (1.20, 1.45, 1.80, 2.50):
            loss = _eval_loss(s_scale, e_mult, best_params[2], best_params[3])
            if loss < best_loss:
                best_loss = loss
                best_params = (s_scale, e_mult, best_params[2], best_params[3])

    for w8_cand in (1.20, 1.55, 1.8722, 2.15, 2.45):
        for w20_cand in (0.1542, 0.25, 0.40, 0.50):
            loss = _eval_loss(best_params[0], best_params[1], w8_cand, w20_cand)
            if loss < best_loss:
                best_loss = loss
                best_params = (best_params[0], best_params[1], w8_cand, w20_cand)

    # Fine local refinement around best_params
    for ds in (-0.08, 0.0, 0.08):
        for dw8 in (-0.15, 0.0, 0.15):
            s_try = max(0.20, best_params[0] + ds)
            w8_try = max(0.50, best_params[2] + dw8)
            loss = _eval_loss(s_try, best_params[1], w8_try, best_params[3])
            if loss < best_loss:
                best_loss = loss
                best_params = (s_try, best_params[1], w8_try, best_params[3])

    s_opt, e_opt, w8_opt, w20_opt = best_params
    w_fit = list(base_w)
    w_fit[0] = round(max(0.15, base_w[0] * s_opt), 4)
    w_fit[1] = round(max(0.30, base_w[1] * s_opt), 4)
    w_fit[2] = round(max(0.50, base_w[2] * s_opt), 4)
    w_fit[3] = round(max(w_fit[2] * 1.15, w_fit[2] * e_opt), 4)
    w_fit[8] = round(w8_opt, 4)
    w_fit[20] = round(w20_opt, 4)
    return FSRS6OfficialParams(w=tuple(w_fit), request_retention=request_retention)



# ==============================================================================
# 3. Core Engine Mathematical Operators (v1.13)
# ==============================================================================

def compute_normalized_pe_gate(R_mem: float, cfg: EngineConfig) -> float:
    """Formula 10A (v1.10, C-66): Asymptotically normalized PE gate with G(1) = 1.000."""
    num = 1.0 + math.exp(-cfg.w4b * (1.0 - cfg.R_gate))
    den = 1.0 + math.exp(-cfg.w4b * (R_mem - cfg.R_gate))
    return num / den

def compute_session_concurrency_cap(
    remaining_time_min: float = 0.0, 
    optimal_interval: float = 10.0, 
    rt_m3_sec: float = 35.0,
    session_budget_min: Optional[float] = None
) -> int:
    """Formula C-51, C-60, C-69: Dynamic remaining-budget concurrency capacity."""
    t_budget = session_budget_min if session_budget_min is not None else remaining_time_min
    remaining_work_window_min = max(0.0, t_budget - 2.0 * optimal_interval)
    cap = math.floor(remaining_work_window_min / (rt_m3_sec / 60.0))
    return max(0, min(20, cap))

def calculate_initial_stability(D: float, E_final: float = 1.0, cfg: Optional[EngineConfig] = None) -> float:
    """
    Formula 6 & Scheme A (v1.9, C-58, C-62):
    Uncalibrated objective initial stability. Strictly does NOT multiply Q_calib.
    Benchmark points (E_final = Theta = 1.0):
      D=1  -> 3.200d
      D=5  -> 1.558d
      D=10 -> 0.633d
    """
    c = cfg or EngineConfig()
    return c.w0 * math.exp(-c.w1 * (D - 1.0)) * ((E_final / c.Theta_CREB) ** c.beta_E)

def memory_retrievability(
    dt_days: float = 0.0,
    stability: Optional[float] = None,
    q_calib: Optional[float] = None,
    delta_days: Optional[float] = None,
    S: Optional[float] = None,
    Q_calib: Optional[float] = None
) -> float:
    """
    Formula 7a (v1.9/v1.10, Scheme A, C-58):
    Latent retrievability with unified S_eff = S * Q_calib.
    R = (1 + (19/81) * dt_days / (S * Q_calib))^(-0.5)
    Supports both named and positional forms for backward compatibility.
    """
    t = dt_days if delta_days is None else delta_days
    s_val = stability if stability is not None else S
    q_val = q_calib if q_calib is not None else (Q_calib if Q_calib is not None else 1.0)
    if s_val is None or s_val <= 0.0:
        return 0.0
    s_eff = s_val * max(0.1, q_val)
    factor = 19.0 / 81.0
    return (1.0 + factor * (t / s_eff)) ** (-0.5)

def modality_retrievability(R_mem: float, modality: int, cfg: EngineConfig, user: Optional[UserState] = None) -> float:
    """
    Formula 7c (v1.10, C-63, C-67):
    Modality-disjoint retrievability with logit boundary protection [eps, 1-eps]
    and logit calibration bias theta_calib.
    """
    R_clamped = max(1e-5, min(1.0 - 1e-5, R_mem))
    logit = math.log(R_clamped / (1.0 - R_clamped))
    theta = 0.0
    if user and getattr(cfg, 'enable_logit_calib', True):
        theta = user.theta_calib if user.theta_calib != 0.0 else user.theta_calib_prev
    logit_m = logit + theta + cfg.modality_offsets.get(modality, 0.0)
    return 1.0 / (1.0 + math.exp(-logit_m))

def predict_p_correct(R_mem: float, modality: int, cfg: EngineConfig, user: Optional[UserState] = None) -> float:
    """Formula 7d: Observation model with guessing and slipping probabilities."""
    Rm = modality_retrievability(R_mem, modality, cfg, user)
    g = cfg.guess_rates.get(modality, 0.0)
    s = cfg.slip_rates.get(modality, 0.02)
    return g + (1.0 - g - s) * Rm

def check_optimal_stop_rel(card: Card, user: UserState, cfg: EngineConfig) -> bool:
    """
    Formula 6 (v1.9/v1.11, C-54, C-59, C-75):
    Scale-free relative MER for session deferral only.
    Guarded against zero division when E <= 0 and against premature deferral
    when user fatigue is below the moderate fatigue threshold (F < 0.55).
    """
    if card.E < 0.70 or card.E <= 0.0:
        return False
    if user.F < 0.55:
        return False
    A_pre = card.A * math.exp(-10.0 / cfg.tau_A)
    Phi = 1.0 - card.R_des * math.exp(-10.0 / cfg.tau_des)
    next_lambda = 1.00 if card.m_last == Modality.CONTEXT_GEN.value else cfg.lambda_gen
    dE_exp = (1.0 - 0.5 * user.F) * next_lambda * Phi * (cfg.alpha_base + cfg.alpha_res * A_pre)
    
    # Scale-free relative stability gain (independent of item difficulty D)
    rel_gain = ((card.E + dE_exp) / card.E) ** cfg.beta_E - 1.0
    cost = max(1e-5, cfg.dt_cost * (1.0 + cfg.omega_F * user.F))
    mer_rel = rel_gain / cost
    return mer_rel < cfg.lambda_stop_rel

# Backward-compatible alias
check_optimal_stop = check_optimal_stop_rel

def get_effective_theta_calib(user: UserState, cfg: EngineConfig, group: str = "D") -> float:
    """
    v1.15 (§11.1 & G6): CRLB Posterior Uncertainty Gate for `theta_calib`.
    For v1.15 calibrated arms (`OUME-Pragmatic`, `A++-cal`, `D-v115`), shrinks `theta_calib`
    proportionally when posterior precision `I_total + lambda_0 < (1 / theta_crlb_sd_gate)^2`
    (i.e. `theta_posterior_sd > 0.20`), and releases 100% of `theta_calib` once
    `theta_posterior_sd <= 0.20`. Preserves legacy `D` behavior bit-for-bit.
    """
    raw_theta = user.theta_calib_prev if user.theta_calib_prev != 0.0 else user.theta_calib
    if group in ("OUME-Pragmatic", "A++-cal", "D-v115"):
        if user.cum_fisher_info <= 0.0:
            return 0.0
        prior_prec = cfg.probe_shrinkage_n0 * 0.35
        target_prec = 1.0 / max(1e-4, cfg.theta_crlb_sd_gate ** 2)  # 25.0 when sd_gate = 0.20
        w_crlb = min(1.0, (user.cum_fisher_info + prior_prec) / target_prec)
        return raw_theta * w_crlb
    return raw_theta


def evaluate_response_grade(
    is_correct: bool,
    used_hint: bool = False,
    rt_ms: int = 4000,
    modality: Union[Modality, int] = Modality.AUDIO_SPELL,
    confidence: Optional[int] = None,
    is_guided_primer: bool = False,
    user: Optional[UserState] = None,
    llm_kappa: float = 0.85,
    rubric_kappa: Optional[float] = None,
    cfg: Optional[EngineConfig] = None,
    record_rt: bool = True
) -> Grade:
    """
    Section 4.5 (v1.15): Production Grade Mapping Operator g = f(accuracy, rt, confidence).
    Enforces:
      1. Hint / wrong penalty -> Grade.AGAIN;
      2. Rolling 50-review per-modality log-RT z-score standardization (with cold-start priors);
      3. Modality & Guided Primer ceiling (m=1 or is_guided_primer capped at Grade.GOOD);
      4. Cold-start protection (first 20 answers of a new user capped at Grade.GOOD);
      5. m=3 LLM rubric reliability fallback (degrades to binary {AGAIN, GOOD} when llm_kappa < 0.70).
    """
    mod_val = int(modality.value) if isinstance(modality, Modality) else int(modality)
    default_mu = {1: math.log(8.0), 2: math.log(14.0), 3: math.log(32.0)}.get(mod_val, math.log(14.0))
    default_sigma = 0.35

    rt_sec = max(0.5, float(rt_ms) / 1000.0)
    ln_rt = math.log(rt_sec)

    mu_mod = default_mu
    sigma_mod = default_sigma
    prev_ans_count = 0

    if user is not None:
        prev_ans_count = user.total_ans_count
        hist = user.rt_stats_by_mod.setdefault(mod_val, [])
        if len(hist) >= 5:
            m_emp = sum(hist) / len(hist)
            var_emp = sum((x - m_emp) ** 2 for x in hist) / max(1, len(hist) - 1)
            sd_emp = max(0.15, math.sqrt(var_emp))
            # Smooth Bayesian blend between cold-start prior and rolling window
            w_emp = min(1.0, len(hist) / 15.0)
            mu_mod = (1.0 - w_emp) * default_mu + w_emp * m_emp
            sigma_mod = (1.0 - w_emp) * default_sigma + w_emp * sd_emp
        if record_rt:
            hist.append(ln_rt)
            if len(hist) > 50:
                del hist[0]
            user.total_ans_count += 1

    z_k = (ln_rt - mu_mod) / max(0.15, sigma_mod)

    # Rule 1: Any incorrect answer or hint usage maps strictly to AGAIN
    if (not is_correct) or used_hint:
        return Grade.AGAIN

    # Rule 5: When m=3 rubric agreement kappa < 0.70, degrade to binary {AGAIN, GOOD}
    eff_kappa = rubric_kappa if rubric_kappa is not None else llm_kappa
    if mod_val == Modality.CONTEXT_GEN.value and eff_kappa < 0.70:
        return Grade.GOOD

    conf = 2 if confidence is None else int(confidence)

    # Rule 2: Candidate grade from (z_k, conf)
    if z_k > 0.85 or conf <= 1:
        cand = Grade.HARD
    elif z_k <= -0.5 and conf >= 3 and mod_val in (Modality.AUDIO_SPELL.value, Modality.CONTEXT_GEN.value):
        cand = Grade.EASY
    else:
        cand = Grade.GOOD

    # Rule 3: Modality & Guided Primer ceiling (m=1 or Guided Primer never exceeds GOOD)
    if mod_val == Modality.VISUAL_CHOICE.value or is_guided_primer:
        if cand == Grade.EASY:
            cand = Grade.GOOD

    # Rule 4: Cold-start protection (first 20 answers cannot trigger EASY)
    if user is not None and prev_ans_count < 20 and cand == Grade.EASY:
        cand = Grade.GOOD

    return cand


def compute_learning_day(
    unix_timestamp_sec: float,
    tz_offset_hours: float = 0.0,
    prev_learning_day: Optional[int] = None,
    epoch_day_zero: int = 0,
    day_boundary_hour_local: float = 4.0
) -> int:
    """
    Section 4.6 (v1.15): Local 04:00 AM day boundary index with monotonic non-decreasing invariant
    across timezone travel:
      d = max(d_prev, floor((t_unix + Delta_tz - 4h) / 86400) - d_epoch0)
    """
    shifted_sec = unix_timestamp_sec + (tz_offset_hours - day_boundary_hour_local) * 3600.0
    d_raw = math.floor(shifted_sec / 86400.0) - epoch_day_zero
    if prev_learning_day is not None:
        return max(prev_learning_day, d_raw)
    return d_raw


def handle_session_transition(
    cards: List[Card],
    user: UserState,
    now_t: float,
    current_day: int,
    cfg: Optional[EngineConfig] = None
) -> Dict[str, int]:
    """
    Section 4.6 (v1.15): Production Session Boundary, Multi-Session, Day-Skip & Interruption Handler.
    Handles:
      1. Same-day multi-session (< 360 min gap on same learning day `d`):
         - Does NOT trigger SLEEP_PENDING overnight settlement;
         - Decays continuous fatigue `user.F` by elapsed minutes;
         - If session interruption gap > (optimal_interval_min + 15.0) min, resets fast variables
           `A <- 0, R_des <- 0` while decaying slow `E` by `exp(-elapsed_min / tau_E)`.
      2. Overnight / >= 360 min consolidation or multi-day skip (`delta_days >= 1`):
         - Resets `user.F <- 0` and freezes `user.theta_calib_prev <- user.theta_calib`;
         - Settles `SLEEP_PENDING` cards into `REVIEW` with `S_0(D, E)`;
         - Ages unfinished `MICRO_PULSING` / `RELEARNING` cards by `delta_days` (`days_in_flight += delta_days`),
           applying `stale_timeout_new` (3d -> NEW) and `stale_timeout_relearn` (7d -> REVIEW).
    """
    c_cfg = cfg or EngineConfig()
    elapsed_min = max(0.0, now_t - user.last_action_t) if user.last_action_t > 0.0 else 1440.0
    is_same_day_multisession = (
        user.last_session_day is not None
        and current_day == user.last_session_day
        and elapsed_min < 360.0
    )

    stats = {
        "settled_sleep_pending": 0,
        "interrupted_fast_reset": 0,
        "stale_reset_to_new": 0,
        "stale_returned_to_review": 0,
        "is_same_day_multisession": 1 if is_same_day_multisession else 0
    }

    if is_same_day_multisession:
        user.F *= math.exp(-elapsed_min / c_cfg.tau_rest)
        interruption_cutoff = c_cfg.optimal_interval_min + 15.0
        for c in cards:
            if c.state in (CardState.MICRO_PULSING, CardState.RELEARNING):
                card_gap = max(0.0, now_t - (c.last_review_t or now_t))
                if card_gap > interruption_cutoff:
                    # §4.6 Rule 5: Mid-session interruption recovery (> 25m)
                    c.A = 0.0
                    c.R_des = 0.0
                    c.E *= math.exp(-card_gap / c_cfg.tau_E)
                    c.refractory_until = min(c.refractory_until, now_t)
                    c.optimal_due_at = min(c.optimal_due_at, now_t)
                    c.last_review_t = now_t - c_cfg.optimal_interval_min
                    stats["interrupted_fast_reset"] += 1
        user.last_action_t = now_t
        return stats

    # Overnight or >= 360m consolidation boundary (including multi-day skips)
    delta_days = max(1, current_day - user.last_session_day) if user.last_session_day is not None else 1
    user.F = 0.0
    user.theta_calib_prev = user.theta_calib
    q_cal = math.exp(get_effective_theta_calib(user, c_cfg, "OUME-Pragmatic"))

    for c in cards:
        c.is_session_deferred = False
        if c.state == CardState.SLEEP_PENDING:
            c.S = calculate_initial_stability(c.D, c.E, c_cfg)
            base_grad_day = c.graduated_day if c.graduated_day is not None else max(1, current_day - delta_days)
            c.scheduled_due_day = max(current_day, base_grad_day + max(1, round(c.S * q_cal)))
            c.state = CardState.REVIEW
            stats["settled_sleep_pending"] += 1
        elif c.state in (CardState.MICRO_PULSING, CardState.RELEARNING):
            c.A = 0.0
            c.R_des = 0.0
            c.E *= (c_cfg.rho_overnight ** delta_days)
            c.days_in_flight += delta_days
            if c.state == CardState.MICRO_PULSING and c.days_in_flight > c_cfg.stale_timeout_new:
                c.state = CardState.NEW
                c.E = 0.0
                c.pulse_count = 0
                c.micro_streak = 0
                c.days_in_flight = 0
                stats["stale_reset_to_new"] += 1
            elif c.state == CardState.RELEARNING and c.days_in_flight > c_cfg.stale_timeout_relearn:
                c.state = CardState.REVIEW
                c.E = 0.0
                c.pulse_count = 0
                c.micro_streak = 0
                c.days_in_flight = 0
                stats["stale_returned_to_review"] += 1
            else:
                c.refractory_until = now_t
                c.optimal_due_at = now_t
                c.last_review_t = now_t - c_cfg.optimal_interval_min

    user.last_session_day = current_day
    user.last_action_t = now_t
    return stats


def get_scaffolding_modality(card: Card, group: str) -> Modality:
    # Pure generation control arm (§5.1.2 G-fixed: m=3 on all intra-day and cross-day exposures)
    if group == "G-fixed":
        return Modality.CONTEXT_GEN

    if group in (
        "A", "A+", "A++", "A++-cal", "A-FSRS6", "A++-FSRS6", "A-FSRS6-fit", "A++-FSRS6-fit",
        "B", "C-fixed", "C-E-fixed", "C-ISI-3m", "C-ISI-10m", "C-ISI-20m"
    ):
        return Modality.AUDIO_SPELL

    if card.state in (CardState.NEW, CardState.MICRO_PULSING, CardState.RELEARNING):
        if card.micro_streak == 0:
            return Modality.VISUAL_CHOICE if card.m_last != Modality.VISUAL_CHOICE.value else Modality.AUDIO_SPELL
        elif card.micro_streak == 1:
            return Modality.AUDIO_SPELL if card.m_last != Modality.AUDIO_SPELL.value else Modality.CONTEXT_GEN
        else:
            return Modality.CONTEXT_GEN
    else:
        return Modality.CONTEXT_GEN if card.m_last == Modality.AUDIO_SPELL.value else Modality.AUDIO_SPELL

def pick_next_card(
    session_cards: List[Card], 
    user: UserState, 
    now_t: float, 
    current_day: int,
    session_budget_end: float,
    session_budget_min: float,
    cfg: Optional[EngineConfig] = None,
    session_start_t: Optional[float] = None,
    group: str = "D",
    new_introduced_today: int = 0,
    fsrs6_params: Optional[FSRS6OfficialParams] = None,
    return_rescue_flag: bool = False
) -> Union[Tuple[Optional[Card], Optional[float]], Tuple[Optional[Card], Optional[float], bool]]:
    """
    Section 4.4 (v1.14/v1.15, Scheme A, C-69, C-75, C-91): Unified reference production scheduler algorithm.
    Features:
    1. Dynamic remaining-time concurrency capacity N_cap(t) bounded by wave cap W_wave = 8;
    2. Early-session new card incubation injection (warm-up first 5 min);
    3. Reviews naturally fill 10m incubation gaps;
    4. End-of-session soft-refractory rescue when any in-flight card would miss session_budget_end.
    Returns: (selected_card, optional_fast_forward_target_t) [or 3-tuple with is_soft_rescue if return_rescue_flag=True]
    """
    def _ret(card_res: Optional[Card], ff_res: Optional[float], rescued: bool = False):
        if return_rescue_flag:
            return card_res, ff_res, rescued
        return card_res, ff_res

    c_cfg = cfg or EngineConfig()
    fsrs_p = fsrs6_params or FSRS6OfficialParams(request_retention=c_cfg.R_target)
    start_t = session_start_t if session_start_t is not None else (session_budget_end - session_budget_min)
    rem_t = max(0.0, session_budget_end - now_t)
    dyn_cap = compute_session_concurrency_cap(rem_t, c_cfg.optimal_interval_min)
    wave_cap = min(dyn_cap, c_cfg.max_wave_concurrency)
    initial_wave_cap = min(
        compute_session_concurrency_cap(session_budget_min, c_cfg.optimal_interval_min),
        c_cfg.max_wave_concurrency
    )
    active_in_flight = [c for c in session_cards if c.state in (CardState.MICRO_PULSING, CardState.RELEARNING)]
    fatigue_ok = (user.F < 0.65) if ("D" in group) else True
    can_inject_new = (new_introduced_today < c_cfg.max_daily_new_cards) and fatigue_ok

    intraday_pulse_groups = (
        "A+", "A++", "A++-scaff", "A++-cal", "A++-FSRS6", "A++-FSRS6-fit",
        "C-fixed", "C-scaff-fixed", "G-fixed", "C-E-fixed",
        "C", "D", "D-v115", "OUME-Pragmatic", "C-ISI-3m", "C-ISI-10m", "C-ISI-20m"
    )
    three_pulse_window_groups = (
        "A++", "A++-scaff", "A++-cal", "A++-FSRS6", "A++-FSRS6-fit",
        "C-fixed", "C-scaff-fixed", "G-fixed", "C-E-fixed",
        "C", "D", "D-v115", "OUME-Pragmatic"
    )
    soft_fallback_groups = (
        "C-fixed", "C-scaff-fixed", "G-fixed", "C-E-fixed",
        "C", "D", "D-v115", "OUME-Pragmatic",
        "C-ISI-3m", "C-ISI-10m", "C-ISI-20m"
    )

    # Priority 1: Ready micro-pulses
    if group in intraday_pulse_groups:
        ready_pulses = [
            c for c in active_in_flight 
            if now_t >= c.optimal_due_at and not getattr(c, "is_session_deferred", False)
        ]
        if ready_pulses:
            ready_pulses.sort(key=lambda c: c.optimal_due_at)
            return _ret(ready_pulses[0], None, False)

    # Priority 2: Early-session new card incubation injection (first 5 min)
    is_warmup = (now_t - start_t) < 5.0
    if is_warmup and len(active_in_flight) < initial_wave_cap and can_inject_new:
        new_cards = [c for c in session_cards if c.state == CardState.NEW]
        if new_cards:
            return _ret(new_cards[0], None, False)

    # Priority 3: Cross-day due reviews (fill 10m incubation gap)
    due_reviews = [
        c for c in session_cards 
        if c.state == CardState.REVIEW and c.scheduled_due_day <= current_day
    ]
    if due_reviews:
        if group in ("A-FSRS6", "A++-FSRS6", "A-FSRS6-fit", "A++-FSRS6-fit"):
            due_reviews.sort(key=lambda c: fsrs6_forgetting_curve(
                max(0.0, (now_t - (c.last_review_t or 0.0)) / 1440.0),
                max(0.1, c.S or 0.5),
                fsrs_p
            ))
        else:
            q_cal_eff = math.exp(get_effective_theta_calib(user, c_cfg, group))
            due_reviews.sort(key=lambda c: memory_retrievability(
                dt_days=(now_t - (c.last_review_t or 0)) / 1440.0, 
                stability=c.S, 
                q_calib=q_cal_eff
            ))
        return _ret(due_reviews[0], None, False)

    # Priority 4: Regular mid-session new card introduction (if remaining time allows)
    req_window = (
        (2.0 * c_cfg.optimal_interval_min + 0.75 * len(active_in_flight) + 1.0)
        if group in three_pulse_window_groups else 0.0
    )
    if len(active_in_flight) < wave_cap and can_inject_new and (rem_t >= req_window):
        new_cards = [c for c in session_cards if c.state == CardState.NEW]
        if new_cards:
            return _ret(new_cards[0], None, False)

    # Priority 5: Soft-refractory fallback (when any non-deferred in-flight card would exceed budget_end)
    active_non_deferred = [
        c for c in active_in_flight 
        if not getattr(c, "is_session_deferred", False)
    ]
    has_remaining_new = (new_introduced_today < c_cfg.max_daily_new_cards) and any(
        c.state == CardState.NEW for c in session_cards
    )
    if group in soft_fallback_groups and (not has_remaining_new or len(active_in_flight) >= wave_cap) and active_non_deferred:
        max_optimal = max(c.optimal_due_at for c in active_non_deferred)
        if max_optimal > session_budget_end:
            eligible_fallback = [
                c for c in active_non_deferred 
                if (now_t - (c.last_review_t or now_t)) >= c_cfg.delta_t_soft
            ]
            if eligible_fallback:
                eligible_fallback.sort(key=lambda c: (now_t - (c.last_review_t or now_t)), reverse=True)
                return _ret(eligible_fallback[0], None, True)

    # Priority 6: Idle fast-forward (waking up at soft-refractory window if optimal_due_at > budget_end)
    if active_non_deferred:
        wake_times = []
        for c in active_non_deferred:
            if c.optimal_due_at > session_budget_end and (not has_remaining_new or len(active_in_flight) >= wave_cap):
                soft_t = (c.last_review_t or now_t) + c_cfg.delta_t_soft
                if soft_t > now_t and soft_t < session_budget_end:
                    wake_times.append(soft_t)
                else:
                    wake_times.append(c.optimal_due_at)
            else:
                wake_times.append(c.optimal_due_at)
        target_t = min(wake_times)
        fast_forward_t = min(target_t, session_budget_end)
        return _ret(None, fast_forward_t, False)
    else:
        return _ret(None, session_budget_end, False)

# ==============================================================================
# 4. State Transition & Review Engine (v1.13/v1.15, C-81, C-84)
# ==============================================================================

def execute_review_transition(
    card: Card,
    now_t: float,
    day: int,
    grade: Grade,
    modality: Modality,
    user: UserState,
    cfg: EngineConfig,
    group: str,
    fsrs6_params: Optional[FSRS6OfficialParams] = None
) -> float:
    # 0A. Official 21-Parameter FSRS-6 1-Exposure Baseline (Groups A-FSRS6, A-FSRS6-fit: strictly 1 exposure, C-81, C-84)
    if group in ("A-FSRS6", "A-FSRS6-fit"):
        if card.state == CardState.NEW:
            fsrs_p = fsrs6_params or FSRS6OfficialParams(request_retention=cfg.R_target)
            card.pulse_count += 1
            card.micro_streak = 0 if grade == Grade.AGAIN else 1
            d0_g = fsrs6_init_difficulty(grade.value, fsrs_p)
            card.D = max(1.0, min(10.0, 0.5 * card.D + 0.5 * d0_g))
            s0_g = fsrs6_init_stability(grade.value, fsrs_p)
            card.S = max(fsrs_p.s_min, s0_g * math.exp(-0.12 * (card.D - 5.0)))
            card.scheduled_due_day = (day + 1) if grade == Grade.AGAIN else (day + fsrs6_next_interval(card.S, fsrs_p))
            card.state = CardState.REVIEW
            if card.graduated_day is None:
                card.graduated_at_pulse = card.pulse_count
                card.graduated_day = day
            card.last_review_t = now_t
            return 0.0

    # 0B. Official 21-Parameter FSRS-6 3-Exposure Short-Term Step Baseline (Groups A++-FSRS6, A++-FSRS6-fit: strictly 3 exposures, C-81, C-84)
    if group in ("A++-FSRS6", "A++-FSRS6-fit"):
        fsrs_p = fsrs6_params or FSRS6OfficialParams(request_retention=cfg.R_target)
        if card.state == CardState.NEW:
            card.pulse_count += 1
            card.micro_streak = 0 if grade == Grade.AGAIN else 1
            d0_g = fsrs6_init_difficulty(grade.value, fsrs_p)
            card.D = max(1.0, min(10.0, 0.5 * card.D + 0.5 * d0_g))
            s0_g = fsrs6_init_stability(grade.value, fsrs_p)
            card.S = max(fsrs_p.s_min, s0_g * math.exp(-0.12 * (card.D - 5.0)))
            card.state = CardState.MICRO_PULSING
            card.optimal_due_at = now_t + 10.0
            card.last_review_t = now_t
            return 0.0
        elif card.state == CardState.MICRO_PULSING:
            card.pulse_count += 1
            card.micro_streak = 0 if grade == Grade.AGAIN else (card.micro_streak + 1)
            card.D = fsrs6_next_difficulty(card.D, grade.value, fsrs_p)
            card.S = fsrs6_short_term_stability(card.S or 0.5, grade.value, fsrs_p)
            if card.pulse_count >= 3:
                # Strict fixed-dose termination at 3 exposures regardless of Grade.AGAIN (C-81)
                card.scheduled_due_day = (day + 1) if grade == Grade.AGAIN else (day + fsrs6_next_interval(card.S, fsrs_p))
                card.state = CardState.REVIEW
                if card.graduated_day is None:
                    card.graduated_at_pulse = card.pulse_count
                    card.graduated_day = day
            else:
                card.optimal_due_at = now_t + 10.0
            card.last_review_t = now_t
            return 0.0

    # 1. Standard FSRS-6 Baseline (Group A: strictly 1 exposure graduation, C-81)
    if group == "A":
        if card.state == CardState.NEW:
            card.pulse_count += 1
            base_s0 = cfg.w0 * math.exp(-cfg.w1 * (card.D - 1.0))
            if grade == Grade.AGAIN:
                card.micro_streak = 0
                card.S = max(cfg.S_min, 0.50 * base_s0)
                card.scheduled_due_day = day + 1
            else:
                card.micro_streak = 1
                card.S = base_s0
                card.scheduled_due_day = day + max(1, round(card.S))
            card.state = CardState.REVIEW
            if card.graduated_day is None:
                card.graduated_at_pulse = card.pulse_count
                card.graduated_day = day
            card.last_review_t = now_t
            return 0.0

    # 1+. FSRS-6 + Static 10m Step (Group A+: strictly 2 exposures even on Grade.AGAIN, C-65, C-81)
    if group == "A+":
        if card.state == CardState.NEW:
            card.pulse_count += 1
            card.micro_streak = 0 if grade == Grade.AGAIN else 1
            card.state = CardState.MICRO_PULSING
            card.optimal_due_at = now_t + 10.0
            card.last_review_t = now_t
            return 0.0
        elif card.state == CardState.MICRO_PULSING:
            card.pulse_count += 1
            base_s0 = cfg.w0 * math.exp(-cfg.w1 * (card.D - 1.0))
            if grade == Grade.AGAIN:
                card.micro_streak = 0
                card.S = max(cfg.S_min, 0.55 * base_s0)
                card.scheduled_due_day = day + 1
            else:
                card.micro_streak += 1
                card.S = base_s0
                card.scheduled_due_day = day + max(1, round(card.S))
            card.state = CardState.REVIEW
            if card.graduated_day is None:
                card.graduated_at_pulse = card.pulse_count
                card.graduated_day = day
            card.last_review_t = now_t
            return 0.0

    # 1++. FSRS-6 + Static 10m/20m Steps (Groups A++, A++-scaff, A++-cal: strictly 3 exposures even on Grade.AGAIN, C-65, C-81)
    if group in ("A++", "A++-scaff", "A++-cal"):
        if card.state == CardState.NEW:
            card.pulse_count += 1
            card.micro_streak = 0 if grade == Grade.AGAIN else 1
            card.state = CardState.MICRO_PULSING
            card.optimal_due_at = now_t + 10.0
            card.m_last = modality.value
            card.last_review_t = now_t
            return 0.0
        elif card.state == CardState.MICRO_PULSING:
            card.pulse_count += 1
            if grade == Grade.AGAIN:
                card.micro_streak = 0
            else:
                card.micro_streak += 1
            if card.pulse_count >= 3:
                base_s0 = cfg.w0 * math.exp(-cfg.w1 * (card.D - 1.0))
                q_cal = math.exp(get_effective_theta_calib(user, cfg, group)) if group == "A++-cal" else 1.0
                if grade == Grade.AGAIN or card.micro_streak < 2:
                    card.S = max(cfg.S_min, 0.60 * base_s0)
                    card.scheduled_due_day = day + 1
                else:
                    card.S = base_s0
                    card.scheduled_due_day = day + max(1, round(card.S * q_cal))
                card.state = CardState.REVIEW
                if card.graduated_day is None:
                    card.graduated_at_pulse = card.pulse_count
                    card.graduated_day = day
            else:
                card.optimal_due_at = now_t + 10.0
            card.m_last = modality.value
            card.last_review_t = now_t
            return 0.0

    # 2. OUME-LongOnly (Group B: strictly 1 exposure graduation, normalized PE gate, C-81)
    if group == "B":
        if card.state == CardState.NEW:
            card.pulse_count += 1
            base_s0 = cfg.w0 * math.exp(-cfg.w1 * (card.D - 1.0))
            if grade == Grade.AGAIN:
                card.micro_streak = 0
                card.S = max(cfg.S_min, 0.50 * base_s0)
                card.scheduled_due_day = day + 1
            else:
                card.micro_streak = 1
                card.S = base_s0
                card.scheduled_due_day = day + max(1, round(card.S))
            card.state = CardState.REVIEW
            if card.graduated_day is None:
                card.graduated_at_pulse = card.pulse_count
                card.graduated_day = day
            card.last_review_t = now_t
            return 0.0

    # 3. Fixed-step OUME arms (C-fixed, C-scaff-fixed, G-fixed, OUME-Pragmatic, C-ISI-3m, C-ISI-10m, C-ISI-20m: strictly 3 exposures in NEW/MICRO_PULSING, strictly 2 in RELEARNING, C-65, C-80, C-81)
    if group in ("C-fixed", "C-scaff-fixed", "G-fixed", "OUME-Pragmatic", "C-ISI-3m", "C-ISI-10m", "C-ISI-20m") and card.state in (CardState.NEW, CardState.MICRO_PULSING, CardState.RELEARNING):
        card.pulse_count += 1
        interval = 3.0 if "3m" in group else (20.0 if "20m" in group else 10.0)
        
        # Telemetry updates
        delta_t = now_t - card.last_review_t if card.last_review_t is not None else 1000.0
        A_pre = card.A * math.exp(-delta_t / cfg.tau_A)
        R_des_pre = card.R_des * math.exp(-delta_t / cfg.tau_des)
        Phi_ref = 1.0 - R_des_pre
        if card.m_last is None or card.m_last == modality.value:
            Lambda_mod = 1.00
        elif modality == Modality.CONTEXT_GEN:
            Lambda_mod = cfg.lambda_gen
        else:
            Lambda_mod = cfg.lambda_shift
        Omega = user.psi_enc * user.V * (1.0 - 0.50 * user.F)
        q_g = cfg.q_weights.get(grade.value, 1.0) if grade != Grade.AGAIN else 0.0
        delta_E = Omega * q_g * Lambda_mod * Phi_ref * (cfg.alpha_base + cfg.alpha_res * A_pre)
        card.E = card.E * math.exp(-delta_t / cfg.tau_E) + delta_E
        card.A = min(1.0, A_pre + cfg.gamma_A * Phi_ref)
        card.R_des = min(1.0, R_des_pre + cfg.gamma_des)

        if grade == Grade.AGAIN:
            card.micro_streak = 0
        else:
            card.micro_streak += 1

        if card.state == CardState.RELEARNING:
            # Strict fixed-dose 2-exposure termination in RELEARNING regardless of Grade.AGAIN (C-81)
            if card.pulse_count >= 2:
                card.state = CardState.REVIEW
                base_s = max(card.S or 1.0, (card.S_pre_lapse or 1.0) * 0.78)
                card.S = min(cfg.S_max, base_s * ((max(0.35, card.E) / cfg.Theta_CREB) ** (0.5 * cfg.beta_E)))
                q_cal = math.exp(get_effective_theta_calib(user, cfg, group))
                card.scheduled_due_day = day + max(1, round((card.S or 1.0) * q_cal))
                card.is_session_deferred = False
            else:
                card.optimal_due_at = now_t + interval
        elif card.pulse_count >= 3:
            # Strict fixed-dose 3-exposure termination in NEW/MICRO_PULSING regardless of Grade.AGAIN (C-81)
            card.state = CardState.SLEEP_PENDING
            if card.graduated_day is None:
                card.graduated_at_pulse = card.pulse_count
                card.graduated_day = day
            card.is_session_deferred = False
        else:
            card.optimal_due_at = now_t + interval
            card.state = CardState.MICRO_PULSING

        card.m_last = modality.value
        card.last_review_t = now_t
        card.refractory_until = now_t + cfg.delta_t_soft
        return delta_E

    # 4. Cross-day REVIEW update (All groups, including Official FSRS-6 Reference arms)
    if card.state == CardState.REVIEW:
        delta_days = max(0.0, (now_t - (card.last_review_t or now_t)) / 1440.0)
        if group in ("A-FSRS6", "A++-FSRS6", "A-FSRS6-fit", "A++-FSRS6-fit"):
            fsrs_p = fsrs6_params or FSRS6OfficialParams(request_retention=cfg.R_target)
            R_fsrs = fsrs6_forgetting_curve(delta_days, card.S or 0.5, fsrs_p)
            card.D = fsrs6_next_difficulty(card.D, grade.value, fsrs_p)
            if grade != Grade.AGAIN:
                card.S = fsrs6_next_recall_stability(card.D, card.S or 0.5, R_fsrs, grade.value, fsrs_p)
                card.scheduled_due_day = day + fsrs6_next_interval(card.S, fsrs_p)
            else:
                card.lapse_count += 1
                card.S_pre_lapse = card.S
                card.S = fsrs6_next_forget_stability(card.D, card.S or 0.5, R_fsrs, fsrs_p)
                if card.lapse_count >= 8:
                    card.state = CardState.LEECH_SUSPENDED
                else:
                    card.scheduled_due_day = day + 1
                    card.state = CardState.REVIEW
            card.m_last = modality.value
            card.last_review_t = now_t
            return 0.0

        q_cal = math.exp(get_effective_theta_calib(user, cfg, group))
        R_mem = memory_retrievability(delta_days, card.S, q_cal)

        # Formula 9: Difficulty update
        card.D = max(1.0, min(10.0, cfg.w_d * cfg.D_init + (1.0 - cfg.w_d) * (card.D - cfg.delta_d * (grade.value - 3))))

        if grade != Grade.AGAIN:
            # Formula 10 Branch A
            mu_g = {Grade.HARD: 0.65, Grade.GOOD: 1.00, Grade.EASY: 1.35}[grade]
            if group in ("A", "A+", "A++", "A++-scaff", "A++-cal"):
                # Standard FSRS linear prediction error with un-gated overdue interval expansion
                overdue_ratio = max(1.0, delta_days / max(0.25, (card.S or 1.0) * q_cal))
                pe = (math.exp(cfg.w4 * (1.0 - R_mem)) - 1.0) * (overdue_ratio ** 0.55)
            else:
                # Formula 10A (v1.10, C-66): Asymptotically Normalized Sigmoid PE gate with G(1) = 1.000
                pe_base = math.exp(cfg.w4 * (1.0 - R_mem)) - 1.0
                sigmoid_gate = compute_normalized_pe_gate(R_mem, cfg)
                pe = pe_base * sigmoid_gate

            gain = cfg.theta_gain * mu_g * ((11.0 - card.D) ** cfg.w2) * ((card.S or 1.0) ** (-cfg.w3)) * pe
            card.S = min(cfg.S_max, (card.S or 1.0) * (1.0 + gain))
            card.scheduled_due_day = day + max(1, round(card.S * q_cal))
        else:
            # Formula 10 Branch B: Lapse (with calibrated effective stability S * exp(theta_calib_prev))
            card.lapse_count += 1
            card.S_pre_lapse = card.S
            s_eff_lapse = max(0.1, (card.S or 1.0) * q_cal)
            lapse_decay = 1.0 - math.exp(-delta_days / s_eff_lapse)
            card.S = max(cfg.S_min, (card.S or 1.0) * cfg.w_lapse * (card.D ** (-cfg.w5)) * lapse_decay)

            if card.lapse_count >= 8:
                card.state = CardState.LEECH_SUSPENDED
            else:
                if group in ("A", "A+", "A++", "A++-scaff", "A++-cal", "B"):
                    card.scheduled_due_day = day + 1
                    card.state = CardState.REVIEW
                else:
                    card.state = CardState.RELEARNING
                    card.pulse_count = 0
                    card.E = cfg.eta_lapse * cfg.Theta_CREB
                    card.micro_streak = 0
                    card.refractory_until = now_t + cfg.delta_t_soft
                    interval = 3.0 if "3m" in group else (20.0 if "20m" in group else cfg.optimal_interval_min)
                    card.optimal_due_at = now_t + interval

        card.m_last = modality.value
        card.last_review_t = now_t
        return 0.0

    # 5. Adaptive Intra-day Micro-pulsing logic (Groups C-E-fixed, C, D, D-v115: dynamic biochemical gate, C-59, C-80)
    if card.state in (CardState.NEW, CardState.MICRO_PULSING, CardState.RELEARNING):
        card.pulse_count += 1
        delta_t = now_t - card.last_review_t if card.last_review_t is not None else 1000.0
        A_pre = card.A * math.exp(-delta_t / cfg.tau_A)
        R_des_pre = card.R_des * math.exp(-delta_t / cfg.tau_des)
        Phi_ref = 1.0 - R_des_pre

        if card.m_last is None or card.m_last == modality.value:
            Lambda_mod = 1.00
        elif modality == Modality.CONTEXT_GEN:
            Lambda_mod = cfg.lambda_gen
        else:
            Lambda_mod = cfg.lambda_shift

        if grade == Grade.AGAIN:
            delta_E = 0.0
            card.micro_streak = 0
            card.D = max(1.0, min(10.0, card.D + cfg.delta_d * 0.5))
        else:
            q_g = cfg.q_weights.get(grade.value, 1.0)
            # Continuous fatigue modulation applies consistently (C-72)
            Omega = user.psi_enc * user.V * (1.0 - 0.50 * user.F)
            delta_E = Omega * q_g * Lambda_mod * Phi_ref * (cfg.alpha_base + cfg.alpha_res * A_pre)
            card.micro_streak += 1

        card.E = card.E * math.exp(-delta_t / cfg.tau_E) + delta_E
        card.A = min(1.0, A_pre + cfg.gamma_A * Phi_ref)
        card.R_des = min(1.0, R_des_pre + cfg.gamma_des)
        card.m_last = modality.value
        card.last_review_t = now_t
        card.refractory_until = now_t + cfg.delta_t_soft

        interval = cfg.optimal_interval_min
        card.optimal_due_at = now_t + interval

        if card.state == CardState.NEW:
            card.state = CardState.MICRO_PULSING
        elif card.state == CardState.MICRO_PULSING:
            # Biochemical hard threshold graduation
            if card.E >= cfg.Theta_CREB and card.micro_streak >= 2:
                card.state = CardState.SLEEP_PENDING
                if card.graduated_day is None:
                    card.graduated_at_pulse = card.pulse_count
                    card.graduated_day = day
                card.is_session_deferred = False
            elif "D" in group and check_optimal_stop_rel(card, user, cfg):
                card.is_session_deferred = True
        elif card.state == CardState.RELEARNING:
            if card.E >= cfg.Theta_CREB and card.micro_streak >= 2:
                card.state = CardState.REVIEW
                base_s = max(card.S or 1.0, (card.S_pre_lapse or 1.0) * 0.78)
                card.S = min(cfg.S_max, base_s * ((card.E / cfg.Theta_CREB) ** (0.5 * cfg.beta_E)))
                q_cal = math.exp(get_effective_theta_calib(user, cfg, group))
                card.scheduled_due_day = day + max(1, round((card.S or 1.0) * q_cal))
                card.is_session_deferred = False

        return delta_E

    return 0.0

def update_theta_calib_cumulative_map(
    user: UserState,
    new_probe_obs: List[Tuple[float, float, int, float]],
    cfg: EngineConfig,
    max_iter: int = 8
) -> float:
    """
    Non-drifting Cumulative MAP Fisher-Scoring estimator for `theta_calib` (v1.15, §11.1, C-95):
    - Accumulates all cross-day probe observations across days into `user.cum_probe_obs` so the
      Gaussian prior `N(0, sigma_0^2)` (`lambda_0 = 1/sigma_0^2 = 5.60`) is applied ONCE to the
      cumulative log-posterior rather than pulling `theta_calib` toward 0 on every daily call.
    - Uses exact Fisher weights `I_j(theta) = (dP_j/dtheta)^2 / [P_j * (1 - P_j)]` with
      exact analytic derivative `dR_j/dtheta = 0.5 * R_j * (1 - R_j^2)` (v1.14 C-87).
    - Updates `user.cum_fisher_info` and Cramér–Rao lower bound `user.theta_posterior_sd`
      for G6 CRLB uncertainty gating (`get_effective_theta_calib`), and clips within `[-1.20, +1.20]`.
    """
    if not new_probe_obs:
        return user.theta_calib
    user.cum_probe_obs.extend(new_probe_obs)

    prior_prec = cfg.probe_shrinkage_n0 * 0.35  # lambda_0 = 16.0 * 0.35 = 5.60 (sigma_0 ≈ 0.4226)
    theta = user.theta_calib
    for _ in range(max_iter):
        q_cur = math.exp(theta)
        grad_sum = 0.0
        fisher_sum = 0.0
        for dt_days, s_card, mod_val, y_obs in user.cum_probe_obs:
            r_j = memory_retrievability(dt_days, max(0.1, s_card), q_cur)
            p_j = max(1e-5, min(1.0 - 1e-5, predict_p_correct(r_j, mod_val, cfg)))
            g_m = cfg.guess_rates.get(mod_val, 0.0)
            s_m = cfg.slip_rates.get(mod_val, 0.10)
            r_tilde = min(1.0 - 1e-6, max(1e-6, (p_j - g_m) / max(1e-9, 1.0 - g_m - s_m)))
            # Exact analytic chain rule: dP_j/dtheta = (1 - g_m - s_m) * r_tilde * (1 - r_tilde) * 0.5 * gamma * (1 + R_j)
            gamma = getattr(cfg, "gamma_logit", 1.0)
            dp_dtheta = (1.0 - g_m - s_m) * r_tilde * (1.0 - r_tilde) * 0.5 * gamma * (1.0 + r_j)
            var_y = max(1e-5, p_j * (1.0 - p_j))
            grad_sum += (y_obs - p_j) * (dp_dtheta / var_y)
            fisher_sum += (dp_dtheta * dp_dtheta) / var_y

        map_grad = grad_sum - prior_prec * theta
        hess = max(1e-6, fisher_sum + prior_prec)
        step = max(-0.35, min(0.35, map_grad / hess))
        theta = min(1.20, max(-1.20, theta + step))
        if abs(step) < 1e-8:
            break

    # Compute final cumulative Fisher information and posterior CRLB SD at converged MAP estimate
    q_final = math.exp(theta)
    final_fisher = 0.0
    for dt_days, s_card, mod_val, _ in user.cum_probe_obs:
        r_j = memory_retrievability(dt_days, max(0.1, s_card), q_final)
        p_j = max(1e-5, min(1.0 - 1e-5, predict_p_correct(r_j, mod_val, cfg)))
        g_m = cfg.guess_rates.get(mod_val, 0.0)
        s_m = cfg.slip_rates.get(mod_val, 0.10)
        r_tilde = min(1.0 - 1e-6, max(1e-6, (p_j - g_m) / max(1e-9, 1.0 - g_m - s_m)))
        gamma = getattr(cfg, "gamma_logit", 1.0)
        dp_dtheta = (1.0 - g_m - s_m) * r_tilde * (1.0 - r_tilde) * 0.5 * gamma * (1.0 + r_j)
        final_fisher += (dp_dtheta * dp_dtheta) / max(1e-5, p_j * (1.0 - p_j))

    user.cum_fisher_info = final_fisher
    user.theta_posterior_sd = 1.0 / math.sqrt(max(1e-6, final_fisher + prior_prec))
    user.theta_calib = theta
    return theta
