"""MMA Rating Engine - Glicko-2 adaptation per spec v1.0.

One fight = one rating period. Time-based RD inflation with standard
downtime = 1 year. Graded outcome scores by finish type. Missed-weight
asymmetric credit (supported; flags default False pending weigh-in data).
Deterministic, chronological, snapshot-semantics replay.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date

SCALE = 173.7178
R0, RD0, SIGMA0 = 1500.0, 350.0, 0.06
TAU = 0.5
EPS = 1e-6
PHI_CAP = RD0 / SCALE
YEAR = 365.25

# Graded scores: method -> (winner S, loser S)
S_TABLE = {
    "KO/TKO": (1.00, 0.00),
    "SUB":    (1.00, 0.00),
    "U-DEC":  (0.90, 0.10),
    "M-DEC":  (0.90, 0.10),   # majority decision scored as unanimous per spec
    "S-DEC":  (0.75, 0.25),
    "DQ":     (0.75, 0.25),   # low-information win
}
DRAW_S = {
    "M-DEC": (0.50, 0.50),    # spec says 0.55/0.45 but data lacks judge attribution
    "S-DEC": (0.50, 0.50),
    "U-DEC": (0.50, 0.50),
    "Other": (0.50, 0.50),
}
# Missed-weight parameters (spec 5.3)
MW_OFFENDER_WIN_DISCOUNT = 0.80
MW_VICTIM_WIN_BONUS = 1.10
MW_VICTIM_LOSS_FLOOR = 0.10


@dataclass
class Fighter:
    fid: str
    name: str
    mu: float = 0.0
    phi: float = RD0 / SCALE
    sigma: float = SIGMA0
    last_fight: date | None = None
    fights: int = 0
    wins: int = 0
    losses: int = 0
    history: list = field(default_factory=list)

    @property
    def rating(self) -> float:
        return SCALE * self.mu + R0

    @property
    def rd(self) -> float:
        return SCALE * self.phi


def g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def expect(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))


def inflate(phi: float, sigma: float, n_years: float) -> float:
    """Pre-fight RD inflation: sqrt(phi^2 + sigma^2 * n), capped at debutant."""
    return min(math.sqrt(phi * phi + sigma * sigma * n_years), PHI_CAP)


def new_sigma(phi: float, v: float, delta: float, sigma: float) -> float:
    """Glickman's Illinois-algorithm volatility update."""
    a = math.log(sigma * sigma)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (TAU * TAU)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * TAU) < 0 and k < 200:
            k += 1
        B = a - k * TAU
    fA, fB = f(A), f(B)
    iters = 0
    while abs(B - A) > EPS and iters < 100:
        iters += 1
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def glicko2_update(mu, phi, sigma, mu_j, phi_j, s):
    """Single-opponent Glicko-2 update. Returns (mu', phi', sigma')."""
    gj = g(phi_j)
    E = expect(mu, mu_j, phi_j)
    v = 1.0 / (gj * gj * E * (1.0 - E))
    delta = v * gj * (s - E)
    sig_p = new_sigma(phi, v, delta, sigma)
    phi_star = math.sqrt(phi * phi + sig_p * sig_p)
    phi_p = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_p = mu + phi_p * phi_p * gj * (s - E)
    return mu_p, phi_p, sig_p


def predict(fa: "PreState", fb: "PreState") -> float:
    """P(A wins) from pre-fight time-inflated states, combined uncertainty."""
    combined = math.sqrt(fa.phi ** 2 + fb.phi ** 2)
    return 1.0 / (1.0 + math.exp(-g(combined) * (fa.mu - fb.mu)))


@dataclass
class PreState:
    mu: float
    phi: float
    sigma: float


def scores_for(outcome: str, method: str, a_missed: bool, b_missed: bool):
    """Return (s_a, s_b, bonus_a, bonus_b) or None for no-update (NC).

    First-listed fighter (A) is the winner when outcome == A_WIN.
    bonus_* is a multiplier on the rating change (mu delta) only.
    """
    if method in ("Overturned", "CNC") or outcome == "NC":
        return None
    if outcome == "DRAW":
        s_a, s_b = DRAW_S.get(method, (0.50, 0.50))
        return s_a, s_b, 1.0, 1.0
    if outcome != "A_WIN":
        return None
    if method not in S_TABLE:
        return None
    s_w, s_l = S_TABLE[method]
    bonus_a = bonus_b = 1.0
    # A won, B lost. Missed-weight asymmetry (spec 5.3):
    if a_missed and not b_missed:
        # offender (A) won: discount; victim (B) loss mirrored so pair sums to 1
        s_w = s_w * MW_OFFENDER_WIN_DISCOUNT
        s_l = 1.0 - s_w
    elif b_missed and not a_missed:
        # victim (A) won: full S + bonus multiplier; offender (B) full penalty
        bonus_a = MW_VICTIM_WIN_BONUS
        # victim loss floor does not apply (victim won)
    # if both missed or neither: rate normally
    return s_w, s_l, bonus_a, bonus_b


def scores_for_loss_floor(s_l: float, loser_is_victim: bool) -> float:
    return max(s_l, MW_VICTIM_LOSS_FLOOR) if loser_is_victim else s_l


class Engine:
    def __init__(self):
        self.fighters: dict[str, Fighter] = {}
        self.predictions: list = []  # (date, p_a_wins, outcome, method, a, b)

    def fighter(self, fid: str, name: str) -> Fighter:
        if fid not in self.fighters:
            self.fighters[fid] = Fighter(fid=fid, name=name)
        f = self.fighters[fid]
        f.name = name  # keep freshest spelling
        return f

    def process(self, fight: dict):
        d = date.fromisoformat(fight["event_date"])
        fa = self.fighter(fight["fighter_a_id"], fight["fighter_a"])
        fb = self.fighter(fight["fighter_b_id"], fight["fighter_b"])

        # Step 0: pre-fight snapshot with time inflation
        n_a = (d - fa.last_fight).days / YEAR if fa.last_fight else 0.0
        n_b = (d - fb.last_fight).days / YEAR if fb.last_fight else 0.0
        pa = PreState(fa.mu, inflate(fa.phi, fa.sigma, max(n_a, 0.0)), fa.sigma)
        pb = PreState(fb.mu, inflate(fb.phi, fb.sigma, max(n_b, 0.0)), fb.sigma)

        # record prediction before updating (for backtest)
        self.predictions.append({
            "date": d, "p_a": predict(pa, pb), "outcome": fight["outcome"],
            "method": fight["method"], "a": fa.fid, "b": fb.fid,
            "a_fights": fa.fights, "b_fights": fb.fights,
        })

        a_mw = fight.get("a_missed_weight", "") in ("1", "True", "true")
        b_mw = fight.get("b_missed_weight", "") in ("1", "True", "true")
        res = scores_for(fight["outcome"], fight["method"], a_mw, b_mw)

        if res is None:
            # NC: no rating update; activity clock advances (spec 5.1)
            fa.last_fight = d
            fb.last_fight = d
            return

        s_a, s_b, bonus_a, bonus_b = res
        # victim loss floor (loser was the victim of a miss)
        if fight["outcome"] == "A_WIN":
            s_b = scores_for_loss_floor(s_b, loser_is_victim=(a_mw and not b_mw))

        # symmetric updates from the same snapshot
        mu_a, phi_a, sig_a = glicko2_update(pa.mu, pa.phi, pa.sigma, pb.mu, pb.phi, s_a)
        mu_b, phi_b, sig_b = glicko2_update(pb.mu, pb.phi, pb.sigma, pa.mu, pa.phi, s_b)

        # missed-weight victim win bonus: scales mu delta only (spec 6 step 4b)
        mu_a = pa.mu + bonus_a * (mu_a - pa.mu)
        mu_b = pb.mu + bonus_b * (mu_b - pb.mu)

        for f, mu, phi, sig, s in ((fa, mu_a, phi_a, sig_a, s_a), (fb, mu_b, phi_b, sig_b, s_b)):
            f.mu, f.phi, f.sigma = mu, phi, sig
            f.last_fight = d
            f.fights += 1
        if fight["outcome"] == "A_WIN":
            fa.wins += 1
            fb.losses += 1
        fa.history.append((d.isoformat(), round(fa.rating, 1)))
        fb.history.append((d.isoformat(), round(fb.rating, 1)))
