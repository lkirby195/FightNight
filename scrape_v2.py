"""v2 Stage-2 feature pipeline.

Stage 1 (untouched): engine.py Glicko-2 replay, tuned params from best_params.json.
Stage 2: strictly pre-fight rolling features per fighter, joined to each bout.

LEAKAGE CONTROLS
  1. Every rolling stat is accumulated AFTER a fight is emitted, never before.
  2. UFC Stats lists the winner first in every bout row. Orientation is
     de-biased deterministically (hash of fight_id) so the label is not
     recoverable from column order.
  3. Career aggregates on fighter pages (SLpM, Str.Acc, TD Avg...) are
     career-to-date-of-scrape and are NEVER used. We recompute our own.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date

import engine

# ---- apply tuned params to the frozen engine ----------------------------
BP = json.load(open("best_params.json"))
engine.TAU = BP["tau"]
engine.SIGMA0 = BP["sigma0"]
engine.S_TABLE["U-DEC"] = (BP["s_udec"], 1 - BP["s_udec"])
engine.S_TABLE["M-DEC"] = (BP["s_udec"], 1 - BP["s_udec"])
engine.S_TABLE["S-DEC"] = (BP["s_sdec"], 1 - BP["s_sdec"])
engine.S_TABLE["DQ"] = (BP["s_sdec"], 1 - BP["s_sdec"])

FINISH = {"KO/TKO", "SUB"}
DEC = {"U-DEC", "M-DEC", "S-DEC"}


def flip(fight_id: str) -> bool:
    """Deterministic 50/50 orientation swap keyed on fight id."""
    return int(hashlib.sha1(fight_id.encode()).hexdigest(), 16) % 2 == 1


def duration_min(rnd, time_str: str) -> float:
    try:
        r = int(rnd)
        mm, ss = time_str.split(":")
        return (r - 1) * 5.0 + int(mm) + int(ss) / 60.0
    except Exception:
        return 5.0


class Career:
    """Running pre-fight totals for one fighter."""

    __slots__ = ("min", "sl", "sa", "kd", "kd_a", "td", "td_a", "sub", "sub_a",
                 "n", "w", "l", "fin_w", "fin_l", "streak", "last",
                 "dmin", "ss_l_d", "ss_att", "opp_ss_l", "opp_ss_att",
                 "td_att", "opp_td_l", "opp_td_att", "ctrl", "ctrled",
                 "head_l", "leg_l", "ground_l")

    def __init__(self):
        self.min = self.sl = self.sa = 0.0
        self.kd = self.kd_a = self.td = self.td_a = self.sub = self.sub_a = 0
        self.n = self.w = self.l = self.fin_w = self.fin_l = 0
        self.streak = 0
        self.last: date | None = None
        # pass C accumulators; dmin = minutes covered by detail data
        self.dmin = 0.0
        self.ss_l_d = self.ss_att = self.opp_ss_l = self.opp_ss_att = 0.0
        self.td_att = self.opp_td_l = self.opp_td_att = 0.0
        self.ctrl = self.ctrled = 0.0
        self.head_l = self.leg_l = self.ground_l = 0.0

    def per15(self, x):
        return 15.0 * x / self.min if self.min > 0 else None

    def snapshot(self, d: date, dob: date | None, height, reach, stance):
        exp = self.n
        return {
            "exp": exp,
            "slpm": (self.sl / self.min if self.min else None),
            "sapm": (self.sa / self.min if self.min else None),
            "kd15": self.per15(self.kd),
            "kda15": self.per15(self.kd_a),
            "td15": self.per15(self.td),
            "tda15": self.per15(self.td_a),
            "sub15": self.per15(self.sub),
            "winrate": (self.w / exp if exp else None),
            "finrate": (self.fin_w / self.w if self.w else 0.0) if exp else None,
            "finishedrate": (self.fin_l / self.l if self.l else 0.0) if exp else None,
            "streak": self.streak,
            "layoff": ((d - self.last).days / 365.25 if self.last else None),
            "age": ((d - dob).days / 365.25 if dob else None),
            "height": height,
            "reach": reach,
            "stance": stance,
            # pass C rates (None until detail-covered minutes exist)
            "ss_acc": (self.ss_l_d / self.ss_att if self.ss_att else None),
            "ss_def": (1.0 - self.opp_ss_l / self.opp_ss_att if self.opp_ss_att else None),
            "td_acc": (self.td / self.td_att if self.td_att else None),
            "td_def": (1.0 - self.opp_td_l / self.opp_td_att if self.opp_td_att else None),
            "ctrl15": (15.0 * self.ctrl / self.dmin if self.dmin else None),
            "ctrled15": (15.0 * self.ctrled / self.dmin if self.dmin else None),
            "pace15": (15.0 * self.ss_att / self.dmin if self.dmin else None),
            "head_share": (self.head_l / self.ss_l_d if self.ss_l_d else None),
            "leg_share": (self.leg_l / self.ss_l_d if self.ss_l_d else None),
            "ground_share": (self.ground_l / self.ss_l_d if self.ss_l_d else None),
        }

    def update(self, mins, sl, sa, kd, kd_a, td, td_a, sub, won, drew, method, d,
               det=None):
        self.min += mins
        self.sl += sl or 0
        self.sa += sa or 0
        self.kd += kd or 0
        self.kd_a += kd_a or 0
        self.td += td or 0
        self.td_a += td_a or 0
        self.sub += sub or 0
        self.n += 1
        self.last = d
        if det is not None:
            self.dmin += mins
            self.ss_l_d += det["ss_l"] or 0
            self.ss_att += det["ss_a"] or 0
            self.opp_ss_l += det["opp_ss_l"] or 0
            self.opp_ss_att += det["opp_ss_a"] or 0
            self.td_att += det["td_a"] or 0
            self.opp_td_l += det["opp_td_l"] or 0
            self.opp_td_att += det["opp_td_a"] or 0
            self.ctrl += (det["ctrl_s"] or 0) / 60.0
            self.ctrled += (det["opp_ctrl_s"] or 0) / 60.0
            self.head_l += det["head_l"] or 0
            self.leg_l += det["leg_l"] or 0
            self.ground_l += det["ground_l"] or 0
        if drew:
            self.streak = 0
            return
        if won:
            self.w += 1
            self.streak = self.streak + 1 if self.streak > 0 else 1
            if method in FINISH:
                self.fin_w += 1
        else:
            self.l += 1
            self.streak = self.streak - 1 if self.streak < 0 else -1
            if method in FINISH:
                self.fin_l += 1


def load():
    details = {}
    try:
        for r in csv.DictReader(open("data/fight_details_v2.csv")):
            details[r["fight_id"]] = r
    except FileNotFoundError:
        pass
    fights = list(csv.DictReader(open("data/fights_v2.csv")))
    fights.sort(key=lambda r: (r["event_date"], int(r["bout_order"])))
    fighters = {}
    for r in csv.DictReader(open("data/fighters_v2.csv")):
        fighters[r["fighter_id"]] = {
            "dob": date.fromisoformat(r["dob"]) if r.get("dob") else None,
            "height": float(r["height_in"]) if r.get("height_in") else None,
            "reach": float(r["reach_in"]) if r.get("reach_in") else None,
            "stance": r.get("stance") or None,
        }
    return fights, fighters, details


def build():
    fights, fighters, details = load()
    eng = engine.Engine()
    car = defaultdict(Career)
    rows = []

    for f in fights:
        d = date.fromisoformat(f["event_date"])
        aid, bid = f["fighter_a_id"], f["fighter_b_id"]
        meta_a = fighters.get(aid, {})
        meta_b = fighters.get(bid, {})

        # --- Stage 1 pre-fight Glicko snapshot -------------------------
        fa = eng.fighter(aid, f["fighter_a"])
        fb = eng.fighter(bid, f["fighter_b"])
        n_a = (d - fa.last_fight).days / engine.YEAR if fa.last_fight else 0.0
        n_b = (d - fb.last_fight).days / engine.YEAR if fb.last_fight else 0.0
        pa = engine.PreState(fa.mu, engine.inflate(fa.phi, fa.sigma, max(n_a, 0)), fa.sigma)
        pb = engine.PreState(fb.mu, engine.inflate(fb.phi, fb.sigma, max(n_b, 0)), fb.sigma)
        p_glicko_a = engine.predict(pa, pb)

        sa_ = car[aid].snapshot(d, meta_a.get("dob"), meta_a.get("height"),
                               meta_a.get("reach"), meta_a.get("stance"))
        sb_ = car[bid].snapshot(d, meta_b.get("dob"), meta_b.get("height"),
                               meta_b.get("reach"), meta_b.get("stance"))

        outcome, method = f["outcome"], f["method"]
        if outcome in ("A_WIN", "DRAW") and method not in ("Overturned", "CNC"):
            swap = flip(f["fight_id"])
            X, Y = (sb_, sa_) if swap else (sa_, sb_)
            gx = (1 - p_glicko_a) if swap else p_glicko_a
            rx, ry = (pb, pa) if swap else (pa, pb)
            label = 0.5 if outcome == "DRAW" else (0 if swap else 1)
            rows.append(dict(
                fight_id=f["fight_id"], date=f["event_date"], method=method,
                weight_class=f["weight_class"], title=f["title_bout"],
                label=label, p_glicko=gx,
                rating_diff=(rx.mu - ry.mu) * engine.SCALE,
                rd_x=rx.phi * engine.SCALE, rd_y=ry.phi * engine.SCALE,
                **{f"x_{k}": v for k, v in X.items()},
                **{f"y_{k}": v for k, v in Y.items()},
            ))

        # --- accumulate AFTER emitting ---------------------------------
        mins = duration_min(f["round"], f["time"])
        num = lambda k: float(f[k]) if f[k] not in ("", None) else 0.0
        won_a = outcome == "A_WIN"
        drew = outcome == "DRAW"
        if outcome != "NC" and method not in ("Overturned", "CNC"):
            det = details.get(f["fight_id"])
            det_a = det_b = None
            if det:
                dv = lambda k: float(det[k]) if det.get(k) not in ("", None) else None
                det_a = {"ss_l": dv("a_ss_l"), "ss_a": dv("a_ss_a"),
                         "opp_ss_l": dv("b_ss_l"), "opp_ss_a": dv("b_ss_a"),
                         "td_a": dv("a_td_a"), "opp_td_l": dv("b_td_l"),
                         "opp_td_a": dv("b_td_a"), "ctrl_s": dv("a_ctrl_s"),
                         "opp_ctrl_s": dv("b_ctrl_s"), "head_l": dv("a_head_l"),
                         "leg_l": dv("a_leg_l"), "ground_l": dv("a_ground_l")}
                det_b = {"ss_l": dv("b_ss_l"), "ss_a": dv("b_ss_a"),
                         "opp_ss_l": dv("a_ss_l"), "opp_ss_a": dv("a_ss_a"),
                         "td_a": dv("b_td_a"), "opp_td_l": dv("a_td_l"),
                         "opp_td_a": dv("a_td_a"), "ctrl_s": dv("b_ctrl_s"),
                         "opp_ctrl_s": dv("a_ctrl_s"), "head_l": dv("b_head_l"),
                         "leg_l": dv("b_leg_l"), "ground_l": dv("b_ground_l")}
            car[aid].update(mins, num("a_sig_str"), num("b_sig_str"), num("a_kd"),
                            num("b_kd"), num("a_td"), num("b_td"), num("a_sub_att"),
                            won_a, drew, method, d, det_a)
            car[bid].update(mins, num("b_sig_str"), num("a_sig_str"), num("b_kd"),
                            num("a_kd"), num("b_td"), num("a_td"), num("b_sub_att"),
                            not won_a and not drew, drew, method, d, det_b)
        eng.process({
            "event_date": f["event_date"], "fighter_a_id": aid, "fighter_a": f["fighter_a"],
            "fighter_b_id": bid, "fighter_b": f["fighter_b"],
            "outcome": outcome, "method": method,
            "a_missed_weight": f.get("a_mw", ""), "b_missed_weight": f.get("b_mw", ""),
        })

    cols = list(rows[0].keys())
    with open("data/features_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote data/features_v2.csv  n={len(rows)}  cols={len(cols)}")


if __name__ == "__main__":
    build()
