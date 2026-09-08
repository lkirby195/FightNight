"""Stage 2: logistic regression on Glicko spine + pre-fight fight-level features.

Antisymmetry is enforced by construction: every feature is a DIFFERENCE
(x - y) or a symmetric aggregate. A model built on raw x_/y_ columns can
learn corner-order artifacts from noise; differences make f(A,B) = -f(B,A)
exact, so P(A) + P(B) = 1 by identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIRED = ["slpm", "sapm", "kd15", "kda15", "td15", "tda15", "sub15",
          "winrate", "finrate", "finishedrate", "streak", "layoff",
          "age", "height", "reach", "exp",
          # pass C
          "ss_acc", "ss_def", "td_acc", "td_def",
          "ctrl15", "ctrled15", "pace15",
          "head_share", "leg_share", "ground_share"]

TRAIN_END = "2023-01-01"
TUNE_END = "2025-01-01"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_matrix(df: pd.DataFrame):
    F = pd.DataFrame(index=df.index)

    # Stage 1 spine
    gl = logit(df.p_glicko)
    F["glicko_logit"] = gl
    F["rd_diff"] = (df.rd_x - df.rd_y) / 100.0

    # Uncertainty enters as SHRINKAGE, not as a main effect. A symmetric term
    # cannot inform an antisymmetric label; it can only modulate how much the
    # rating gap should be trusted. So it goes in as an interaction.
    rd_sum = (df.rd_x + df.rd_y) / 100.0
    F["gl_x_rdsum"] = gl * (rd_sum - rd_sum.mean())

    # debut / experience-shortage indicators (11.7% of rows have no prior fight)
    x_debut = df.x_exp.fillna(0).eq(0) | df.x_slpm.isna()
    y_debut = df.y_exp.fillna(0).eq(0) | df.y_slpm.isna()
    F["debut_diff"] = x_debut.astype(float) - y_debut.astype(float)

    # paired differences; missing -> 0 (neutral) with the indicator carrying it
    for c in PAIRED:
        xs, ys = df[f"x_{c}"], df[f"y_{c}"]
        med = pd.concat([xs, ys]).median()
        F[f"d_{c}"] = (xs.fillna(med) - ys.fillna(med)).astype(float)

    # experience is heavily skewed -> log scale, plus a symmetric "both green" term
    F["d_exp"] = np.log1p(df.x_exp.fillna(0)) - np.log1p(df.y_exp.fillna(0))
    exp_min = np.log1p(np.minimum(df.x_exp.fillna(0), df.y_exp.fillna(0)))
    F["gl_x_expmin"] = gl * (exp_min - exp_min.mean())

    # layoff: cap at 4y, ring-rust is not linear past that
    F["d_layoff"] = np.clip(F["d_layoff"], -4, 4)

    # stance matchup: southpaw edge, antisymmetric
    def sp(col):
        return df[col].fillna("").str.lower().str.startswith("southpaw").astype(float)
    def orth(col):
        return df[col].fillna("").str.lower().str.startswith("orthodox").astype(float)
    F["southpaw_edge"] = sp("x_stance") * orth("y_stance") - sp("y_stance") * orth("x_stance")

    F["title"] = 0.0  # symmetric; kept out of the antisymmetric core
    F = F.drop(columns=["title"])
    return F.fillna(0.0)


def load():
    df = pd.read_csv("data/features_v2.csv", parse_dates=["date"])
    df = df[df.label != 0.5].copy()          # drop 64 draws for fitting
    X = build_matrix(df)
    y = df.label.astype(int).values
    return df, X, y


def split(df):
    tr = df.date < TRAIN_END
    tu = (df.date >= TRAIN_END) & (df.date < TUNE_END)
    ho = df.date >= TUNE_END
    return tr.values, tu.values, ho.values
