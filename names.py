"""Fighter-name resolution shared by card_report (and card_settle through it),
clv_eval, betting_system and site/build.py.

BestFightOdds and UFC Stats spell fighters differently. Every comparison goes
through the same steps, in this order:

  1. alias   data/name_aliases.csv (bfo_name -> ufcstats_name) rewrites a BFO
             spelling to the UFC Stats one ("Patricio Freire" -> "Patricio
             Pitbull"). Exact match on the raw BFO string; applied to BFO
             names only (bfo_norm / bfo_lastn), never to UFC Stats names.
  2. norm    ASCII-fold, lower-case, letters and spaces only. Hyphens become
             spaces ("Cortes-Acosta" == "Cortes Acosta") or, with hyphen="",
             nothing ("Sangcha-An" == "Sangcha'an", "Al-Hassan" == "Alhassan").
  3. suffix  a trailing generational suffix (Jr, Sr, II, III, IV) is dropped
             on BOTH sides ("Michael Aswell Jr." == "Michael Aswell",
             "Sean King III" == "Sean King").
  4. find    make_finder(): exact name, then the space-less joined form, then
             the fuzzy fallback (same surname, same first three letters of the
             first name, exactly one candidate) -- card_report's find, unchanged.

Adding an alias is a data change (one CSV row), not a code change. Anything
that changes norm() moves historical joins: re-run clv_eval.py and diff
data/bfo_joined.csv before trusting the record again.
"""
from __future__ import annotations

import csv
import os
import re
import unicodedata

ALIASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "name_aliases.csv")
SUFFIX_RE = re.compile(r" (?:jr|sr|ii|iii|iv)$")
_aliases: dict[str, str] | None = None


def load_aliases(path: str = ALIASES) -> dict[str, str]:
    """bfo_name -> ufcstats_name from `path`; empty when the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["bfo_name"].strip(): r["ufcstats_name"].strip()
                for r in csv.DictReader(fh) if (r.get("bfo_name") or "").strip()}


def alias(name: str) -> str:
    """The UFC Stats spelling of a BFO name when data/name_aliases.csv lists
    it, else the name unchanged."""
    global _aliases
    if _aliases is None:
        _aliases = load_aliases()
    return _aliases.get(str(name).strip(), name)


def norm(s: str, hyphen: str = " ") -> str:
    """ASCII-fold, lower-case, letters and spaces only, generational suffix
    dropped. hyphen="" gives the joined form (see the module docstring)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", hyphen)).strip()
    return SUFFIX_RE.sub("", s)


def lastn(s: str, hyphen: str = " ") -> str:
    """Surname token of norm(s): the real surname, not "jr" or "iii"."""
    p = norm(s, hyphen).split()
    return p[-1] if p else ""


def bfo_norm(s: str, hyphen: str = " ") -> str:
    """norm() of a BestFightOdds name, aliases applied first."""
    return norm(alias(s), hyphen)


def bfo_lastn(s: str, hyphen: str = " ") -> str:
    """lastn() of a BestFightOdds name, aliases applied first."""
    return lastn(alias(s), hyphen)


def make_finder(fighters):
    """fighters: iterable of (fighter_id, UFC Stats name). -> find(bfo_name)
    returning the fighter_id or None (card_report.main's find, shared with
    site/build.py). Exact name first, then the space-less joined form
    ("Sangcha-An" vs "Sangcha'an"), then the fuzzy fallback: same surname and
    same first three letters of the first name, accepted only when exactly one
    fighter fits."""
    byname, byjoined = {}, {}
    for fid, name in fighters:
        n = norm(name)
        byname.setdefault(n, fid)
        byjoined.setdefault(n.replace(" ", ""), fid)

    def find(name):
        n = bfo_norm(name)
        if n in byname:
            return byname[n]
        if n.replace(" ", "") in byjoined:
            return byjoined[n.replace(" ", "")]
        t = n.split()
        if not t:
            return None
        c = [fid for nm, fid in byname.items()
             if nm.endswith(" " + t[-1]) and nm.split()[0][:3] == t[0][:3]]
        return c[0] if len(c) == 1 else None
    return find
