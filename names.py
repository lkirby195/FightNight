"""Fighter-name resolution shared by card_report (and card_settle through it),
clv_eval, betting_system and site/build.py.

BestFightOdds and UFC Stats spell fighters differently. Every comparison goes
through the same steps, in this order:

  1. alias   data/name_aliases.csv (bfo_name -> ufcstats_name) rewrites a BFO
             spelling to the UFC Stats one ("Patricio Freire" -> "Patricio
             Pitbull"). Exact match on the raw BFO string; applied to BFO
             names only (bfo_norm / bfo_lastn), never to UFC Stats names.
             A row with an EMPTY ufcstats_name says the BFO name is a
             different person from any near-miss on file (a debutant who
             shares a surname): it is never matched and never raises.
  2. norm    ASCII-fold, lower-case, letters and spaces only. Hyphens become
             spaces ("Cortes-Acosta" == "Cortes Acosta") or, with hyphen="",
             nothing ("Sangcha-An" == "Sangcha'an", "Al-Hassan" == "Alhassan").
  3. suffix  a trailing generational suffix (Jr, Sr, II, III, IV) is dropped
             on BOTH sides ("Michael Aswell Jr." == "Michael Aswell",
             "Sean King III" == "Sean King").
  4. find    make_finder(): exact name, then the space-less joined form.
             Nothing else: there is no fuzzy auto-accept (removed 2026-10-22;
             it used to take the same surname and first three letters).
  5. resolve make_resolver(): a bout's two names at once. When one name
             resolves and the other does not, and the other is within
             NEAR_MISS of a fighter on file (difflib ratio over the normalised
             names), that is a spelling miss, not a debut: NameJoinError is
             raised with the candidate and nothing is guessed. When neither
             resolves, the bout is a likely debut and resolves to (None, None)
             without error (signals.classify then says projectable=False,
             reason "debut"). Aliases are added by hand, in the CSV.

Adding an alias is a data change (one CSV row), not a code change. Anything
that changes norm() moves historical joins: re-run clv_eval.py and diff
data/bfo_joined.csv before trusting the record again.

clv_eval.join_bfo applies the same near-miss test to every BFO bout, over
the full history (the 2026-10-22 gate was lifted on 2026-10-26 once the 46
historical misspellings it had stopped on were aliased by hand: Cattar /
Kattar, Curelaba / Cutelaba, Oliynyk / Oleinik, Waterson / Waterson-Gomez,
...). A new misspelling anywhere in the history stops clv_eval until a row is
added.
"""
from __future__ import annotations

import csv
import difflib
import os
import re
import unicodedata

ALIASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "name_aliases.csv")
SUFFIX_RE = re.compile(r" (?:jr|sr|ii|iii|iv)$")
NEAR_MISS = 0.85                        # difflib ratio: at or above, a miss is an error
_aliases: dict[str, str] | None = None


class NameJoinError(SystemExit):
    """A BFO name matched no fighter exactly but is a near-miss of one on
    file. Never resolved automatically: the message names the candidate and
    the alias row that would accept it."""


def load_aliases(path: str = ALIASES) -> dict[str, str]:
    """bfo_name -> ufcstats_name from `path` ("" = distinct person, see the
    module docstring); empty when the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["bfo_name"].strip(): (r.get("ufcstats_name") or "").strip()
                for r in csv.DictReader(fh) if (r.get("bfo_name") or "").strip()}


def _table() -> dict[str, str]:
    global _aliases
    if _aliases is None:
        _aliases = load_aliases()
    return _aliases


def alias(name: str) -> str:
    """The UFC Stats spelling of a BFO name when data/name_aliases.csv lists
    it, else the name unchanged (also unchanged for a distinct-person row)."""
    return _table().get(str(name).strip(), "") or name


def distinct(name: str) -> bool:
    """True when data/name_aliases.csv says this BFO name is a different
    person from any near-miss on file (empty ufcstats_name)."""
    t = _table()
    k = str(name).strip()
    return k in t and t[k] == ""


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


def ratio(a: str, b: str) -> float:
    """Similarity of two normalised names (difflib, 0..1); the joined forms
    are tried too so "Su Mudaerji" / "Sumudaerji" score as the same string."""
    r = difflib.SequenceMatcher(None, a, b).ratio()
    aj, bj = a.replace(" ", ""), b.replace(" ", "")
    return max(r, difflib.SequenceMatcher(None, aj, bj).ratio())


def near_misses(name: str, fighters, threshold: float = NEAR_MISS):
    """Fighters on file whose normalised name is within `threshold` of the
    BFO name (aliases applied) -> [(fighter_id, name, ratio)], best first.
    Empty for a distinct-person alias row."""
    if distinct(name):
        return []
    n = bfo_norm(name)
    if not n:
        return []
    out = [(fid, fname, ratio(n, norm(fname))) for fid, fname in fighters]
    out = [x for x in out if x[2] >= threshold]
    out.sort(key=lambda x: -x[2])
    return out


def make_finder(fighters):
    """fighters: iterable of (fighter_id, UFC Stats name). -> find(bfo_name)
    returning the fighter_id or None. Exact normalised name, then the
    space-less joined form ("Sangcha-An" vs "Sangcha'an"); no fuzzy step."""
    byname, byjoined = {}, {}
    for fid, name in fighters:
        n = norm(name)
        byname.setdefault(n, fid)
        byjoined.setdefault(n.replace(" ", ""), fid)

    def find(name):
        n = bfo_norm(name)
        if not n:
            return None
        if n in byname:
            return byname[n]
        return byjoined.get(n.replace(" ", ""))
    return find


def make_resolver(fighters):
    """fighters: iterable of (fighter_id, UFC Stats name). -> resolve(n1, n2)
    returning (id1, id2) for a BFO bout, either id None when that fighter is
    not on file. Raises NameJoinError when exactly one name resolves and the
    other is a near-miss of a fighter on file (a spelling miss is never
    accepted and never silently called a debut)."""
    fl = list(fighters)
    find = make_finder(fl)

    def resolve(n1, n2):
        i1, i2 = find(n1), find(n2)
        if (i1 is None) != (i2 is None):
            miss, other = (n1, n2) if i1 is None else (n2, n1)
            c = near_misses(miss, fl)
            if c:
                fid, fname, r = c[0]
                raise NameJoinError(
                    f"name join: BFO '{miss}' (vs {other}) matches no fighter exactly "
                    f"but is {r:.2f} to '{fname}' ({fid}). Not resolved automatically: "
                    f"if it is the same person add the row '{miss},{fname}' to "
                    f"data/name_aliases.csv; if it is a different person add "
                    f"'{miss},' (empty ufcstats_name).")
        return i1, i2
    return resolve


def near_miss_pair(n1: str, n2: str, fa: str, fb: str, threshold: float = NEAR_MISS):
    """The same test for a bout-level join: BFO names (n1, n2) against a
    fights_v2 bout (fa, fb). -> (bfo_name, candidate, ratio) when one BFO
    name equals one fighter exactly and the other is a near-miss of the other
    fighter, else None."""
    b1, b2 = bfo_norm(n1), bfo_norm(n2)
    a, b = norm(fa), norm(fb)
    for bx, by, fx, fy, raw_y in ((b1, b2, a, b, n2), (b1, b2, b, a, n2),
                                  (b2, b1, a, b, n1), (b2, b1, b, a, n1)):
        if bx == fx and by != fy and not distinct(raw_y):
            r = ratio(by, fy)
            if r >= threshold:
                return raw_y, fb if fy == b else fa, r
    return None
