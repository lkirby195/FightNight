"""UFC Stats HTTP session with SHA-256 proof-of-work solver.

The site serves an interstitial that requires finding n such that
sha256(f"{nonce}:{n}") starts with `target` zeros, then POSTing
(nonce, n) to /__c to receive the clearance cookie.
"""
from __future__ import annotations

import hashlib
import re
import time

import requests

BASE = "http://ufcstats.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_NONCE_RE = re.compile(r'nonce\s*=\s*"([0-9a-f]+)"')
_TARGET_RE = re.compile(r"target\s*=\s*new Array\((\d+)\+1\)")
_CHALLENGE_MARK = "Checking your browser"


def _solve(nonce: str, zeros: int) -> int:
    prefix = "0" * zeros
    n = 0
    while True:
        if hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(prefix):
            return n
        n += 1


class UFCStatsSession:
    def __init__(self, delay: float = 0.4, retries: int = 6):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.delay = delay
        self.retries = retries
        self._last = 0.0

    def _throttle(self):
        gap = time.time() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.time()

    def _clear(self, html: str) -> bool:
        m_nonce = _NONCE_RE.search(html)
        m_target = _TARGET_RE.search(html)
        if not (m_nonce and m_target):
            return False
        n = _solve(m_nonce.group(1), int(m_target.group(1)))
        r = self.s.post(
            f"{BASE}/__c",
            data={"nonce": m_nonce.group(1), "n": n},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        return r.status_code < 300

    def get(self, url: str) -> str:
        if url.startswith("/"):
            url = BASE + url
        last_err = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                r = self.s.get(url, timeout=30)
                r.raise_for_status()
                if _CHALLENGE_MARK in r.text[:400]:
                    if not self._clear(r.text):
                        raise RuntimeError("PoW clearance rejected")
                    continue
                return r.text
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET failed after {self.retries} tries: {url} ({last_err})")
