"""Reporting periods.

  '14d'          fixed 14-day windows counted from period_anchor: [anchor + 14k, anchor + 14(k+1))
                 label 'YYYYMMDD' of the first day
  'semimonthly'  1st-15th and 16th-end of month (as in ai-cropgrowth): label 'YYYYMMH1' / 'YYYYMMH2'

Periods are half-open [start, end).
"""
from __future__ import annotations

from collections import namedtuple

import pandas as pd

CADENCES = ("14d", "semimonthly")
Period = namedtuple("Period", "start end label")


def period_of(date, cadence="14d", anchor="2026-01-01"):
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}")
    d = pd.Timestamp(date).normalize()
    if cadence == "14d":
        a = pd.Timestamp(anchor).normalize()
        k = (d - a).days // 14
        start = a + pd.Timedelta(days=14 * k)
        return Period(start, start + pd.Timedelta(days=14), start.strftime("%Y%m%d"))
    m0 = d.replace(day=1)
    m1 = m0 + pd.offsets.MonthBegin(1)
    mid = m0 + pd.Timedelta(days=15)
    if d < mid:
        return Period(m0, mid, m0.strftime("%Y%m") + "H1")
    return Period(mid, m1, m0.strftime("%Y%m") + "H2")


def last_complete_period(as_of, cadence="14d", anchor="2026-01-01"):
    """Newest period whose last day is <= as_of (data through as_of)."""
    d = pd.Timestamp(as_of).normalize()
    cur = period_of(d, cadence, anchor)
    if d + pd.Timedelta(days=1) >= cur.end:
        return cur
    return period_of(cur.start - pd.Timedelta(days=1), cadence, anchor)


def periods_between(start, end, cadence="14d", anchor="2026-01-01"):
    """Every period overlapping [start, end] (dates inclusive)."""
    out, p = [], period_of(start, cadence, anchor)
    end = pd.Timestamp(end).normalize()
    while p.start <= end:
        out.append(p)
        p = period_of(p.end, cadence, anchor)
    return out


def parse_label(label, cadence="14d", anchor="2026-01-01"):
    """'20260101' / '202601H2' -> Period."""
    label = str(label)
    if cadence == "semimonthly":
        d = pd.Timestamp(label[:6] + ("01" if label.endswith("H1") else "16"))
    else:
        d = pd.Timestamp(label)
    p = period_of(d, cadence, anchor)
    if p.label != label:
        raise ValueError(f"{label!r} is not a {cadence} period label (nearest: {p.label})")
    return p
