from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from config import CONFIG


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start: datetime
    end: datetime
    label: str


def generate_windows(start: datetime, end: datetime) -> List[TimeWindow]:
    windows: List[TimeWindow] = []
    cursor = start
    delta = timedelta(days=CONFIG.stage.window_days)
    step = timedelta(days=CONFIG.stage.step_days)
    idx = 1
    while cursor + delta <= end + timedelta(days=1):
        w_end = cursor + delta
        windows.append(
            TimeWindow(
                index=idx,
                start=cursor,
                end=w_end,
                label=f"time point {idx}",
            )
        )
        cursor += step
        idx += 1
    return windows


def filter_window(docs: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    return docs[(docs["date"] >= window.start) & (docs["date"] < window.end)]
