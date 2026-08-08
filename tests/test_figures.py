"""Tests for openscope_ccf.figures._spontaneous_start window validation (no network).

These pin the fix that a stimulus-free LFP window must FULLY fit within the actual
recording range — otherwise Welch would run on a short/empty slice. The fake NWB
handle mimics only the interface _spontaneous_start touches.
"""
import numpy as np

from openscope_ccf.figures import _spontaneous_start


class _Grp:
    def __init__(self, start, stop):
        self._d = {"start_time": np.asarray(start, float), "stop_time": np.asarray(stop, float)}

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]


class _Iv(dict):
    pass


class _FH:
    """Minimal fake: fh.get('intervals') -> mapping of '<name>_presentations' -> group."""
    def __init__(self, blocks):
        # blocks: list of (onset, stop) per stimulus block
        iv = _Iv()
        for i, (o, s) in enumerate(blocks):
            iv[f"block{i}_presentations"] = _Grp([o], [s])
        self._iv = iv

    def get(self, k):
        return self._iv if k == "intervals" else None


def test_pre_stimulus_leadin_chosen_when_long_enough():
    # first onset at 40 s, window 30 s, recording [0, 200] -> lead-in [0,40] fits
    fh = _FH([(40.0, 100.0), (120.0, 180.0)])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=200.0) == 0.0


def test_interblock_gap_chosen_when_leadin_too_short():
    # lead-in only 5 s (too short); inter-block gap [100,140]=40 s fits
    fh = _FH([(5.0, 100.0), (140.0, 180.0)])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=200.0) == 100.0


def test_tail_chosen_only_if_room_remains():
    # lead-in short, gap short (10 s), but tail after 150 s to t_hi=200 = 50 s fits
    fh = _FH([(5.0, 100.0), (110.0, 150.0)])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=200.0) == 150.0


def test_returns_none_when_no_full_window_exists():
    # lead-in 5 s, gap 10 s, tail only 5 s (last stop 150, t_hi 155) -> nothing fits 30 s
    fh = _FH([(5.0, 100.0), (110.0, 150.0)])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=155.0) is None


def test_returns_none_when_recording_shorter_than_window():
    fh = _FH([(5.0, 20.0)])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=25.0) is None


def test_no_intervals_returns_tlo_when_long_enough():
    fh = _FH([])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=200.0) == 0.0


def test_no_intervals_but_recording_too_short_returns_none():
    fh = _FH([])
    assert _spontaneous_start(fh, 30.0, t_lo=0.0, t_hi=10.0) is None
