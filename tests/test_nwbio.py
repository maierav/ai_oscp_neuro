"""Tests for the units->electrodes row mapping (openscope_ccf.nwbio).

Uses a minimal in-memory fake that mimics the tiny slice of the h5py NWB API
``unit_electrode_rows`` touches — no network, no real file.
"""
import numpy as np
import pytest

from openscope_ccf.nwbio import unit_electrode_rows


class _DS:
    """Fake h5py dataset: supports ``[:]`` and ``[...]`` returning a numpy array."""
    def __init__(self, arr):
        self._a = np.asarray(arr)
    def __getitem__(self, k):
        return self._a[k]
    def __len__(self):
        return len(self._a)


class _FakeNWB:
    """Fake h5py.File exposing units + electrodes for unit_electrode_rows."""
    def __init__(self, group_name, device_name, eci):
        self._g = {
            "units": {
                "device_name": _DS(np.array(device_name, dtype=object)),
                "extremum_channel_index": _DS(np.array(eci)),
            },
            "general/extracellular_ephys/electrodes": {
                "group_name": _DS(np.array(group_name, dtype=object)),
            },
        }
    def __getitem__(self, key):
        return self._g[key]


def _fh(group_name, device_name, eci):
    return _FakeNWB([s.encode() for s in group_name],
                    [s.encode() for s in device_name], eci)


def test_offset_applied_per_probe():
    # two probes, 3 electrode rows each: A=[0,1,2], B=[3,4,5]
    fh = _fh(["A", "A", "A", "B", "B", "B"],
             ["A", "B", "B"], [0, 0, 2])
    # unit0: probe A local 0 -> row 0; unit1: probe B local 0 -> row 3; unit2: probe B local 2 -> row 5
    assert list(unit_electrode_rows(fh)) == [0, 3, 5]


def test_out_of_range_index_raises_not_clamped():
    # probe A has 3 rows (0..2); local index 5 is invalid -> must raise, not clamp to 2
    fh = _fh(["A", "A", "A", "B", "B", "B"], ["A"], [5])
    with pytest.raises(IndexError):
        unit_electrode_rows(fh)


def test_negative_index_raises():
    fh = _fh(["A", "A", "A"], ["A"], [-1])
    with pytest.raises(IndexError):
        unit_electrode_rows(fh)


def test_unknown_device_raises():
    fh = _fh(["A", "A", "A"], ["Z"], [0])
    with pytest.raises(KeyError):
        unit_electrode_rows(fh)


def test_noncontiguous_probe_block_raises():
    # probe A rows are interleaved with B (0,2 vs 1) -> not one contiguous block
    fh = _fh(["A", "B", "A"], ["A"], [0])
    with pytest.raises(ValueError):
        unit_electrode_rows(fh)


def test_boundary_index_ok():
    # last valid local index (blocklen-1) must be accepted
    fh = _fh(["A", "A", "A"], ["A"], [2])
    assert list(unit_electrode_rows(fh)) == [2]
