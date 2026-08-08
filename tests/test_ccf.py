"""Unit tests for CCF acronym decoding (openscope_ccf.ccf).

Pure functions, no network — safe to run in CI.
"""
import pytest

from openscope_ccf.ccf import decode_ccf, decode_many


def test_isocortex_layer_split():
    d = decode_ccf("VISp5")
    assert d == dict(area="VISp", layer="5", tissue="grey", group="visual_ctx")
    d = decode_ccf("VISl2/3")
    assert d["area"] == "VISl" and d["layer"] == "2/3" and d["group"] == "visual_ctx"


def test_motor_and_pfc_groups():
    assert decode_ccf("MOp5")["group"] == "motor_ctx"
    assert decode_ccf("ACAd6a")["group"] == "cingulate/PFC"
    assert decode_ccf("SSp-bfd4")["group"] == "somatosensory_ctx"


def test_hippocampus_subfields_are_not_layers():
    for a in ("CA1", "CA2", "CA3"):
        d = decode_ccf(a)
        assert d["area"] == a and d["layer"] is None and d["group"] == "hippocampus"


def test_dentate_gyrus_splits_area_and_layer():
    d = decode_ccf("DG-mo")
    assert d["area"] == "DG" and d["layer"] == "mo" and d["group"] == "hippocampus"


def test_fiber_tract_flagged():
    d = decode_ccf("fi")
    assert d["tissue"] == "fiber_tract" and d["group"] == "white_matter"


def test_unknown_is_unassigned_not_grey():
    # regression: "unknown" must NOT be classified as grey matter
    d = decode_ccf("unknown")
    assert d["tissue"] == "unassigned" and d["group"] == "unassigned"


@pytest.mark.parametrize("a", ["root", "void", ""])
def test_explicit_unassigned_tokens(a):
    assert decode_ccf(a)["tissue"] == "unassigned"


def test_nan_and_none_decode_to_unassigned_not_typeerror():
    # regression: a missing acronym must not raise an opaque TypeError in the regex
    for missing in (None, float("nan")):
        d = decode_ccf(missing)
        assert d["tissue"] == "unassigned" and d["area"] is None


def test_non_string_nonnull_raises_typeerror():
    with pytest.raises(TypeError):
        decode_ccf(123)


def test_thalamus_and_striatum():
    assert decode_ccf("LGd")["group"] == "thalamus"
    assert decode_ccf("CP")["group"] == "striatum"


def test_decode_many_matches_scalar():
    xs = ["VISp5", "CA1", "fi", "unknown"]
    assert decode_many(xs) == [decode_ccf(x) for x in xs]
