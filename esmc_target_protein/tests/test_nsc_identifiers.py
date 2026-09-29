"""Unit tests for NSC identifier canonicalization (Tests R, S)."""

import pytest
from esmc_target.identifiers import canonicalize_nsc_id


def test_nsc_canonicalization_valid():
    assert canonicalize_nsc_id(740) == "NSC-740"
    assert canonicalize_nsc_id("740") == "NSC-740"
    assert canonicalize_nsc_id("NSC-740") == "NSC-740"
    assert canonicalize_nsc_id("NSC 740") == "NSC-740"
    assert canonicalize_nsc_id("nsc740") == "NSC-740"
    assert canonicalize_nsc_id(" NSC-740 ") == "NSC-740"
    assert canonicalize_nsc_id("nsc-0740") == "NSC-740"


def test_nsc_canonicalization_invalid():
    with pytest.raises(ValueError, match="nsc_id is missing or None"):
        canonicalize_nsc_id(None)

    with pytest.raises(ValueError, match="nsc_id is NaN"):
        canonicalize_nsc_id(float("nan"))

    with pytest.raises(ValueError, match="nsc_id is empty"):
        canonicalize_nsc_id("  ")

    with pytest.raises(ValueError, match="unparseable nsc_id"):
        canonicalize_nsc_id("DRUG_ABC")

    with pytest.raises(ValueError, match="unparseable nsc_id"):
        canonicalize_nsc_id("NSC-XYZ")
