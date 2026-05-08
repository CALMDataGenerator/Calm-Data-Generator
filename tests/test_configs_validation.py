"""Tests for Pydantic config validators."""

import pytest
from pydantic import ValidationError

from calm_data_generator.generators.configs import (
    DriftConfig,
    EvolutionFeatureConfig,
    ReportConfig,
)

# ── DriftConfig ────────────────────────────────────────────────────────────────

def test_drift_config_valid_defaults():
    cfg = DriftConfig()
    assert cfg.magnitude == 0.2
    assert cfg.profile == "sigmoid"


def test_drift_config_magnitude_zero_raises():
    with pytest.raises(ValidationError, match="magnitude"):
        DriftConfig(magnitude=0.0)


def test_drift_config_magnitude_negative_raises():
    with pytest.raises(ValidationError, match="magnitude"):
        DriftConfig(magnitude=-1.0)


def test_drift_config_profile_invalid_raises():
    with pytest.raises(ValidationError, match="profile"):
        DriftConfig(profile="exponential")


def test_drift_config_profile_valid():
    for p in ("linear", "sigmoid", "cosine"):
        cfg = DriftConfig(profile=p)
        assert cfg.profile == p


def test_drift_config_start_before_end_valid():
    cfg = DriftConfig(start_index=0, end_index=100)
    assert cfg.start_index == 0


def test_drift_config_start_after_end_raises():
    with pytest.raises(ValidationError, match="start_index"):
        DriftConfig(start_index=100, end_index=50)


def test_drift_config_start_equals_end_raises():
    with pytest.raises(ValidationError, match="start_index"):
        DriftConfig(start_index=50, end_index=50)


# ── EvolutionFeatureConfig ─────────────────────────────────────────────────────

def test_evolution_valid_type():
    for t in ("linear", "cycle", "sigmoid", "step", "noise", "random_walk"):
        cfg = EvolutionFeatureConfig(type=t)
        assert cfg.type == t


def test_evolution_invalid_type_raises():
    with pytest.raises(ValidationError, match="type"):
        EvolutionFeatureConfig(type="unknown_type")


def test_evolution_period_zero_raises():
    with pytest.raises(ValidationError, match="period"):
        EvolutionFeatureConfig(type="cycle", period=0.0)


def test_evolution_period_negative_raises():
    with pytest.raises(ValidationError, match="period"):
        EvolutionFeatureConfig(type="cycle", period=-10.0)


def test_evolution_period_none_allowed():
    cfg = EvolutionFeatureConfig(type="linear", period=None)
    assert cfg.period is None


# ── ReportConfig ───────────────────────────────────────────────────────────────

def test_report_config_valid_defaults():
    cfg = ReportConfig()
    assert cfg.output_dir == "output"
    assert cfg.auto_report is True


def test_report_config_resample_rule_none():
    cfg = ReportConfig(resample_rule=None)
    assert cfg.resample_rule is None


def test_report_config_resample_rule_valid_string():
    cfg = ReportConfig(resample_rule="1D")
    assert cfg.resample_rule == "1D"


def test_report_config_resample_rule_invalid_string_raises():
    with pytest.raises(ValidationError, match="resample_rule"):
        ReportConfig(resample_rule="INVALID_FREQ_XYZ")


def test_report_config_resample_rule_int_allowed():
    cfg = ReportConfig(resample_rule=5)
    assert cfg.resample_rule == 5
