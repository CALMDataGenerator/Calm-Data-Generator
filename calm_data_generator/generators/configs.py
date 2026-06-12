from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DateConfig(BaseModel):
    """
    Configuration for timestamp injection in generated data.
    """

    date_col: str = "timestamp"
    start_date: Optional[str] = None
    frequency: int = 1
    step: Optional[Dict[str, int]] = None  # e.g. {"days": 1}


class DriftConfig(BaseModel):
    """
    Configuration for drift injection.
    """

    method: str = "inject_feature_drift"
    drift_type: str = "gaussian_noise"
    feature_cols: Optional[List[str]] = None
    magnitude: float = 0.2

    # Selection parameters
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    block_index: Optional[int] = None
    block_column: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None

    # Advanced / Window parameters
    center: Optional[Union[int, float]] = None
    width: Optional[Union[int, float]] = None
    profile: str = "sigmoid"
    speed_k: float = 1.0
    direction: str = "up"
    inconsistency: float = 0.0

    # For specialized drifts
    drift_value: Optional[float] = None
    drift_values: Optional[Dict[str, float]] = None

    # Extra params for specific methods (e.g., custom params for custom drift)
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("magnitude")
    @classmethod
    def magnitude_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("magnitude must be > 0")
        return v

    @field_validator("profile")
    @classmethod
    def profile_valid(cls, v: str) -> str:
        valid = {"linear", "sigmoid", "cosine"}
        if v not in valid:
            raise ValueError(f"profile must be one of {valid}")
        return v

    @model_validator(mode="after")
    def start_before_end(self) -> "DriftConfig":
        if self.start_index is not None and self.end_index is not None:
            if self.start_index >= self.end_index:
                raise ValueError("start_index must be < end_index")
        return self

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


class EvolutionFeatureConfig(BaseModel):
    """
    Configuration for evolving a single feature.
    """

    type: str  # 'linear', 'cycle', 'sigmoid'
    slope: Optional[float] = 0.0
    intercept: Optional[float] = 0.0
    amplitude: Optional[float] = 1.0
    period: Optional[float] = 100.0
    phase: Optional[float] = 0.0
    center: Optional[float] = None
    width: Optional[float] = None

    # Extra parameters for specific evolution types
    rate: Optional[float] = None      # for exponential_growth, decay
    noise_std: Optional[float] = None # for noise
    step_std: Optional[float] = None  # for random_walk
    step: Optional[float] = None      # for step
    value: Optional[float] = None     # for step

    # Fields for evolve_type "driven_by" (Pilar 5)
    driver_col: Optional[str] = None      # column whose current value drives the delta
    func: Optional[str] = None            # "linear"|"exponential"|"power"|"polynomial"
    func_params: Optional[dict] = None    # parameters for func

    @field_validator("type")
    @classmethod
    def type_valid(cls, v: str) -> str:
        valid = {"linear", "trend", "cycle", "sinusoidal", "sigmoid", "exponential_growth", "exponential_decay", "noise", "random_walk", "step", "driven_by"}
        if v not in valid:
            raise ValueError(f"type must be one of {valid}")
        return v

    @field_validator("period")
    @classmethod
    def period_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("period must be > 0")
        return v

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


class ScenarioConfig(BaseModel):
    """
    Configuration for scenario injection (evolution and target construction).
    """

    state_config: Optional[Dict] = None
    evolve_features: Dict[str, Union[Dict, EvolutionFeatureConfig]] = Field(
        default_factory=dict
    )
    construct_target: Optional[Dict] = None


class ReportConfig(BaseModel):
    """
    Configuration for report generation.
    """

    output_dir: str = "output"
    auto_report: bool = True
    minimal: bool = False
    target_column: Optional[str] = None
    time_col: Optional[str] = None
    block_column: Optional[str] = None
    resample_rule: Optional[Union[str, int]] = None
    privacy_check: bool = False
    discriminator: bool = False
    focus_columns: Optional[List[str]] = None
    constraints_stats: Optional[Dict[str, int]] = None
    sequence_config: Optional[Dict] = None
    per_block_external_reports: bool = False
    use_scgft: bool = False

    @field_validator("resample_rule")
    @classmethod
    def resample_rule_valid(cls, v: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
        if v is not None and isinstance(v, str):
            import pandas as pd
            try:
                pd.tseries.frequencies.to_offset(v)
            except ValueError:
                raise ValueError(f"resample_rule '{v}' is not a valid pandas frequency string")
        return v
