import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.tabular.RealGenerator import RealGenerator
from calm_data_generator.reports.QualityReporter import QualityReporter
from calm_data_generator.reports.Visualizer import Visualizer


def _make_real_df():
    data = {
        "age": np.random.randint(20, 60, 100),
        "salary": np.random.normal(50000, 15000, 100),
        "department": np.random.choice(["Sales", "HR", "Tech"], 100),
        "target": np.random.choice([0, 1], 100),
    }
    return pd.DataFrame(data)


def test_imports_no_sdv():
    """Verify the library does not import SDV as a dependency."""
    import sys

    # SDV should not be imported as a side-effect of loading calm_data_generator
    import calm_data_generator  # noqa: F401
    assert "sdv" not in sys.modules, "SDV was imported as a side-effect of calm_data_generator"


def test_synthcity_available():
    """Verify Synthcity is importable"""
    try:
        import synthcity
    except ImportError:
        pytest.fail("Synthcity should be installed")


def test_sdmetrics_available():
    """Verify SDMetrics is importable"""
    try:
        import sdmetrics
    except ImportError:
        pytest.fail("SDMetrics should be installed")


def test_quality_reporter_renaming(tmp_path):
    real_df = _make_real_df()
    reporter = QualityReporter(verbose=False)
    synth_df = real_df.copy()
    try:
        reporter.generate_report(
            real_df=real_df,
            synthetic_df=synth_df,
            generator_name="TestGen",
            output_dir=str(tmp_path),
            minimal=True,
        )
    except Exception as e:
        pytest.fail(f"QualityReporter failed with minimal=True: {e}")


def test_real_generator_plugins():
    """Verify RealGenerator can init synthcity plugins (mocked or real)"""
    # checks if plugins are loading without smartnoise/sdv errors
    gen = RealGenerator()
    # Just init shouldn't crash
    assert gen is not None


def test_presets_import_does_not_recurse():
    """Regression test: `from calm_data_generator import presets` used to raise
    RecursionError because the lazy-import map had a self-referential entry for the
    `presets` subpackage (it tried to import itself to resolve itself)."""
    from calm_data_generator import presets

    assert hasattr(presets, "FastPreset")


def test_real_generator_generate_has_runtime_docstring():
    """Regression test: `generate()`'s docstring used to be written *after* executable
    code, making it a dead string literal rather than an actual docstring (__doc__ was
    None). Same bug existed in `generate_custom` and all preset `.generate()` methods."""
    assert RealGenerator.generate.__doc__
    assert RealGenerator.generate_custom.__doc__


def test_all_presets_have_generate_docstring():
    import inspect

    from calm_data_generator import presets

    preset_classes = [
        obj for name, obj in inspect.getmembers(presets, inspect.isclass)
        if name.endswith("Preset") and obj.__module__.startswith("calm_data_generator.presets")
    ]
    assert len(preset_classes) >= 18  # sanity check we actually found the presets

    missing = [cls.__name__ for cls in preset_classes if not cls.generate.__doc__]
    assert missing == [], f"Presets with no generate() docstring: {missing}"
