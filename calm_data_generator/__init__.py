# CALM-Data-Generator - Synthetic Data Generation Library

__version__ = "2.3.0"

__all__ = [
    "RealGenerator",
    "QualityReporter",
    "ClinicalDataGenerator",
    "ComplexGenerator",
    "StreamGenerator",
    "DriftInjector",
    "ScenarioInjector",
    "CausalEngine",
    "presets",
]

_lazy_map = {
    "RealGenerator": ("calm_data_generator.generators.tabular", "RealGenerator"),
    "QualityReporter": ("calm_data_generator.reports.QualityReporter", "QualityReporter"),
    "ClinicalDataGenerator": ("calm_data_generator.generators.clinical", "ClinicalDataGenerator"),
    "ComplexGenerator": ("calm_data_generator.generators.complex", "ComplexGenerator"),
    "DriftInjector": ("calm_data_generator.generators.drift", "DriftInjector"),
    "ScenarioInjector": ("calm_data_generator.generators.dynamics", "ScenarioInjector"),
    "CausalEngine": ("calm_data_generator.generators.dynamics", "CausalEngine"),
}


def __getattr__(name: str):
    if name == "StreamGenerator":
        try:
            from calm_data_generator.generators.stream import StreamGenerator
            return StreamGenerator
        except ImportError:
            return None

    if name == "presets":
        # `presets` is a real subpackage; import it directly.
        # (A self-referential lazy_map entry here would recurse infinitely.)
        import importlib
        return importlib.import_module("calm_data_generator.presets")

    if name in _lazy_map:
        module_path, attr = _lazy_map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module 'calm_data_generator' has no attribute {name!r}")
