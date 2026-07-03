# Changelog

All notable changes to CALM-Data-Generator are documented here.

---

## [2.2.1] — 2026-04-29

### New Features

- **`RealGenerator.encode_to_latent()` / `decode_from_latent()`**: Round-trip a dataset through the trained model's latent space for `tvae`, `rtvae`, `scvi` and `scanvi`. Handles preprocessing (TabularEncoder, conditioning tensors, SCVI library size) internally so external drift analyses don't need to reimplement the encode/decode pipeline per method.

### Bug Fixes

- **`QualityReporter` — cross-duplicate detection**: fixed a crash/silent-zero when real and synthetic columns had mismatched dtypes by restricting the merge to shared columns and casting synthetic dtypes to match before comparing.
- **`RealBlockGenerator.generate_block()`**: `model_params` is now forwarded as `**kwargs` to the underlying `RealGenerator.generate()` call instead of being passed as an unused positional argument, so per-block model parameters actually take effect.
- **`ExternalReporter`**: the `minimal` flag is now passed through to `ydata-profiling`'s `ProfileReport` (previously accepted but silently ignored).

### Dependencies

- `scgft-evaluator` pin changed from a git URL (`git+https://github.com/nasim23ea/scgft-evaluator.git`) to a PyPI version range (`>=0.1.0,<0.2.0`).

### Developer Experience

- Ruff-driven import sorting and whitespace cleanup applied across `RealBlockGenerator.py`, `QualityReporter.py`, `ExternalReporter.py`.

---

## [2.2.0] — 2026-04-23

### New Features

- **`RealGenerator.generate()` — `cond` parameter**: Pass a conditioning array/DataFrame directly to Synthcity plugins (`tvae`, `rtvae`, `ctgan`, `dpgan`, `pategan`). Propagated to both `fit()` and `generate()` so the model trains and samples under the same conditional.
- **`RealGenerator.generate()` — `constraints` retry loop**: When constraint filtering drops rows, the generator now automatically regenerates (`needed × 2` samples, up to 5 retries) to always return `n_samples` rows. Previously the shortfall was logged but not recovered.
- **Full reproducibility across all methods**: `random_state` is now consistently propagated everywhere it was missing:
  - `_get_synthesizer()` injects `random_state` into all Synthcity plugin constructors at init time.
  - `generate()` calls for `tvae`, `rtvae`, `ctgan`, `bn`, `dpgan`, `pategan`, `ddpm`, `timegan`, `timevae`, `fflows`, `conditional_drift` and the latent-differentiation fallback path now all pass `random_state`.
  - All bare `np.random.*` calls in `scvi`, `scanvi`, `gears`, `fcs_generic` and `privatize()` replaced with `np.random.default_rng(self.random_state)`.

### Documentation

- **scGFT Evaluator** — added to `Acknowledgements & Credits` in `README.md` and `README_ES.md`.
- **scGFT Evaluator** — added usage examples and cross-references in `SingleCell / Gene Expression` and `Quality Reporting` sections of both READMEs.
- **`PRESETS_REFERENCE.md` / `PRESETS_REFERENCE_ES.md`** — added scGFT validation example after `SingleCellQualityPreset`.
- **`REAL_GENERATOR_REFERENCE.md` / `REAL_GENERATOR_REFERENCE_ES.md`** — added scGFT validation section after `to_anndata` workflow.
- **`README_ES.md`** — added `scgft-evaluator` row to Single-Cell / Omics ecosystem table.

### Dependencies

- `scgft-evaluator>=0.1.0,<0.2.0` promoted from optional (`requirements.txt` only) to mandatory dependency in `pyproject.toml`.

### Developer Experience

- Added `.pre-commit-config.yaml` with `ruff` (linter + import sorting), `trailing-whitespace`, `end-of-file-fixer`, and `check-merge-conflict` hooks.
- Added `[tool.ruff]` configuration in `pyproject.toml` (`line-length = 120`, selects `E`, `F`, `W`, `I`, ignores `E501` and `F401`).

---

## [2.1.0] — 2026-04-17

### Performance

- **Clinic.py** — vectorized group transition loop with boolean masks; eliminates O(n) `.loc` calls per patient
- **DriftInjector** — removed redundant `df.copy()` in `inject_composite_drift`; each drift method already copies internally
- **RealGenerator — FCS encoding** — replaced per-iteration `copy()` + in-place mutation with `assign()`; avoids two full DataFrame copies per (iteration × column)
- **DriftInjector — `_apply_cat_drift`** — vectorized by value group using `rng.choice(size=n)`; O(cats) calls instead of O(n)
- **RealGenerator — privatization** — replaced `apply(randomize)` closure with numpy mask + grouped `np.random.choice`; eliminates per-element Python overhead
- **QualityReporter** — skip unconditional `df.copy()` when no resampling; defer copies to inside the conditional block
- **persistence_models** — replaced `copy()` + in-place mutation with `assign()`; skip copy entirely for native-cat models (LGBM/XGB)
- **RealGenerator** — cache `select_dtypes` result; collapse `encoding_info` loop to dict comprehension

### Bug Fixes

- **FCSModel RNG** — replaced `np.random` global calls with seeded `numpy.random.default_rng(random_state)` for reproducibility
- **ScGFT integration** — fixed `ScGFT_Evaluator.run_all()` call signature (`genes_top`, `col_grupo`, `grupo_a`, `grupo_b`); removed invalid `label_col` parameter
- **RealGenerator** — cleaned duplicate and unused imports

### Dependencies

- Added `statsmodels>=0.14.0,<0.15.0` and `tqdm>=4.60.0,<5.0.0` (were missing from requirements)
- Migrated scGFT from vendored `scGFT_Evaluator.py` to installable `scgft-evaluator` package

---

## [2.0.0] — 2026-03-27

### New Features

#### ComplexGenerator — Abstract Mathematical Layer
- New `ComplexGenerator(BaseGenerator)` abstract class as an intermediate layer between `BaseGenerator` and domain-specific generators.
- Provides three reusable mathematical engines without code duplication:
  - `_generate_correlated_module(n, marginals, sigma)` — Gaussian Copula (unconditional) with PSD matrix repair via `scipy.linalg.eigh`.
  - `_generate_conditional_data(n, cond_data, cond_marginals, tgt_marginals, cov)` — Conditional Gaussian Copula with RQR for discrete marginals.
  - `apply_stochastic_effects(df, entity_ids, effect_config)` — 7 stochastic effect types + `simple_additive_shift` alias.
- `ClinicalDataGenerator` now inherits from `ComplexGenerator` instead of `BaseGenerator`.

#### Causal Dynamics (DriftInjector + ScenarioInjector)
- **`CausalEngine`** — DAG-based causal cascade propagation (`generators/dynamics/CausalEngine.py`):
  - Topological sort via Kahn's algorithm with cycle detection.
  - Differential propagation: `delta_child = f(v_parent + delta) - f(v_parent)`.
  - Transfer functions: `linear`, `exponential`, `power`, `polynomial`, or any callable.
- **`DriftInjector.inject_functional_drift()`** — drift magnitude per row = f(current value of `driver_col`). Supports additive and multiplicative modes.
- **`DriftInjector.inject_causal_cascade()`** — applies a `CausalEngine` cascade with DriftInjector's row-selection and reporting system.
- **`ScenarioInjector` evolve type `driven_by`** — feature delta per row = f(value of another column). Decoupled from time index.
- **`generators/utils/propagation.py`** — shared utility module:
  - `propagate_numeric_drift(df, rows, driver_col, delta_driver, correlations)` — extracted from both `DriftInjector` and `ScenarioInjector` to eliminate duplication.
  - `apply_func(func_name, params, x)` — evaluates named transfer functions over arrays.
- **`EvolutionFeatureConfig`** extended with `driver_col`, `func`, `func_params` fields.

### Bug Fixes

- **RealGenerator — CART/RF datetime columns**: `_synthesize_fcs_generic` now converts datetime columns to `int64` before the FCS loop, fixing `DType DateTime64DType cannot be promoted` errors.
- **RealGenerator — `bn` method dispatch**: `elif method == "bayesian_network"` extended to `elif method in ("bayesian_network", "bn")`, fixing synthesis returning `None` when `method="bn"` was used.
- **RealGenerator — `conditional_drift` Synthcity API**: removed invalid `cond=` parameter from `syn.generate()` — TVAE/CTGAN are unconditional generators and do not support inference-time conditioning.
- **RealGenerator — `windowed_copula` 1D array**: `copula.random(n)` can return a 1D array when `n=1`; now reshaped to 2D before `scaler.inverse_transform()`.
- **`ClinicalDataGenerator` — two remaining `_generate_module_data` calls**: updated to `_generate_correlated_module` after the ComplexGenerator refactor.
- **`test_disease_effects_fix.py`**: converted from a module-level script to a proper pytest function.

### Testing

- All `unittest.TestCase` test files converted to pure pytest (9 files, 41 tests).
- New `tests/test_causal_engine.py` — 10 tests covering DAG propagation, cycle detection, partial rows, topological order.
- New `tests/test_functional_drift.py` — 8 tests covering functional drift, causal cascade, `driven_by`, and `propagate_numeric_drift`.
- Full test suite: **186 passed, 8 skipped, 0 failed**.

### Documentation

- New `CAUSAL_ENGINE_REFERENCE.md` / `_ES.md` — complete DAG reference with IoT, Finance, and Clinical examples.
- New `COMPLEX_GENERATOR_REFERENCE.md` / `_ES.md` — reference for the three mathematical engines.
- New `Library Reference` section in `DOCUMENTATION.md` / `_ES.md` — maps every synthesis method to its underlying library with links to official docs.
- Updated `DRIFT_INJECTOR_REFERENCE.md` / `_ES.md` — added `inject_functional_drift` and `inject_causal_cascade`.
- Updated `SCENARIO_INJECTOR_REFERENCE.md` / `_ES.md` — added `driven_by` evolution type.
- Updated `API.md` / `API_ES.md` — added `generators.dynamics` (CausalEngine) and `generators.utils` sections.
- Updated `CLINICAL_GENERATOR_REFERENCE.md` / `_ES.md` — inheritance from ComplexGenerator, `additive_shift` warning for proteins.
- Updated `README.md` / `README_ES.md` — expanded "Core Technologies" with full library tables and links; added Scenario Evolution section.
- Tutorials updated: `advanced_drifts.py` and `scenario_injector.py` include examples for `inject_functional_drift`, `inject_causal_cascade`, and `driven_by`.

---

## [1.2.0] — Previous Release

- `differentiation_factor` parameter for TVAE and scVI (increases class separability in latent space).
- `clipping_mode` parameter: `'strict'`, `'permissive'`, or `'none'`.
- `use_latent_sampling` for scVI.
- `_apply_postprocess_distribution` for intelligent class-distribution-aware resampling.
- Windowed Copula synthesis method.
- Conditional Drift synthesis method.
- Differential Privacy methods: DPGAN, PATEGAN.
