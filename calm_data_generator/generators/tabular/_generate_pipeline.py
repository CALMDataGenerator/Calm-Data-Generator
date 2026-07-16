"""
Mixin providing the internal pipeline stages for RealGenerator.generate().

`generate()` itself stays a slim orchestrator in RealGenerator.py; each stage
below is an "extract method" split of what used to be one 740-line function.
Order of calls matches the original inline order exactly — this is a
behavior-preserving refactor, not a redesign.

Methods: _validate_generate_call, _resolve_generate_config,
         _resolve_generate_distributions, _dispatch_synthesis,
         _apply_generate_postprocess, _apply_generate_constraints,
         _finalize_generate_output.
"""

import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from calm_data_generator.generators.configs import DateConfig, DriftConfig, ReportConfig


class _GeneratePipelineMixin:

    def _validate_generate_call(
        self, method: str, n_samples: int, epsilon: Optional[float], kwargs: dict
    ) -> None:
        """Validates epsilon/DP method pairing, method name, and warns on unsupported kwargs."""
        _dp_methods = {"dpgan", "pategan"}
        if epsilon is not None:
            if epsilon <= 0:
                raise ValueError(
                    f"epsilon must be positive (got {epsilon}). "
                    f"Typical values: 0.1 (strong privacy) to 10.0 (weak privacy)."
                )
            if method not in _dp_methods:
                self.logger.warning(
                    "epsilon=%.2f has no effect with method='%s'. "
                    "Use method='dpgan' or method='pategan' for differential privacy.",
                    epsilon, method,
                )

        self._validate_method(method)

        # Validate differentiation and clipping parameters for unsupported methods
        if kwargs.get("differentiation_factor", 0.0) > 0 and method not in ["tvae", "rtvae", "scvi"]:
            self.logger.warning(
                f"differentiation_factor is only supported for 'tvae', 'rtvae' and 'scvi'. "
                f"It will be ignored for method '{method}'."
            )
        if kwargs.get("clipping_mode", "strict") != "none" and method not in ["tvae", "scvi", "ctgan"]:
            # Note: ctgan handles its own clipping or ignores it gracefully, but we primarily
            # target the new logic for tvae/scvi.
            pass

        # Note: params was used for default values, now all methods use **kwargs from model_params
        self.logger.info(
            f"Starting generation of {n_samples} samples using method '{method}'..."
        )

    def _resolve_generate_config(
        self,
        output_dir: Optional[str],
        report_config: Optional[Union[ReportConfig, Dict]],
        date_config: Optional["DateConfig"],
        date_start: Optional[str],
        date_every: int,
        date_step: Optional[Dict[str, int]],
        date_col: str,
    ):
        """Resolves ReportConfig, effective output_dir, and DateConfig (legacy args)."""
        # Resolve ReportConfig (defaults to None if not provided)
        if report_config:
            if isinstance(report_config, dict):
                report_config = ReportConfig(**report_config)
        # We don't necessarily force creation here, reporter handles None.
        # But we might want to consolidate output_dir logic.

        # Determine effective output_dir
        # Logic: 1. report_config.output_dir (if provided/default 'output')?
        #        2. output_dir arg
        #        3. self.output_dir (if exists)
        #        4. '.'

        effective_output_dir = (
            output_dir
            or (report_config.output_dir if report_config else None)
            or getattr(self, "output_dir", None)
            or "."
        )
        # Update report_config if exists
        if report_config:
            report_config.output_dir = effective_output_dir

        # Resolve Date Config
        if date_config is None and date_start is not None:
            # Construct from legacy args
            from calm_data_generator.generators.configs import DateConfig

            self.logger.warning(
                "generate(date_start=..., date_every=..., date_step=...) is a legacy "
                "shorthand. Prefer generate(date_config=DateConfig(start_date=..., "
                "frequency=..., step=..., date_col=...)) instead."
            )
            date_config = DateConfig(
                start_date=date_start,
                frequency=date_every,
                step=date_step,
                date_col=date_col,
            )

        return report_config, effective_output_dir, date_config

    def _resolve_generate_distributions(
        self,
        data,
        target_col: Optional[str],
        balance: bool,
        custom_distribution: Optional[Dict],
        custom_distributions: Optional[Dict],
    ):
        """Merges the legacy `custom_distribution` alias, validates, and applies `balance`."""
        # Merge shorthand aliases into canonical params
        if custom_distribution:
            self.logger.warning(
                "generate(custom_distribution=...) (singular) is a legacy alias. "
                "Prefer generate(custom_distributions=...) (plural) instead."
            )
            custom_distributions = custom_distribution if custom_distributions is None else {**custom_distribution, **custom_distributions}

        if custom_distributions:
            custom_distributions = self._validate_custom_distributions(
                custom_distributions, data
            )
        if (
            balance
            and target_col
            and (custom_distributions is None or target_col not in custom_distributions)
        ):
            self.logger.info(
                f"'balance' is True. Generating balanced distribution for '{target_col}'."
            )
            target_classes = data[target_col].unique()
            custom_distributions = custom_distributions or {}
            custom_distributions[target_col] = {
                c: 1 / len(target_classes) for c in target_classes
            }
        return custom_distributions

    def _dispatch_synthesis(
        self,
        method: str,
        data,
        n_samples: int,
        target_col: Optional[str],
        custom_distributions: Optional[Dict],
        cond: Optional[Any],
        epsilon: Optional[float],
        delta: float,
        original_adata,
        kwargs: dict,
    ):
        """Dispatches to the `_synthesize_*` method matching `method`, wrapped with a
        friendly RuntimeError (suggesting alternatives) on training failure."""
        synth = None

        _METHOD_ALTERNATIVES = {
            "ctgan": ["tvae", "cart"],
            "tvae": ["cart", "ctgan"],
            "rtvae": ["tvae", "cart"],
            "cart": ["rf", "lgbm", "resample"],
            "rf": ["cart", "lgbm"],
            "lgbm": ["cart", "rf"],
            "gmm": ["copula", "cart"],
            "copula": ["gmm", "cart"],
            "scvi": ["tvae", "cart"],
            "scanvi": ["scvi", "tvae"],
            "ddpm": ["tvae", "ctgan"],
            "diffusion": ["tvae", "ctgan"],
            "timegan": ["timevae", "fflows"],
            "timevae": ["timegan", "fflows"],
            "fflows": ["timegan", "timevae"],
        }

        try:
            if method == "ctgan":
                synth = self._synthesize_ctgan(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    cond=cond,
                    **kwargs,
                )
            elif method == "great":
                synth = self._synthesize_great(
                    data,
                    n_samples,
                    target_col=target_col,
                    **kwargs,
                )
            elif method == "dpgan":
                _dp_kwargs = {**(kwargs or {})}
                if epsilon is not None:
                    _dp_kwargs.setdefault("epsilon", epsilon)
                    _dp_kwargs.setdefault("delta", delta)
                synth = self._synthesize_dpgan(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    cond=cond,
                    **_dp_kwargs,
                )
            elif method == "pategan":
                _dp_kwargs = {**(kwargs or {})}
                if epsilon is not None:
                    _dp_kwargs.setdefault("epsilon", epsilon)
                    _dp_kwargs.setdefault("delta", delta)
                synth = self._synthesize_pategan(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    cond=cond,
                    **_dp_kwargs,
                )
            elif method == "tvae":
                synth = self._synthesize_tvae(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    cond=cond,
                    **kwargs,
                )
            elif method == "rtvae":
                synth = self._synthesize_rtvae(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    cond=cond,
                    **kwargs,
                )
            elif method == "conditional_drift":
                synth = self._synthesize_conditional_drift(
                    data = data,
                    n_samples = n_samples,
                    time_col = kwargs.get("time_col") if kwargs else None,
                    n_stages = kwargs.get("n_stages", 5) if kwargs else 5,
                    general_stages = kwargs.get("general_stages") if kwargs else None,
                    base_method = kwargs.get("base_method", "tvae") if kwargs else "tvae",
                    **{k: v for k, v in kwargs.items() if k not in {"time_col", "n_stages", "base_method", "general_stages"}}
                )
            elif method == "windowed_copula":
                synth = self._synthesize_windowed_copula(
                    data=data,
                    n_samples=n_samples,
                    time_col=kwargs.get("time_col") if kwargs else None,
                    n_windows=kwargs.get("n_windows", 5) if kwargs else 5,
                    generate_at=kwargs.get("generate_at") if kwargs else None,
                     **{k: v for k, v in kwargs.items() if k not in {"time_col", "n_windows", "generate_at"}}
                )

            elif method == "resample":
                synth = self._synthesize_resample(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                )
            elif method == "kde":
                synth = self._synthesize_kde(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "cart":
                synth = self._synthesize_cart(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "hmm":
                synth = self._synthesize_hmm(
                    data,
                    n_samples,
                    n_components=kwargs.get("n_components", 4),
                    covariance_type=kwargs.get("covariance_type", "full"),
                    n_iter=kwargs.get("n_iter", 100),
                    **{k: v for k, v in kwargs.items() if k not in {"n_components", "covariance_type", "n_iter"}}
                )

            elif method == "xgboost":
                synth = self._synthesize_xgboost(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {})
                )
            elif method == "copula":
                synth = self._synthesize_copula(
                    data,
                    n_samples,
                    "copula",
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "rf":
                synth = self._synthesize_rf(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "lgbm":
                synth = self._synthesize_lgbm(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "gmm":
                synth = self._synthesize_gmm(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )

            elif method == "smote":
                synth = self._synthesize_smote(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "adasyn":
                synth = self._synthesize_adasyn(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )

            elif method in ["diffusion", "ddpm"]:
                synth = self._synthesize_ddpm(data, n_samples, **(kwargs or {}))
            elif method == "timegan":
                synth = self._synthesize_timegan(data, n_samples, **(kwargs or {}))
            elif method == "timevae":
                synth = self._synthesize_timevae(data, n_samples, **(kwargs or {}))
            elif method == "fflows":
                synth = self._synthesize_fflows(data, n_samples, **(kwargs or {}))
            elif method in ("bayesian_network", "bn"):
                synth = self._synthesize_bn(
                    data, n_samples, target_col=target_col, **(kwargs or {})
                )
            elif method == "scvi":
                # Pass original_adata if available to avoid redundant conversion
                synth = self._synthesize_scvi(
                    original_adata if original_adata is not None else data,
                    n_samples,
                    target_col=target_col,
                    **(kwargs or {}),
                )
            elif method == "scanvi":
                # Pass original_adata if available to avoid redundant conversion
                synth = self._synthesize_scanvi(
                    original_adata if original_adata is not None else data,
                    n_samples,
                    target_col=target_col,
                    **(kwargs or {}),
                )
            elif method == "gears":
                # Pass original_adata if available to avoid redundant conversion
                synth = self._synthesize_gears(
                    original_adata if original_adata is not None else data,
                    n_samples,
                    target_col=target_col,
                    **(kwargs or {}),
                )

        except Exception as _train_exc:
            self.logger.debug("Training traceback:", exc_info=True)
            _alternatives = _METHOD_ALTERNATIVES.get(method, ["cart", "resample"])
            _alt_str = "', '".join(_alternatives)
            raise RuntimeError(
                f"Model training failed (method='{method}', n_samples={n_samples}, "
                f"n_rows={len(data) if hasattr(data, '__len__') else '?'}). "
                f"Reason: {_train_exc}. "
                f"Try: method='{_alt_str}'."
            ) from _train_exc

        return synth

    def _apply_generate_postprocess(
        self,
        synth,
        method: str,
        custom_distributions: Optional[Dict],
        target_col: Optional[str],
        n_samples: int,
    ):
        """Applies post-process distribution resampling for methods that don't support it natively."""
        # Methods that handle custom_distributions natively during generation:
        #   cart, rf, lgbm (FCS resampling), resample (weighted sampling),
        #   ctgan (conditional per-class generation), smote/adasyn (sampling_strategy),
        #   gmm (post-process inside _synthesize_gmm).
        # Everything else needs post-process resampling on the output.
        _POSTPROCESS_METHODS = {"copula", "bn", "bayesian_network", "ddpm", "diffusion", "scvi", "great"}
        _TIMESERIES_METHODS = {"timegan", "timevae", "fflows"}

        if synth is not None and custom_distributions and method in _TIMESERIES_METHODS:
            self.logger.warning(
                f"'balance' / 'custom_distribution' is not supported for "
                f"time series method '{method}'. The parameters will be ignored. "
                f"Time series methods operate on sequences, not individual class rows."
            )
        elif synth is not None and custom_distributions and method in _POSTPROCESS_METHODS:
            self.logger.info(
                f"Method '{method}' does not support custom distributions natively. "
                f"Applying post-process resampling."
            )
            synth = self._apply_postprocess_distribution(
                synth, custom_distributions, target_col, n_samples
            )
        return synth

    def _apply_generate_constraints(self, synth, constraints, n_samples: int, cond):
        """Filters `synth` by `constraints`, retrying generation to backfill dropped rows."""
        if synth is None or not constraints:
            return synth

        self.logger.info(
            f"Applying {len(constraints)} constraints to generated data..."
        )

        def _apply_constraints_mask(df: pd.DataFrame) -> pd.Series:
            mask = pd.Series(True, index=df.index)
            for const in constraints:
                col = const.get("col")
                op = const.get("op")
                val = const.get("val")
                if col not in df.columns:
                    self.logger.warning(f"Constraint column '{col}' not found. Skipping.")
                    continue
                if op == ">":
                    mask &= df[col] > val
                elif op == "<":
                    mask &= df[col] < val
                elif op == ">=":
                    mask &= df[col] >= val
                elif op == "<=":
                    mask &= df[col] <= val
                elif op == "==":
                    mask &= df[col] == val
                elif op == "!=":
                    mask &= df[col] != val
            return mask

        initial_count = len(synth)
        synth = synth[_apply_constraints_mask(synth)].reset_index(drop=True)
        dropped = initial_count - len(synth)

        if dropped > 0:
            self.logger.warning(
                f"Constraints filtering dropped {dropped} rows ({dropped / initial_count:.1%})."
            )

        # Retry loop: regenerate until n_samples satisfied or max retries reached
        _MAX_RETRIES = 5
        _retry = 0
        while len(synth) < n_samples and _retry < _MAX_RETRIES and self.synthesizer is not None:
            _retry += 1
            needed = n_samples - len(synth)
            oversample = max(needed * 2, 64)
            self.logger.info(
                f"Constraints retry {_retry}/{_MAX_RETRIES}: generating {oversample} extra rows to fill {needed} missing."
            )
            try:
                gen_kwargs = {"count": oversample}
                if cond is not None:
                    gen_kwargs["cond"] = cond
                extra = self.synthesizer.generate(**gen_kwargs).dataframe()
                extra = extra[_apply_constraints_mask(extra)].reset_index(drop=True)
                synth = pd.concat([synth, extra], ignore_index=True)
            except Exception as e:
                self.logger.warning(f"Constraints retry {_retry} failed: {e}")
                break

        if len(synth) < n_samples:
            self.logger.warning(
                f"Could only generate {len(synth)}/{n_samples} rows satisfying constraints after {_retry} retries. "
                "Consider loosening constraints or improving the model."
            )
        else:
            synth = synth.iloc[:n_samples].reset_index(drop=True)

        return synth

    def _finalize_generate_output(
        self,
        synth,
        data,
        method: str,
        target_col: Optional[str],
        block_column: Optional[str],
        output_dir: Optional[str],
        effective_output_dir: str,
        date_config,
        drift_injection_config: Optional[List[Union[Dict, DriftConfig]]],
        dynamics_config: Optional[Dict],
        save_dataset: bool,
        adversarial_validation: bool,
        report_config,
    ):
        """Applies dynamics injection, date injection, drift injection, reporting and
        dataset saving to a successfully generated `synth`. Returns the final `synth`."""
        self.logger.info(f"Successfully synthesized {len(synth)} samples.")

        # --- Dynamics Injection (Feature Evolution & Target Construction) ---
        if synth is not None and dynamics_config:
            self.logger.debug("Applying dynamics config...")
            self.logger.info("Applying dynamics injection config...")
            from calm_data_generator.generators.dynamics.ScenarioInjector import (
                ScenarioInjector,
            )

            injector = ScenarioInjector(seed=self.random_state)
            if "evolve_features" in dynamics_config:
                synth = injector.evolve_features(
                    synth, evolution_config=dynamics_config["evolve_features"]
                )
            if "construct_target" in dynamics_config:
                synth = injector.construct_target(
                    synth, **dynamics_config["construct_target"]
                )

        # --- Date Injection (if not done in dynamics) ---
        if date_config and date_config.start_date:
            synth = self._inject_dates(
                df=synth,
                date_col=date_config.date_col,
                date_start=date_config.start_date,
                date_every=date_config.frequency,
                date_step=date_config.step,
            )

        # --- Drift Injection ---
        if synth is not None and drift_injection_config:
            self.logger.debug("Applying drift injection...")
            self.logger.info("Applying drift injection config...")
            drift_out_dir = (
                output_dir or "."
            )  # Drift injector might need a dir, fallback to current
            time_col_name = date_config.date_col if date_config else "timestamp"
            from calm_data_generator.generators.drift.DriftInjector import DriftInjector

            drift_injector = DriftInjector(
                original_df=synth,  # We drift the synthetic data
                output_dir=drift_out_dir,
                generator_name=f"{method}_drifted",
                target_column=target_col,
                block_column=block_column,
                time_col=time_col_name,
                random_state=self.random_state,
            )

            for drift_conf in drift_injection_config:
                # Determine method and params
                method_name = "inject_feature_drift"  # Default
                params_drift = {}
                drift_obj = None

                if isinstance(drift_conf, DriftConfig):
                    method_name = drift_conf.method
                    drift_obj = drift_conf
                    params_drift = drift_conf.params or {}
                elif isinstance(drift_conf, dict):
                    # Support nested {"method": ..., "params": ...} or flat
                    if "method" in drift_conf and "params" in drift_conf:
                        method_name = drift_conf.get("method")
                        params_drift = drift_conf.get("params", {})
                    else:
                        # Flat dict
                        method_name = drift_conf.get(
                            "drift_method",
                            drift_conf.get("method", "inject_feature_drift"),
                        )
                        params_drift = drift_conf

                if hasattr(drift_injector, method_name):
                    self.logger.info(f"Injecting drift: {method_name}")
                    drift_method = getattr(drift_injector, method_name)
                    try:
                        # Add 'df' if not present
                        if "df" not in params_drift:
                            params_drift["df"] = synth

                        # Call method
                        if drift_obj:
                            # Pass config object explicitly
                            res = drift_method(
                                drift_config=drift_obj, **params_drift
                            )
                        else:
                            # Pass params (will be converted to config internally if needed)
                            res = drift_method(**params_drift)

                        # Update synth if result is dataframe
                        if isinstance(res, pd.DataFrame):
                            synth = res
                    except Exception as e:
                        self.logger.error(
                            f"Failed to apply drift {method_name}: {e}"
                        )
                        raise e
                else:
                    self.logger.warning(
                        f"Drift method '{method_name}' not found in DriftInjector."
                    )

        if self.auto_report and output_dir:
            self.logger.debug("Generating report...")
            time_col_name = date_config.date_col if date_config else "timestamp"

            # Build drift_config for report if drift was applied
            report_drift_config = None
            if drift_injection_config:
                # Summarize drift configuration for the report
                drift_methods = []
                for d in drift_injection_config:
                    if isinstance(d, DriftConfig):
                        drift_methods.append(d.method)
                    else:
                        drift_methods.append(
                            d.get("method", d.get("drift_method", "unknown"))
                        )

                report_drift_config = {
                    "drift_type": ", ".join(drift_methods),
                    "drift_magnitude": "See config",
                    "affected_columns": "Multiple (via drift_injection_config)",
                }

            self.reporter.generate_comprehensive_report(
                real_df=data,
                synthetic_df=synth,
                generator_name=f"RealGenerator_{method}",
                output_dir=effective_output_dir or output_dir,  # Use effective dir
                target_column=target_col,
                time_col=time_col_name,
                drift_config=report_drift_config,
                discriminator=adversarial_validation,
                report_config=report_config,  # Pass the config object
            )

        # Save the generated dataset for inspection
        if save_dataset:  # Only save if save_dataset is True
            self.logger.debug("Saving dataset...")
            if not output_dir:
                raise ValueError(
                    "output_dir must be provided if save_dataset is True"
                )
            try:
                save_path = os.path.join(output_dir, f"synthetic_data_{method}.csv")
                synth.to_csv(save_path, index=False)
                self.logger.info(
                    f"Generated synthetic dataset saved to: {save_path}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save synthetic dataset: {e}")

        return synth
