#!/usr/bin/env python3
"""
Real Data Generator - Advanced data synthesis for real datasets.

This module provides the RealGenerator class, which serves as a powerful tool for
synthesizing data that mimics the characteristics of a real-world dataset. It integrates
several synthesis methods, from classic statistical approaches to modern deep learning models.

Key Features:
- **Multiple Synthesis Methods**: Supports a variety of methods including:
  - `cart`: FCS-like method using Decision Trees.
  - `rf`: FCS-like method using Random Forests.
  - `lgbm`: FCS-like method using LightGBM.
  - `gmm`: Gaussian Mixture Models (for numeric data).
  - `ctgan`, `tvae`: Advanced deep learning/graph
  - `resample`: Simple resampling with replacement.
- **Conditional Synthesis**: Can generate data that follows custom-defined distributions for specified columns.
- **Target Balancing**: Automatically balances the distribution of the target variable.
- **Date Injection**: Capable of adding a timestamp column with configurable start dates and steps.
- **Comprehensive Reporting**: Automatically generates a detailed quality report comparing the synthetic data to the original, including visualizations and statistical metrics.
"""

import logging
import math
import os
import tempfile
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Synthcity and customized dependencies are lazy-loaded
# Model imports
# Custom logger and reporter
from calm_data_generator.generators.base import BaseGenerator
from calm_data_generator.generators.configs import DateConfig, DriftConfig, ReportConfig

# Synthcity import
from calm_data_generator.generators.persistence_models import FCSModel

from ._generate_pipeline import _GeneratePipelineMixin
from ._synth_latent import _LatentMixin
from ._synth_privacy import _PrivacySynthMixin
from ._synth_scvi import _ScviSynthMixin
from ._synth_tabular import _TabularSynthMixin
from ._synth_timeseries import _TimeSeriesSynthMixin
from ._synth_utils import _SynthUtilsMixin

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RealGenerator(
    BaseGenerator,
    _LatentMixin,
    _SynthUtilsMixin,
    _TabularSynthMixin,
    _TimeSeriesSynthMixin,
    _ScviSynthMixin,
    _PrivacySynthMixin,
    _GeneratePipelineMixin,
):
    """
    A class for advanced data synthesis from a real dataset, offering multiple generation methods and detailed reporting.
    """

    def __init__(
        self,
        auto_report: bool = True,
        minimal_report: bool = False,
        logger: Optional[logging.Logger] = None,
        random_state: Optional[int] = None,
        verbose_training: bool = False,
    ):
        # Handle Synthcity loggers (uses loguru)
        self.training_history = {}
        self._loguru_handler_id = None # Track our specific sink
        try:
            import re

            from loguru import logger as loguru_logger

            # Note: We avoid calling loguru_logger.remove() globally to not break other libs
            # We only remove our own if it existed (e.g. from a prior failed __init__)
            if hasattr(self, "_loguru_handler_id") and self._loguru_handler_id is not None:
                try:
                    loguru_logger.remove(self._loguru_handler_id)
                except ValueError as e:
                    self.logger.debug(f"Could not remove loguru handler {self._loguru_handler_id}: {e}")

            def capture_synthcity_metrics(message):
                # Only capture if THIS instance is currently the one training
                # We use a simple cross-check flag
                if not getattr(self, "_is_training_active", False):
                    return
                # Synthcity TVAE/VAE loss looks like: "[epoch/max_iter] Loss: 1.234"
                loss_match = re.search(r"Loss: ([\d\.]+)", message)
                if loss_match:
                    if "loss_evolution" not in self.training_history:
                        self.training_history["loss_evolution"] = []
                    self.training_history["loss_evolution"].append(float(loss_match.group(1)))

            # We add a sink that belongs to this specific instance
            self._loguru_handler_id = loguru_logger.add(capture_synthcity_metrics, level="DEBUG")
        except ImportError:
            pass

        # ... other initializations ...
        """
        Initializes the RealGenerator.

        Args:
            auto_report (bool): If True, automatically generates a quality report after synthesis.
            minimal_report (bool): If True, generates minimal reports (faster, no correlations/PCA).
            logger (Optional[logging.Logger]): An external logger instance. If None, a new one is created.
            random_state (Optional[int]): Seed for random number generation for reproducibility.
        """
        super().__init__(
            random_state=random_state,
            auto_report=auto_report,
            minimal_report=minimal_report,
            logger=logger,
        )
        self.verbose_training = verbose_training
        from calm_data_generator.reports.QualityReporter import QualityReporter
        self.reporter = QualityReporter(minimal=minimal_report)
        self.synthesizer = None
        self.metadata = None
        self.method = None  # Track the method used for training

    def _generate_from_fitted(self, n_samples: int) -> pd.DataFrame:
        """Helper to generate data from a fitted synthesizer."""
        # scVI / scANVI: use the same generation pipeline as _synthesize_scvi/_synthesize_scanvi
        if self.method in ("scvi", "scanvi"):
            target_col = self.metadata.get("target_col") if self.metadata else None
            if self.method == "scanvi":
                return self._synthesize_scanvi(
                    self.synthesizer.adata, n_samples, target_col=target_col
                )
            else:
                return self._synthesize_scvi(
                    self.synthesizer.adata, n_samples, target_col=target_col
                )

        if self.method in [
            "ctgan",
            "tvae",
            "great",
            "rtvae",
            "timegan",
            "timevae",
            "ddpm",
            "gears",
            "dpgan",
            "pategan",
        ]:
            # Synthcity plugins usually return a loader/object with .dataframe()
            # or sometimes just a dataframe? Let's assume standard plugin behavior.
            res = self.synthesizer.generate(count=n_samples)
            if hasattr(res, "dataframe"):
                return res.dataframe()
            return res
        elif self.method in ["cart", "rf", "lgbm", "CART", "RF", "LGBM", "xgboost"]:
            # FCS Methods
            if hasattr(self.synthesizer, "generate"):
                return self.synthesizer.generate(n_samples)
            else:
                self.logger.warning(
                    f"Synthesizer for {self.method} does not have generate method via FCSModel."
                )

        elif self.method == "copula":
            # Gaussian Copula (copulae lib)
            samples = self.synthesizer.random(n_samples)

            # Restore state
            cols = self.metadata.get("columns")
            numeric_cols = self.metadata.get("numeric_cols")
            scaler = self.metadata.get("scaler")

            if scaler is None or numeric_cols is None or len(numeric_cols) == 0:
                self.logger.warning(
                    "Copula scaler or numeric columns not found in metadata. Returning raw samples."
                )
                return pd.DataFrame(samples, columns=cols)

            # Inverse transform
            synth_numeric = pd.DataFrame(
                scaler.inverse_transform(samples), columns=numeric_cols
            )

            # We need to handle non-numeric columns if they existed?
            # _synthesize_copula handles them by resampling. We need that data?
            # If we don't have the original data, we can't resample!
            # We should probably store a sample of non-numeric data in metadata if we want to support this?
            # For now, let's just return numeric parts and warn.
            if len(cols) > len(numeric_cols):
                self.logger.warning(
                    "Original non-numeric data not stored. Only numeric columns generated for Copula."
                )
                # Fill others with NaNs or similar?
                for c in cols:
                    if c not in numeric_cols:
                        synth_numeric[c] = np.nan

            return synth_numeric[cols]

        elif self.method == "gmm":
            # sklearn GMM
            synth_data, _ = self.synthesizer.sample(n_samples)
            cols = self.metadata.get("columns") if self.metadata else None
            return pd.DataFrame(synth_data, columns=cols)
        elif self.method == "conditional_drift":
            stage_col = self.metadata.get("stage_col", "__drift_stage__")
            n_windows = self.metadata.get("n_stages", 5)
            generate_at = self.metadata.get("generate_at")

            if generate_at is None:
                stages = [str(s) for s in range(n_windows)]
            else:
                stages = [str(s) for s in generate_at]

            samples_per_stage = max(1, n_samples // len(stages))
            remainder = n_samples - samples_per_stage * len(stages)

            parts = []
            for i, stage in enumerate(stages):
                count = samples_per_stage + (1 if i < remainder else 0)
                cond = pd.DataFrame({stage_col: [stage] * count})
                part = self.synthesizer.generate(count=count, cond=cond).dataframe()
                parts.append(part)

            result = pd.concat(parts, ignore_index=True)
            result = result.drop(columns=[stage_col])
            return result

        elif self.method == "kde":
            # sklearn KDE
            synth_data = self.synthesizer.sample(n_samples, random_state=self.random_state)
            cols = self.metadata.get("columns") if self.metadata else None
            return pd.DataFrame(synth_data, columns=cols)
        elif self.method == "windowed_copula":
            fitted_copulas = self.synthesizer
            numeric_cols = self.metadata.get("numeric_cols", [])
            scalers = self.metadata.get("scalers", [])
            generate_at = self.metadata.get("generate_at", [0.0])
            n_windows = self.metadata.get("n_windows", 5)

            samples_per_point = max(1, n_samples // len(generate_at))
            parts = []
            for t in generate_at:
                idx = t * (n_windows - 1)
                i_low = min(int(idx), n_windows - 2)
                i_high = i_low + 1
                alpha = idx - i_low
                n_low = max(1, round(samples_per_point * (1 - alpha)))
                n_high = max(1, samples_per_point - n_low)
                part_low = pd.DataFrame(scalers[i_low].inverse_transform(fitted_copulas[i_low].random(n_low)), columns=numeric_cols)
                part_high = pd.DataFrame(scalers[i_high].inverse_transform(fitted_copulas[i_high].random(n_high)), columns=numeric_cols)
                parts.append(pd.concat([part_low, part_high], ignore_index=True))
            return pd.concat(parts, ignore_index=True)[:n_samples].reset_index(drop=True)

        elif self.method == "hmm":
            samples, _ = self.synthesizer.sample(n_samples)
            numeric_cols = self.metadata.get("numeric_cols")
            scaler = self.metadata.get("scaler")
            return pd.DataFrame(scaler.inverse_transform(samples), columns=numeric_cols)

        elif self.method == "diffusion":
            import torch

            model = self.synthesizer
            meta = self.metadata
            steps = meta["steps"]
            n_features = meta["n_features"]
            alphas = meta["alphas"]
            alphas_cumprod = meta["alphas_cumprod"]
            betas = meta["betas"]
            scaler = meta["scaler"]
            encoders = meta["encoders"]
            numeric_cols = meta["numeric_cols"]

            model.eval()
            with torch.no_grad():
                x = torch.randn(n_samples, n_features)
                for t in reversed(range(steps)):
                    t_batch = torch.full((n_samples,), t, dtype=torch.float32)
                    pred_noise = model(x, t_batch)

                    alpha = alphas[t]
                    alpha_cumprod = alphas_cumprod[t]
                    beta = betas[t]

                    if t > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = 0

                    x = (1 / torch.sqrt(alpha)) * (
                        x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
                    ) + torch.sqrt(beta) * noise

            synth_array = scaler.inverse_transform(x.numpy())
            synth_df = pd.DataFrame(synth_array, columns=meta["columns"])

            for col, le in encoders.items():
                synth_df[col] = (
                    synth_df[col].round().clip(0, len(le.classes_) - 1).astype(int)
                )
                synth_df[col] = le.inverse_transform(synth_df[col])

            for col in numeric_cols:
                # We don't know original type exactly unless we stored it, but we can guess or use float
                pass  # Keep as float?

            return synth_df

        from calm_data_generator.generators.tabular.CustomPluginAdapter import CustomPluginAdapter
        if isinstance(self.synthesizer, CustomPluginAdapter):
            return self.synthesizer.generate(n_samples)

        # Fallback
        self.logger.warning(
            f"Persistence for method '{self.method}' not fully implemented. Attempting generic generation."
        )
        if hasattr(self.synthesizer, "sample"):
            return pd.DataFrame(self.synthesizer.sample(n_samples)[0])
        elif hasattr(self.synthesizer, "generate"):
            return self.synthesizer.generate(n_samples)

        raise NotImplementedError(
            f"Generation from loaded model not implemented for method '{self.method}'"
        )

    def __getstate__(self):
        """Prepares the object for pickling by removing the logger."""
        state = self.__dict__.copy()
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state):
        """Restores the object from a pickle and re-initializes the logger."""
        self.__dict__.update(state)
        from calm_data_generator.logger import get_logger

        self.logger = get_logger(self.__class__.__name__)

    def save(self, path: str):
        """
        Saves the trained generator to a file.

        Args:
            path (str): File path to save the generator (e.g., 'generator.pkl').
        """
        self.logger.info(f"Saving generator model to {path}...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # We use a zip file to store the wrapper (this object) and the model separately
        # if the model is complex (like synthcity plugins which fail mostly with joblib).
        # We strip the synthesizer from self before pickling, save it using its own method
        # or joblib if supported, and then zip them together.

        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                # 1. Handle Synthesizer
                # Check if we have a synthcity-like save method
                synthesizer_backup = self.synthesizer

                # We temporarily unset the synthesizer to pickle the wrapper/metadata cleanly
                self.synthesizer = None

                # Save wrapper state
                # joblib.dump with filename None returns bytes? No, it writes to stream or returns list of filenames.
                # Actually joblib.dump doesn't easily return bytes. Pickle does.
                # Let's use tempdir strategy for safety.

                with tempfile.TemporaryDirectory() as tmpdir:
                    wrapper_path = os.path.join(tmpdir, "wrapper.pkl")
                    joblib.dump(self, wrapper_path)
                    zf.write(wrapper_path, "wrapper.pkl")

                    # Restore synthesizer to self (in memory)
                    self.synthesizer = synthesizer_backup

                    if self.synthesizer is not None:
                        native_save = False

                        # scVI / scANVI: save to a directory, then zip it
                        if self.method in ("scvi", "scanvi"):
                            try:
                                with tempfile.TemporaryDirectory() as scvi_tmpdir:
                                    scvi_save_dir = os.path.join(scvi_tmpdir, "scvi_model")
                                    self.synthesizer.save(scvi_save_dir, overwrite=True)
                                    # Add every file in the directory to the zip under scvi_model/
                                    for fname in os.listdir(scvi_save_dir):
                                        fpath = os.path.join(scvi_save_dir, fname)
                                        zf.write(fpath, os.path.join("scvi_model", fname))
                                native_save = True
                                self.logger.info(f"{self.method.upper()} model saved via native save.")
                            except Exception as e:
                                self.logger.warning(
                                    f"Native scVI/scANVI save failed: {e}. Retrying with joblib."
                                )

                        # Synthcity Plugin check: .save() returns bytes
                        if not native_save and hasattr(self.synthesizer, "save") and hasattr(
                            self.synthesizer, "load"
                        ):
                            try:
                                # Synthcity save returns bytes directly (no path arg)
                                model_bytes = self.synthesizer.save()
                                if isinstance(model_bytes, bytes):
                                    with zf.open("model.bytes", "w") as f_model:
                                        f_model.write(model_bytes)
                                    native_save = True
                                else:
                                    self.logger.warning(
                                        f"Native save returned non-bytes: {type(model_bytes)}. Retrying with joblib."
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Native save failed for {self.method}: {e}. Retrying with joblib."
                                )

                        if not native_save:
                            # Fallback to joblib
                            with tempfile.NamedTemporaryFile(
                                delete=False
                            ) as tmp_model_file:
                                joblib.dump(self.synthesizer, tmp_model_file.name)
                                tmp_model_path = tmp_model_file.name

                            zf.write(tmp_model_path, "model.pkl")
                            os.remove(tmp_model_path)

            self.logger.info("Generator saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save generator: {e}")
            raise e

    @classmethod
    def load(cls, path: str) -> "RealGenerator":
        """
        Loads a generator from a file.

        Args:
            path (str): File path to load the generator from.

        Returns:
            RealGenerator: The loaded generator instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        try:
            # Check if it is a zip file (new format) or just a pickle (old/simple format)
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as zf:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zf.extractall(tmpdir)

                        wrapper_path = os.path.join(tmpdir, "wrapper.pkl")
                        obj = joblib.load(wrapper_path)

                        if not isinstance(obj, cls):
                            raise TypeError(
                                f"Loaded object is not instances of {cls.__name__}. Got: {type(obj)}"
                            )

                        # Re-init logger
                        if not hasattr(obj, "logger") or obj.logger is None:
                            from calm_data_generator.logger import get_logger

                            obj.logger = get_logger(obj.__class__.__name__)

                        # Load model if exists
                        # Check for bytes model first (Synthcity)
                        try:
                            # Check list of files in zip
                            zip_files = zf.namelist()
                            obj.logger.info(f"Files in zip: {zip_files}")
                            loaded_model = None

                            # scVI / scANVI: restore from scvi_model/ directory in zip
                            if obj.method in ("scvi", "scanvi") and any(
                                f.startswith("scvi_model/") for f in zip_files
                            ):
                                try:
                                    scvi_files = [f for f in zip_files if f.startswith("scvi_model/")]
                                    scvi_restore_dir = os.path.join(tmpdir, "scvi_model")
                                    os.makedirs(scvi_restore_dir, exist_ok=True)
                                    for f in scvi_files:
                                        fname = os.path.basename(f)
                                        with zf.open(f) as src, open(os.path.join(scvi_restore_dir, fname), "wb") as dst:
                                            dst.write(src.read())
                                    import scvi
                                    if obj.method == "scanvi":
                                        loaded_model = scvi.model.SCANVI.load(scvi_restore_dir)
                                    else:
                                        loaded_model = scvi.model.SCVI.load(scvi_restore_dir)
                                    obj.logger.info(f"{obj.method.upper()} model loaded via native load.")
                                except Exception as e:
                                    obj.logger.warning(f"Native scVI/scANVI load failed: {e}")

                            elif "model.bytes" in zip_files:
                                model_bytes = zf.read("model.bytes")
                                if obj.method in [
                                    "ctgan",
                                    "tvae",
                                    "rtvae",
                                    "great",
                                    "timegan",
                                    "timevae",
                                    "ddpm",
                                    "gears",
                                    "dpgan",
                                    "pategan",
                                ]:
                                    try:
                                        from synthcity.plugins import Plugins

                                        plugin = Plugins().get(obj.method)
                                        loaded_model = plugin.load(model_bytes)
                                    except Exception as e:
                                        obj.logger.warning(f"Native load failed: {e}")

                            elif "model.pkl" in zip_files:
                                model_file_extracted = os.path.join(tmpdir, "model.pkl")
                                loaded_model = joblib.load(model_file_extracted)

                            if loaded_model is not None:
                                obj.synthesizer = loaded_model
                            else:
                                obj.logger.warning(
                                    "Could not load internal model. Generator might be uninitialized."
                                )

                        except Exception as e:
                            obj.logger.error(f"Failed to load internal model: {e}")
                            # Don't raise, return obj partially loaded? No, better to raise or warn.
                            # But let's allow return if user wants to inspect metadata.

                        obj.logger.info(f"Generator loaded successfully from {path}.")
                        return obj
            else:
                # Standard Pickle fallback
                obj = joblib.load(path)
                if not hasattr(obj, "logger") or obj.logger is None:
                    from calm_data_generator.logger import get_logger

                    obj.logger = get_logger(obj.__class__.__name__)
                obj.logger.info(f"Generator loaded successfully from {path}.")
                return obj

        except Exception as e:
            # We can't log easily if we failed to load the object, so raise
            raise RuntimeError(f"Failed to load generator from {path}: {e}")

    def generate_custom(
        self,
        data: pd.DataFrame,
        model,
        n_samples: int,
        fit_fn=None,
        generate_fn=None,
        postprocess_fn=None,
        method_name: str = "custom",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data using a user-supplied (custom) model via a plugin adapter.

        Wraps any external model in a CustomPluginAdapter, fits it on ``data`` and samples
        ``n_samples`` rows from it. Useful to plug third-party synthesizers into the same API.

        Args:
            data (pd.DataFrame): The real dataset used to fit the custom model.
            model: The custom model instance to wrap.
            n_samples (int): Number of synthetic rows to generate.
            fit_fn: Optional callable to fit the model. If None, the adapter uses its default.
            generate_fn: Optional callable to sample from the model. If None, the adapter uses its default.
            postprocess_fn: Optional callable applied to the generated output.
            method_name (str): Label used for logging and metadata. Defaults to "custom".
            columns (Optional[List[str]]): Columns to generate. Defaults to all columns in ``data``.

        Returns:
            pd.DataFrame: The generated synthetic DataFrame.
        """
        from calm_data_generator.generators.tabular.CustomPluginAdapter import CustomPluginAdapter


        adapter = CustomPluginAdapter(
            model= model,
            fit_fn= fit_fn,
            generate_fn= generate_fn,
            postprocess_fn= postprocess_fn,
            columns= columns or data.columns.to_list(),
            method_name= method_name
        )

        self.logger.info(f"Training custom model {method_name}")

        adapter.fit(data)

        self.synthesizer = adapter
        self.method = method_name
        self.metadata = {"columns": data.columns.tolist()}


        self.logger.info(f"Generating {n_samples} with method {method_name}")

        return adapter.generate(n_samples)

    def generate(
        self,
        data: Optional[Union[pd.DataFrame, Any]] = None,
        n_samples: Optional[int] = None,
        method: str = "cart",
        target_col: Optional[str] = None,
        block_column: Optional[str] = None,
        output_dir: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        custom_distribution: Optional[Dict] = None,
        date_config: Optional["DateConfig"] = None,
        date_start: Optional[str] = None,
        date_every: int = 1,
        date_step: Optional[Dict[str, int]] = None,
        date_col: str = "timestamp",
        # End legacy
        balance: bool = False,
        save_dataset: bool = False,
        drift_injection_config: Optional[List[Union[Dict, DriftConfig]]] = None,
        dynamics_config: Optional[Dict] = None,
        constraints: Optional[List[Dict]] = None,
        cond: Optional[Any] = None,
        adversarial_validation: bool = False,
        report_config: Optional[Union[ReportConfig, Dict]] = None,
        epsilon: Optional[float] = None,
        delta: float = 1e-5,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        The main public method to generate synthetic data.

        This method has a large surface area because it dispatches to many different
        synthesis methods (`method=...`), each with its own concerns (drift, dynamics,
        privacy, reporting). Most calls only need `data`, `n_samples`, and `method`;
        everything else is optional and grouped below by concern. For a simpler,
        sklearn-style flow that separates training from sampling, see `.fit()`/`.sample()`.

        Args:
            data (Union[pd.DataFrame, AnnData]): The real dataset (DataFrame) or AnnData
                object to be synthesized. If None, samples from a previously fitted/loaded
                model instead of training (see `n_samples`).
            n_samples (int): The number of synthetic samples to generate.
            method (str): The synthesis method to use (e.g. "cart", "ctgan", "tvae", "gmm").
            target_col (Optional[str]): The name of the target variable column.
            block_column (Optional[str]): The name of the column defining data blocks.
            cond (Optional[Any]): Conditional value(s) to bias generation towards, if the
                method supports conditional sampling.
            constraints (Optional[List[Dict]]): Hard constraints the output must satisfy
                (e.g. value ranges, uniqueness). Violating rows are regenerated/dropped.
            adversarial_validation (bool): If True, runs the DiscriminatorReporter to
                compute adversarial validation metrics (AUC) alongside the report.
            epsilon (Optional[float]): Differential-privacy budget. Only has an effect
                with method="dpgan" or method="pategan".
            delta (float): Differential-privacy delta, paired with `epsilon`.
            **kwargs: Method-specific hyperparameters.
                Common parameters:
                - differentiation_factor (float): Enhances class separation in latent
                  space (TVAE/scVI).
                - clipping_mode (str): 'strict' (default), 'permissive', or 'none' for
                  output range control.
                - clipping_factor (float): Tolerance factor for 'permissive' clipping
                  (default 0.1).
                - use_latent_sampling (bool): For scVI, whether to sample from real
                  data's latent space.

            Drift & dynamics:
                drift_injection_config (Optional[List[Dict]]): List of drift injection
                    configurations, applied to the generated output.
                dynamics_config (Optional[Dict]): Configuration for dynamics injection
                    (feature evolution, target construction).

            Reporting & output:
                output_dir (Optional[str]): Directory to save the report and dataset.
                    Optional if save_dataset is False and auto_report is False.
                save_dataset (bool): If True, saves the generated dataset to a CSV file.
                report_config (Optional[ReportConfig]): Structured report configuration
                    (target_column, privacy_check, tstr, etc.) — an alternative to
                    passing report-related args individually.

            Distributions & balancing:
                custom_distributions (Optional[Dict]): Per-column distribution overrides.
                balance (bool): If True, balances the distribution of the target column.

            Legacy shorthand aliases (prefer the modern equivalent — a `logger.warning`
            is emitted when these are used):
                custom_distribution (Optional[Dict]): Legacy alias for
                    `custom_distributions` (singular vs. plural).
                date_config (Optional[DateConfig]): Configuration for date injection.
                    Prefer this over the four `date_*` args below.
                date_start, date_every, date_step, date_col: Legacy shorthand for
                    building a `DateConfig` inline. Prefer passing `date_config=
                    DateConfig(start_date=..., frequency=..., step=..., date_col=...)`.

        Returns:
            Optional[pd.DataFrame]: The generated synthetic DataFrame, or None if synthesis fails.
        """
        # BUGFIX: Reset history and active state at the start of each generation
        self.training_history = {}
        self._is_training_active = True

        # Ensure reporter is fresh and respects the current minimal mode
        from calm_data_generator.reports.QualityReporter import QualityReporter
        self.reporter = QualityReporter(minimal=self.minimal_report)
        # Handle file paths as input
        if isinstance(data, str):
            import os

            try:
                import anndata
            except ImportError:
                anndata = None

            if not os.path.exists(data):
                self.logger.error(f"File not found: {data}")
                return None

            ext = os.path.splitext(data)[1].lower()
            if ext == ".h5ad":
                try:
                    if anndata is None:
                        raise ImportError("anndata is required for .h5ad files.")
                    self.logger.info(f"Loading AnnData from {data}...")
                    data = anndata.read_h5ad(data)
                except Exception as e:
                    self.logger.error(f"Failed to load .h5ad file: {e}")
                    return None
            elif ext == ".h5":
                try:
                    # Try loading as Pandas HDF5 first
                    self.logger.info(f"Loading HDF5 data from {data}...")
                    # We try common keys or default
                    try:
                        data = pd.read_hdf(data)
                    except (ValueError, KeyError, ImportError):
                        # If it fails, it might be an AnnData stored in H5 format
                        # or requires a specific key
                        if anndata is None:
                            raise ImportError(
                                "anndata is required for AnnData H5 files."
                            )
                        data = anndata.read_h5ad(data)
                except Exception as e:
                    self.logger.error(f"Failed to load .h5 file: {e}")
                    return None
            elif ext == ".csv":
                try:
                    self.logger.info(f"Loading CSV data from {data}...")
                    data = pd.read_csv(data)
                except Exception as e:
                    self.logger.error(f"Failed to load .csv file: {e}")
                    return None
            else:
                self.logger.error(f"Unsupported file format for direct loading: {ext}")
                return None

        # Handle generation from loaded model (data is None)
        if data is None:
            if self.synthesizer is None:
                raise ValueError(
                    "Data must be provided to train the generator, or a model must be loaded first."
                )
            if n_samples is None:
                raise ValueError(
                    "n_samples must be provided when generating from a loaded model."
                )

            self.logger.info(
                f"Generating {n_samples} samples from loaded '{self.method}' model..."
            )
            return self._generate_from_fitted(n_samples)

        if n_samples is None:
            raise ValueError("n_samples must be provided.")

        # Handle AnnData input
        original_adata = None
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            self.logger.info(
                "AnnData input detected. Converting to DataFrame for general processing."
            )
            original_adata = data
            # Convert AnnData to DataFrame for general validation and reporting
            df = data.to_df()
            # Add obs (metadata) to the dataframe
            for col in data.obs.columns:
                df[col] = data.obs[col].values
            data = df

        self._validate_generate_call(method, n_samples, epsilon, kwargs)

        report_config, effective_output_dir, date_config = self._resolve_generate_config(
            output_dir, report_config, date_config, date_start, date_every, date_step, date_col
        )

        custom_distributions = self._resolve_generate_distributions(
            data, target_col, balance, custom_distribution, custom_distributions
        )

        synth = self._dispatch_synthesis(
            method, data, n_samples, target_col, custom_distributions, cond,
            epsilon, delta, original_adata, kwargs,
        )

        synth = self._apply_generate_postprocess(
            synth, method, custom_distributions, target_col, n_samples
        )

        synth = self._apply_generate_constraints(synth, constraints, n_samples, cond)

        if synth is not None:
            synth = self._finalize_generate_output(
                synth, data, method, target_col, block_column, output_dir,
                effective_output_dir, date_config, drift_injection_config,
                dynamics_config, save_dataset, adversarial_validation, report_config,
            )
            self.logger.debug(f"Returning synth for method '{method}'.")
            self._is_training_active = False  # Release log capturing
            return synth
        else:
            self.logger.debug(
                f"Synthesis method '{method}' failed to generate data (synth is None)."
            )
            self.logger.error(f"Synthesis method '{method}' failed to generate data.")
            self._is_training_active = False
            return None

    def fit(
        self,
        data: Union[pd.DataFrame, Any],
        method: str = "cart",
        target_col: Optional[str] = None,
        **kwargs,
    ) -> "RealGenerator":
        """
        Fit the generator on `data` (sklearn-style). Call `.sample(n_samples)` afterwards
        to draw synthetic rows from the fitted model, as many times as you like, without
        retraining.

        This is a thin wrapper around `generate()`: it trains the model exactly the same
        way, but discards the throwaway dataset produced during training and skips report
        generation (no `output_dir` is used). For a one-shot call that trains and returns
        a synthetic dataset in one step, use `generate()` directly instead.

        Args:
            data (Union[pd.DataFrame, AnnData]): The real dataset to fit the model on.
            method (str): The synthesis method to use (see `generate()`). Defaults to "cart".
            target_col (Optional[str]): The name of the target variable column, if any.
            **kwargs: Any other keyword argument accepted by `generate()` (e.g.
                custom_distributions, constraints, dynamics_config, differentiation_factor).

        Returns:
            RealGenerator: self, so calls can be chained, e.g. `gen.fit(df).sample(1000)`.
        """
        kwargs.pop("n_samples", None)
        kwargs.pop("output_dir", None)
        kwargs.pop("save_dataset", None)
        try:
            n_rows = len(data)
        except TypeError:
            n_rows = 100
        fit_n_samples = max(1, min(n_rows, 500))

        self.generate(
            data=data,
            n_samples=fit_n_samples,
            method=method,
            target_col=target_col,
            save_dataset=False,
            **kwargs,
        )
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Draw `n_samples` synthetic rows from a previously fitted model (sklearn-style).

        Call `.fit(data)` first. Unlike `generate()`, this does not retrain — it reuses
        the model fitted by the last `.fit()` (or `.load()`) call.

        Args:
            n_samples (int): Number of synthetic rows to generate.

        Returns:
            pd.DataFrame: The generated synthetic DataFrame.
        """
        if self.synthesizer is None:
            raise ValueError(
                "No fitted model found. Call `.fit(data)` (or `.load(path)`) before `.sample(n_samples)`."
            )
        result = self.generate(data=None, n_samples=n_samples)
        if result is None:
            raise RuntimeError("Sampling from the fitted model failed; check logs for details.")
        return result
