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


class RealGenerator(BaseGenerator, _LatentMixin, _SynthUtilsMixin, _TabularSynthMixin, _TimeSeriesSynthMixin, _ScviSynthMixin, _PrivacySynthMixin):
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
        # BUGFIX: Reset history and active state at the start of each generation
        self.training_history = {}
        self._is_training_active = True

        # Ensure reporter is fresh and respects the current minimal mode
        from calm_data_generator.reports.QualityReporter import QualityReporter
        self.reporter = QualityReporter(minimal=self.minimal_report)
        """
        The main public method to generate synthetic data.

        Args:
            data (Union[pd.DataFrame, AnnData]): The real dataset (DataFrame) or AnnData object to be synthesized.
            n_samples (int): The number of synthetic samples to generate.
            method (str): The synthesis method to use.
            target_col (Optional[str]): The name of the target variable column.
            block_column (Optional[str]): The name of the column defining data blocks.
            output_dir (Optional[str]): Directory to save the report and dataset. Optional if save_dataset is False.
            custom_distributions (Optional[Dict]): A dictionary to specify custom distributions for columns.
            custom_distribution (Optional[Dict]): Alias for custom_distributions.
            date_config (Optional[DateConfig]): Configuration for date injection.
            balance (bool): If True, balances the distribution of the target column.
            save_dataset (bool): If True, saves the generated dataset to a CSV file.
            drift_injection_config (Optional[List[Dict]]): List of drift injection configurations.
            dynamics_config (Optional[Dict]): Configuration for dynamics injection (feature evolution, target construction).
            **kwargs: Hyperparameters for the models.
                Common v1.2.0 parameters:
                - differentiation_factor (float): Enhances class separation in latent space (TVAE/scVI).
                - clipping_mode (str): 'strict' (default), 'permissive', or 'none' for output range control.
                - clipping_factor (float): Tolerance factor for 'permissive' clipping (default 0.1).
                - use_latent_sampling (bool): For scVI, whether to sample from real data's latent space.
            adversarial_validation (bool): If True, runs the DiscriminatorReporter to compute adversarial validation metrics (AUC).

        Returns:
            Optional[pd.DataFrame]: The generated synthetic DataFrame, or None if synthesis fails.
        """
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

            date_config = DateConfig(
                start_date=date_start,
                frequency=date_every,
                step=date_step,
                date_col=date_col,
            )

        # Merge shorthand aliases into canonical params
        if custom_distribution:
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
        # synth default
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

        # --- Post-process distribution enforcement ---
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

        # --- Constraints Application ---
        if synth is not None and constraints:
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

        if synth is not None:
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

            self.logger.debug(f"Returning synth for method '{method}'.")
            self._is_training_active = False # Release log capturing
            return synth
        else:
            self.logger.debug(
                f"Synthesis method '{method}' failed to generate data (synth is None)."
            )
            self.logger.error(f"Synthesis method '{method}' failed to generate data.")
            self._is_training_active = False
            return None
