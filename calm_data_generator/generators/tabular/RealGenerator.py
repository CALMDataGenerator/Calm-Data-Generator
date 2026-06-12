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

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RealGenerator(BaseGenerator):
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

    def get_latest_embeddings(self) -> Optional[np.ndarray]:
        """
        Returns the latest latent embeddings generated by methods supporting
        latent space differentiation (e.g., tvae, ctgan, scvi). Returns None if
        no embeddings were saved.
        """
        return getattr(self, "latest_embeddings", None)

    def get_training_history(self) -> Optional[dict]:
        """
        Returns the training history dict from the last run, if available.
        For scVI/scANVI: contains metrics like 'train_loss_epoch', 'elbo_train',
        'reconstruction_loss_train', 'kl_local_train'.
        Returns None if not available (e.g., non-scVI models).
        """
        return getattr(self, "training_history", None)

    def get_synthesizer_model(self) -> Any:
        """
        Returns the underlying synthesizer model used for generation.

        For deep-learning methods the returned object is the high-level wrapper
        (e.g. Synthcity plugin, scvi.model.SCVI, GEARS model).  Use
        ``get_encoder`` / ``get_decoder`` to reach the internal networks.

        Returns None if no model has been trained yet.
        """
        return self.synthesizer

    def get_encoder(self) -> Optional[Any]:
        """
        Returns the encoder network of the trained model, if it exists.

        Supported methods and what is returned:
            - tvae / rtvae: the Synthcity PyTorch encoder
            - scvi: ``module.z_encoder`` (or ``module.encoder``)
            - ddpm / timegan / timevae / fflows: the internal encoder if the
              Synthcity plugin exposes one, otherwise None
            - gears: None (GEARS has no explicit encoder/decoder split)
            - gmm / copula / bayesian_network: None (no neural encoder)
            - smote / adasyn / resample / cart / rf / lgbm: None

        Returns:
            The encoder module, or None if not available.
        """
        if not self.synthesizer:
            return None

        if self.method in ("tvae", "rtvae"):
            try:
                return self.synthesizer.model.model.encoder
            except AttributeError:
                return None

        if self.method == "scvi":
            module = getattr(self.synthesizer, "module", None)
            if module is None:
                return None
            return getattr(module, "z_encoder", None) or getattr(module, "encoder", None)

        if self.method in ("ddpm", "timegan", "timevae", "fflows"):
            try:
                return self.synthesizer.model.model.encoder
            except AttributeError:
                return None

        return None

    def get_decoder(self) -> Optional[Any]:
        if not self.synthesizer:
            return None

        if self.method in ("tvae", "rtvae"):
            try:
                return self.synthesizer.model.model.decoder
            except AttributeError:
                return None

        if self.method == "scvi":
            module = getattr(self.synthesizer, "module", None)
            if module is None:
                return None
            return getattr(module, "decoder", None)

        if self.method in ("ddpm", "timegan", "timevae", "fflows"):
            try:
                return self.synthesizer.model.model.decoder
            except AttributeError:
                return None

        return None

    def encode_to_latent(
        self,
        data,
        target_col: Optional[str] = None,
    ) -> "torch.Tensor":
        """
        Encodes ``data`` into the model's latent space, handling the full
        preprocessing pipeline for each supported method.

        - **tvae / rtvae**: runs TabularEncoder → VAE encoder → returns ``mu``
          (mean of the approximate posterior).  The TabularEncoder was fitted
          on the *complete* training DataFrame (including ``target_col``), so
          ``data`` must contain the same columns, in the same order, as the
          training data — ``target_col`` included.
        - **scvi / scanvi**: calls ``model.get_latent_representation()`` and
          returns the result as a float32 tensor.

        Unlike :meth:`get_encoder`, which exposes the raw PyTorch module, this
        method handles the method-specific preprocessing automatically. Use it
        when implementing external drift analyses that need latent
        representations for both TVAE and SCVI without rewriting the encoding
        pipeline each time.

        Parameters
        ----------
        data:
            - TVAE/RTVAE: a :class:`pandas.DataFrame` with **all** columns
              used during training (including ``target_col``).
            - SCVI/SCANVI: a :class:`pandas.DataFrame` *or* an
              ``AnnData`` object whose ``var_names`` match the training genes.
        target_col:
            Label column used to build the optional conditioning tensor for
            TVAE models trained with a conditional dimension.  For SCVI this
            argument is ignored.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(n_samples, n_latent)``.

        Raises
        ------
        RuntimeError
            If no model has been trained yet, or the method is not supported.
        """
        if not self.synthesizer:
            raise RuntimeError("No trained model found. Call generate() first.")

        if self.method in ("tvae", "rtvae"):
            tabular_model = self.synthesizer.model
            pytorch_model = tabular_model.model
            pytorch_model.eval()

            # TVAE trains the TabularEncoder on the FULL DataFrame (including
            # target_col).  We must pass the same columns here; do not drop
            # target_col before encoding.
            data_encoded = tabular_model.encode(data)
            data_tensor = torch.tensor(
                data_encoded.values, dtype=torch.float32
            ).to(pytorch_model.device)

            # Append conditional dimensions expected by the encoder.
            # n_units_conditional may be non-zero even without explicit cond (Synthcity default).
            # Always append a cond tensor of the right size; use label one-hot when target_col
            # is available, zeros otherwise.
            cond_dim = getattr(pytorch_model, "n_units_conditional", 0)
            if cond_dim > 0:
                if target_col and isinstance(data, pd.DataFrame) and target_col in data.columns:
                    cond_tensor = self._build_cond_tensor(
                        data, target_col, cond_dim, pytorch_model.device
                    )
                else:
                    cond_tensor = torch.zeros(len(data_tensor), cond_dim, device=pytorch_model.device)
                data_tensor = torch.cat([data_tensor, cond_tensor], dim=1)

            with torch.no_grad():
                encoder_out = pytorch_model.encoder(data_tensor)
                mu = encoder_out[0] if isinstance(encoder_out, (tuple, list)) else encoder_out
            return mu

        if self.method in ("scvi", "scanvi"):
            model = self.synthesizer
            z = model.get_latent_representation()
            return torch.tensor(z, dtype=torch.float32)

        raise RuntimeError(
            f"encode_to_latent() is not supported for method '{self.method}'. "
            "Supported: tvae, rtvae, scvi, scanvi."
        )

    def decode_from_latent(
        self,
        z: "torch.Tensor",
        data=None,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Decodes latent vectors ``z`` back to the original feature space.

        - **tvae / rtvae**: runs VAE decoder → TabularEncoder inverse_transform.
        - **scvi / scanvi**: calls the model's generative pass and returns
          a DataFrame with the original gene columns.

        Parameters
        ----------
        z:
            Float32 tensor of shape ``(n_samples, n_latent)`` produced by
            :meth:`encode_to_latent` (or a manually perturbed version of it).
        data:
            Reference data used to infer library size (SCVI only) and column
            names.  Can be a :class:`pandas.DataFrame` or ``AnnData``.
        target_col:
            Label column to attach to the output DataFrame.  When provided
            together with ``data``, labels are copied from the reference.

        Returns
        -------
        pandas.DataFrame
            Decoded samples in the original feature space.

        Raises
        ------
        RuntimeError
            If no model has been trained yet, or the method is not supported.
        """
        if not self.synthesizer:
            raise RuntimeError("No trained model found. Call generate() first.")

        n_samples = z.shape[0]

        if self.method in ("tvae", "rtvae"):
            tabular_model = self.synthesizer.model
            pytorch_model = tabular_model.model
            pytorch_model.eval()

            cond_dim = getattr(pytorch_model, "n_units_conditional", 0)
            cond_tensor = None
            if cond_dim > 0:
                if target_col is not None and data is not None and isinstance(data, pd.DataFrame) and target_col in data.columns:
                    cond_tensor = self._build_cond_tensor(
                        data, target_col, cond_dim, pytorch_model.device
                    )
                else:
                    cond_tensor = torch.zeros(n_samples, cond_dim, device=pytorch_model.device)

            z = z.to(pytorch_model.device)
            with torch.no_grad():
                reconstructed = pytorch_model.decoder(z, cond_tensor)

            # Get encoded column names: encode the full data (TabularEncoder
            # was fitted on the complete DataFrame including target_col).
            enc_cols = tabular_model.encode(data).columns if data is not None else None
            reconstructed_df = pd.DataFrame(
                reconstructed.cpu().numpy(),
                columns=enc_cols,
            )
            synth_df = tabular_model.decode(reconstructed_df)
            return synth_df

        if self.method in ("scvi", "scanvi"):
            model = self.synthesizer
            device = model.device if hasattr(model, "device") else next(model.module.parameters()).device

            if data is not None and hasattr(data, "X"):
                raw = self._safe_to_dense(data.X)
                orig_log_lib = np.log(raw.sum(axis=1) + 1e-8)
            elif data is not None and isinstance(data, pd.DataFrame):
                num_cols = data.select_dtypes(include=[np.number]).columns
                if target_col:
                    num_cols = [c for c in num_cols if c != target_col]
                orig_log_lib = np.log(data[num_cols].values.sum(axis=1) + 1e-8)
            else:
                orig_log_lib = np.zeros(n_samples)

            z = z.to(device)
            library_tensor = torch.tensor(
                orig_log_lib, dtype=torch.float32
            ).unsqueeze(1).to(device)
            batch_index = torch.zeros(n_samples, 1, dtype=torch.long).to(device)

            y_tensor = None
            if getattr(model.module, "dispersion", "gene") == "gene-label" and data is not None:
                try:
                    label_registry = model.adata_manager.get_state_registry("labels")
                    cat_mapping = label_registry.categorical_mapping
                    label_map = {str(cat): i for i, cat in enumerate(cat_mapping)}
                    if hasattr(data, "obs") and target_col in data.obs.columns:
                        labels = data.obs[target_col].astype(str).values
                    elif isinstance(data, pd.DataFrame) and target_col in data.columns:
                        labels = data[target_col].astype(str).values
                    else:
                        labels = None
                    if labels is not None:
                        y_tensor = torch.tensor(
                            [label_map.get(l, 0) for l in labels],
                            dtype=torch.long,
                        ).unsqueeze(1).to(device)
                except Exception:
                    pass

            with torch.no_grad():
                gen_out = model.module.generative(
                    z=z,
                    library=library_tensor,
                    batch_index=batch_index,
                    y=y_tensor,
                )
                px_dist = gen_out["px"]
                vals = (
                    px_dist.sample().cpu().numpy()
                    if hasattr(px_dist, "sample")
                    else px_dist.mean.cpu().numpy()
                )

            col_names = (
                data.var_names.tolist()
                if hasattr(data, "var_names")
                else [c for c in (data.columns if isinstance(data, pd.DataFrame) else []) if c != target_col]
            )
            synth_df = pd.DataFrame(vals, columns=col_names)
            if target_col is not None and data is not None:
                if hasattr(data, "obs") and target_col in data.obs.columns:
                    synth_df[target_col] = data.obs[target_col].values[:n_samples]
                elif isinstance(data, pd.DataFrame) and target_col in data.columns:
                    synth_df[target_col] = data[target_col].values[:n_samples]
            return synth_df

        raise RuntimeError(
            f"decode_from_latent() is not supported for method '{self.method}'. "
            "Supported: tvae, rtvae, scvi, scanvi."
        )

    @staticmethod
    def to_anndata(
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        obs_cols: Optional[List[str]] = None
    ):
        """
        Converts a generated synthetic DataFrame back to an AnnData object.

        Args:
            df (pd.DataFrame): The synthetic data.
            target_col (str): Column to use as 'cell_type' in adata.obs.
            obs_cols (list): Additional columns to move from features (X) to metadata (obs).

        Returns:
            anndata.AnnData: The data in single-cell format.
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("anndata is required. Please install it with 'pip install anndata'")

        obs_cols = obs_cols or []
        if target_col and target_col not in obs_cols:
            obs_cols.append(target_col)

        # Filter existing obs_cols
        obs_cols = [c for c in obs_cols if c in df.columns]

        # Features (X)
        x_cols = [c for c in df.columns if c not in obs_cols]

        # Numeric only for X usually
        numeric_x = df[x_cols].select_dtypes(include=[np.number])
        if len(numeric_x.columns) < len(x_cols):
             warnings.warn(f"Dropping non-numeric columns from X: {set(x_cols) - set(numeric_x.columns)}")

        adata = ad.AnnData(X=numeric_x.values.astype(np.float32))
        adata.var_names = numeric_x.columns.tolist()
        adata.obs_names = [f"cell_{i}" for i in range(len(df))]

        if obs_cols:
            adata.obs = df[obs_cols].copy()
            if target_col and target_col in adata.obs.columns:
                adata.obs["cell_type"] = adata.obs[target_col].astype(str)

        return adata

    def _get_model_params(
        self, method: str, user_params: Optional[Dict] = None
    ) -> Dict:
        """Merges user parameters with defaults based on the method."""
        # Standard parameter names (matching sklearn/lightgbm/Synthcity APIs)
        defaults = {
            # FCS methods (CART, RF, LGBM)
            "iterations": 10,
            # GMM
            "n_components": 5,
            "covariance_type": "full",
            # Synthcity (CTGAN, TVAE)
            "epochs": 300,
            "batch_size": 100,
            # SMOTE/ADASYN
            "k_neighbors": 5,  # SMOTE
            "n_neighbors": 5,  # ADASYN
            # Time Series
            "sequence_key": None,
            # Diffusion
            "steps": 50,
        }
        params = defaults.copy()
        if user_params:
            params.update(user_params)
        return params

    def _validate_method(self, method: str):
        """Validates the synthesis method."""
        valid_methods = [
            "cart",
            "rf",
            "hmm",
            "kde",
            "conditional_drift",
            "xgboost",
            "great",
            "rtvae",
            "lgbm",
            "gmm",
            "copula",
            "ctgan",
            "tvae",
            "bn",
            "resample",
            "adasyn",
            "smote",
            "diffusion",
            "ddpm",
            "timegan",
            "timevae",
            "fflows",
            "scvi",
            "scanvi",
            "gears",
            "windowed_copula",
            "dpgan",
            "pategan",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown synthesis method '{method}'. Valid methods are: {valid_methods}"
            )

    def _get_synthesizer(
        self,
        method: str,
        **model_kwargs,
    ):
        """Initializes and returns the appropriate Synthcity plugin."""
        try:
            from synthcity.plugins import Plugins
        except ImportError:
            raise ImportError(
                "synthcity is required for this method. Please install it."
            )

        if self.random_state is not None and "random_state" not in model_kwargs:
            model_kwargs["random_state"] = self.random_state
        return Plugins().get(method, **model_kwargs)

    @staticmethod
    def _normalize_epoch_params(model_kwargs: dict) -> dict:
        """Renames the user-facing ``epochs`` key to Synthcity's ``n_iter``.

        Several Synthcity plugins expose training length under ``n_iter``,
        but callers commonly pass ``epochs``. This helper centralises the
        rename so each ``_synthesize_*`` method does not duplicate the check.
        """
        if "epochs" in model_kwargs:
            model_kwargs["n_iter"] = model_kwargs.pop("epochs")
        return model_kwargs

    def _concat_and_shuffle(self, parts: list) -> pd.DataFrame:
        """Concatenates a list of DataFrames and shuffles the result.

        Used after generating per-class chunks so that the final output is
        not ordered by class. Uses ``self.random_state`` for reproducibility.
        """
        return (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1.0, random_state=self.random_state)
            .reset_index(drop=True)
        )

    @staticmethod
    def _safe_to_dense(X) -> np.ndarray:
        """Converts sparse arrays / matrices to a dense ``numpy.ndarray``.

        Handles three cases that come up when working with AnnData/scipy:
        objects exposing ``.toarray()`` (sparse matrices), ``np.matrix``
        instances exposing ``.A1``, and plain array-likes.
        """
        if hasattr(X, "toarray"):
            arr = X.toarray()
        else:
            arr = np.array(X)
        if hasattr(arr, "A1"):
            arr = arr.A1
        return arr

    @staticmethod
    def _build_cond_tensor(
        data: pd.DataFrame,
        target_col: str,
        cond_dim: int,
        device,
    ) -> "torch.Tensor":
        """Builds the one-hot conditioning tensor expected by Synthcity VAEs.

        Maps each label in ``data[target_col]`` to an index based on the
        order of unique classes, then scatters a 1.0 onto that index in a
        ``(len(data), cond_dim)`` zero tensor (clamping indices to fit the
        conditional dimension).
        """
        target_series = data[target_col]
        unique_classes = target_series.unique()
        label_to_idx = {str(c): i for i, c in enumerate(unique_classes)}
        label_indices = torch.tensor(
            [label_to_idx.get(str(l), 0) for l in target_series.values],
            dtype=torch.long,
            device=device,
        )
        cond_tensor = torch.zeros(len(data), cond_dim, device=device)
        cond_tensor.scatter_(
            1, label_indices.unsqueeze(1).clamp(max=cond_dim - 1), 1.0
        )
        return cond_tensor

    def _validate_custom_distributions(
        self, custom_distributions: Dict, data: pd.DataFrame
    ) -> Dict:
        """Validates and normalizes custom distribution dictionaries."""
        if not isinstance(custom_distributions, dict):
            raise TypeError("custom_distributions must be a dictionary.")
        validated_distributions = custom_distributions.copy()
        for col, dist in validated_distributions.items():
            if col not in data.columns:
                raise ValueError(
                    f"Column '{col}' specified in custom_distributions does not exist in the dataset."
                )
            if not isinstance(dist, dict):
                raise TypeError(
                    f"The distribution for column '{col}' must be a dictionary."
                )
            if not dist:
                self.logger.warning(
                    f"Distribution for column '{col}' is empty. It will be ignored."
                )
                continue
            if any(p < 0 for p in dist.values()):
                raise ValueError(f"Proportions for column '{col}' cannot be negative.")
            total_proportion = sum(dist.values())
            if not math.isclose(total_proportion, 1.0):
                self.logger.warning(
                    f"Proportions for column '{col}' do not sum to 1.0 (sum={total_proportion}). They will be normalized."
                )
                validated_distributions[col] = {
                    k: v / total_proportion for k, v in dist.items()
                }
        return validated_distributions

    def _patch_synthcity_encoder(self):
        """
        Monkeypatches the Synthcity TabularEncoder to show a progress bar during fitting.
        This is a workaround as Synthcity's encoder fitting can be slow on large datasets
        due to BayesianGMM components.
        """
        try:
            import synthcity.logger as log
            from synthcity.plugins.core.models.tabular_encoder import TabularEncoder

            # Avoid double patching
            if getattr(TabularEncoder, "_is_patched", False):
                return

            def fit_with_progress(self_encoder, raw_data, discrete_columns=None):
                from synthcity.utils.dataframe import discrete_columns as find_cat_cols
                from synthcity.utils.serialization import dataframe_hash

                if discrete_columns is None:
                    discrete_columns = find_cat_cols(
                        raw_data, self_encoder.categorical_limit
                    )

                self_encoder.output_dimensions = 0
                self_encoder._column_raw_dtypes = raw_data.infer_objects().dtypes
                self_encoder._column_transform_info_list = []

                # --- START PATCH: Add tqdm ---
                # Use tqdm.write to ensure it prints well with the progress bar
                tqdm.write(
                    "Building Synthcity metadata/encoding (this may take a while)..."
                )

                columns_to_encode = [
                    name
                    for name in raw_data.columns
                    if name not in self_encoder.whitelist
                ]

                # Use tqdm for progress bar
                pbar = tqdm(columns_to_encode, desc="Encoding Features", leave=False)

                for name in pbar:
                    # Update description to show current column
                    pbar.set_description(f"Encoding '{name}'")

                    column_hash = dataframe_hash(raw_data[[name]])
                    log.info(f"Encoding {name} {column_hash}")
                    ftype = "discrete" if name in discrete_columns else "continuous"
                    column_transform_info = self_encoder._fit_feature(
                        raw_data[name], ftype
                    )

                    self_encoder.output_dimensions += (
                        column_transform_info.output_dimensions
                    )
                    self_encoder._column_transform_info_list.append(
                        column_transform_info
                    )

                pbar.close()
                # --- END PATCH ---

                return self_encoder

            TabularEncoder.fit = fit_with_progress
            TabularEncoder._is_patched = True

        except ImportError:
            pass  # Synthcity not installed or structure changed

    def _synthesize_split_by_class(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        synthcity_plugin: str,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Trains a separate model per class and merges the results proportionally.

        This is the most reliable way to generate data where classes are clearly
        separable (high ARI). Each model sees only its own class, so the generated
        data is realistic and stays within each class's distribution.

        The number of generated samples per class is proportional to the original
        class counts, so the class balance is preserved.

        Args:
            data: Input DataFrame with a `target_col` column.
            n_samples: Total number of synthetic samples to generate.
            target_col: Column name for class labels.
            synthcity_plugin: Name of the Synthcity plugin to use ('ctgan' or 'tvae').
            **model_kwargs: Additional parameters passed to the Synthcity plugin.

        Returns:
            A DataFrame of synthetic data with all classes merged.
        """
        self.logger.info(
            f"Split-by-class mode: training one {synthcity_plugin.upper()} per class..."
        )
        target_series = data[target_col]
        class_counts = target_series.value_counts()
        total_original = len(data)

        dfs = []
        for cls, count in class_counts.items():
            # Proportional sample count for this class
            n_cls = max(1, int(round(n_samples * count / total_original)))

            self.logger.info(
                f"  Training model for class '{cls}' ({count} orig. samples, "
                f"generating {n_cls} synthetic)..."
            )
            subset = data[target_series == cls].drop(columns=[target_col])

            syn = self._get_synthesizer(synthcity_plugin, **model_kwargs)
            syn.fit(subset)

            synth = syn.generate(count=n_cls, random_state=self.random_state).dataframe()
            synth[target_col] = cls
            dfs.append(synth)

        result = pd.concat(dfs, ignore_index=True)
        self.logger.info(
            f"Split-by-class complete. Generated {len(result)} samples across "
            f"{len(class_counts)} classes."
        )
        return result



    def _synthesize_ctgan(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        cond: Optional[Any] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using CTGAN via Synthcity.

        When ``custom_distributions`` is provided and ``target_col`` is set,
        CTGAN leverages its conditional generator: a separate generation call
        is made per class with the exact sample count dictated by the
        distribution, then the results are merged.  This produces *genuinely
        new* synthetic rows per class instead of resampling an unconditional
        output.
        """
        self.logger.info("Starting CTGAN synthesis via Synthcity...")
        self._patch_synthcity_encoder()  # Apply patch
        model_kwargs = self._normalize_epoch_params(model_kwargs)

        # Filter out parameters not supported by CTGAN plugin
        model_kwargs.pop("differentiation_factor", None)
        model_kwargs.pop("clipping_mode", None)
        model_kwargs.pop("clipping_factor", None)

        syn = self._get_synthesizer("ctgan", **model_kwargs)
        _fit_kw = {"cond": cond} if cond is not None else {}
        syn.fit(data, **_fit_kw)
        self.synthesizer = syn
        self.method = "ctgan"
        self.metadata = {"columns": data.columns.tolist()}

        # Conditional generation per class when distributions are requested
        col = None
        if custom_distributions and target_col and target_col in custom_distributions:
            col = target_col
        elif custom_distributions:
            col = next(iter(custom_distributions))

        if cond is not None and custom_distributions:
            self.logger.warning(
                "cond and custom_distributions both provided for CTGAN — cond ignored when generating per class."
            )

        if col and col in data.columns:
            dist = custom_distributions[col]
            self.logger.info(
                f"CTGAN: generating conditionally per class on '{col}' — {dist}"
            )
            frames = []
            for cls, proportion in dist.items():
                n_cls = max(1, round(n_samples * proportion))
                self.logger.info(
                    f"  Generating {n_cls} samples for class '{cls}'..."
                )
                cls_df = syn.generate(count=n_cls, random_state=self.random_state).dataframe()
                cls_df[col] = cls
                frames.append(cls_df)
            synth_df = self._concat_and_shuffle(frames)
        else:
            gen_kwargs = {"count": n_samples}
            if cond is not None:
                gen_kwargs["cond"] = cond
            synth_df = syn.generate(**gen_kwargs, random_state=self.random_state).dataframe()

        return synth_df

    def _synthesize_great(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes data using GREAT via Synthcity.

        Args:
            data: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            target_col: Optional target column for latent differentiation.
            custom_distributions: Optional distributions to follow.

        """
        self.logger.info("Starting GREAT synthesis via Synthcity...")
        self._patch_synthcity_encoder()  # Apply patch
        model_kwargs = self._normalize_epoch_params(model_kwargs)

        syn = self._get_synthesizer("great", **model_kwargs)
        try:
            syn.fit(data)
        except Exception as e:
            self.logger.error(f"GREAT training failed: {e}")
            return None

        self.synthesizer = syn
        self.method = "great"
        self.metadata = {"columns": data.columns.tolist()}


        synth_df = syn.generate(count=n_samples, random_state=self.random_state).dataframe()

        return synth_df

    def _synthesize_tvae(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        cond: Optional[Any] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes data using TVAE via Synthcity.

        Args:
            data: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            target_col: Optional target column for latent differentiation.
            **model_kwargs: Parameters for TVAE and v1.2.0 logic:
                - differentiation_factor (float): latent space separation factor.
                - clipping_mode (str): 'strict', 'permissive', or 'none'.
                - clipping_factor (float): tolerance for permissive clipping.
                - epochs (int): number of training epochs.
        """
        self.logger.info("Starting TVAE synthesis via Synthcity...")
        self._patch_synthcity_encoder()  # Apply patch
        model_kwargs = self._normalize_epoch_params(model_kwargs)
        differentiation_factor = model_kwargs.pop("differentiation_factor", 0.0)
        clipping_mode = model_kwargs.pop("clipping_mode", "strict")
        clipping_factor = model_kwargs.pop("clipping_factor", 0.1)

        syn = self._get_synthesizer("tvae", **model_kwargs)
        # cond for fit must have len(data) rows; cond for generate must have n_samples rows.
        # Resize each independently so both calls always get the right length.
        if cond is not None:
            cond_arr = np.asarray(cond)
            if len(cond_arr) != len(data):
                self.logger.warning(
                    f"cond length {len(cond_arr)} != data length {len(data)}; np.resize will wrap/truncate."
                )
            cond_fit = pd.Series(np.resize(cond_arr, len(data)))
            cond_gen = pd.Series(np.resize(cond_arr, n_samples))
        else:
            cond_fit = cond_gen = None
        _fit_kw = {"cond": cond_fit} if cond_fit is not None else {}
        syn.fit(data, **_fit_kw)
        self.synthesizer = syn
        self.method = "tvae"
        self.metadata = {"columns": data.columns.tolist(), "target_col": target_col}

        if differentiation_factor > 0.0 and target_col and target_col in data.columns:
              synth_df = self.apply_latent_differentiation(
                        syn=syn,
                        data=data,
                        n_samples=n_samples,
                        target_col=target_col,
                        differentiation_factor=differentiation_factor,
                        method="tvae",
                        clipping_mode=clipping_mode,
                        clipping_factor=clipping_factor)
        else:
            # Custom distributions
            col = None
            if custom_distributions and target_col and target_col in custom_distributions:
                col = target_col
            elif custom_distributions:
                col = next(iter(custom_distributions))

            if col and col in data.columns:
                dist = custom_distributions[col]
                self.logger.info(
                    f"TVAE: generating conditionally per class on '{col}' — {dist}"
                )
                frames = []
                for cls, proportion in dist.items():
                    n_cls = max(1, round(n_samples * proportion))
                    cls_df = syn.generate(count=n_cls, random_state=self.random_state).dataframe()
                    cls_df[col] = cls
                    frames.append(cls_df)
                synth_df = self._concat_and_shuffle(frames)
            else:
                gen_kwargs = {"count": n_samples}
                if cond_gen is not None:
                    gen_kwargs["cond"] = cond_gen
                synth_df = syn.generate(**gen_kwargs, random_state=self.random_state).dataframe()

        return synth_df

    def _synthesize_conditional_drift(
        self,
        data: pd.DataFrame,
        n_samples: int,
        time_col: Optional[str]  = None,
        n_stages: int = 5,
        base_method: str = "tvae",
        general_stages: Optional[List[int]] =  None,
        custom_distributions: Optional[dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        stage_col = "__drift_stage__"
        df = data.copy()

        if time_col and time_col in df.columns:
            df[stage_col] = pd.cut(df[time_col], bins=n_stages, labels=False).astype(str)
        else:
           df[stage_col]  = pd.cut(np.arange(len(df)), bins=n_stages, labels=False).astype(str)

        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader


        plugin_name = "tvae" if base_method =="tvae" else "ctgan"
        syn = Plugins().get(plugin_name, **kwargs)

        syn.fit(GenericDataLoader(df))

        self.synthesizer = syn
        self.method = "conditional_drift"
        self.metadata = {
            "stage_col" : stage_col,
            "time_col" : time_col,
            "n_stages" : n_stages,
            "base_method" : base_method,
            "columns" : data.columns.tolist()
        }

        stages_to_generate = (
            [str(s) for s in general_stages ]
            if general_stages  is not None
            else [str(s) for s in range(n_stages)]
        )
        sample_per_stage = max(1, n_samples // len(stages_to_generate))
        remainder = n_samples - sample_per_stage * len(stages_to_generate)

        generated_parts = []

        for i, _ in enumerate(stages_to_generate):
            count = sample_per_stage + (1 if i < remainder else 0)
            part = syn.generate(count=count, random_state=self.random_state).dataframe()
            generated_parts.append(part)

        result = pd.concat(generated_parts, ignore_index=True)

        result = result.drop(columns=[stage_col])
        if custom_distributions:
            col = next(iter(custom_distributions), None)
            if col and col in result.columns:
                dist = custom_distributions[col]
                self.logger.info(
                    f" Applying custom_distributions on '{col}' — {dist} (resampling from staged result)"
                )
                frames = []
                for cls, proportion in dist.items():
                    n_cls = max(1, round(n_samples * proportion))
                    cls_pool = result[result[col] == cls]
                    if len(cls_pool) == 0:
                        cls_pool = result
                    frames.append(
                        cls_pool.sample(n=n_cls, replace=len(cls_pool) < n_cls,
                                        random_state=self.random_state)
                        .assign(**{col: cls})
                    )
                result = self._concat_and_shuffle(frames)


        return result
    def _synthesize_rtvae(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict] = None,
        target_col: Optional[str] = None,
        cond: Optional[Any] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes data using RTVAE via Synthcity.

        Args:
            data: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            target_col: Optional target column for latent differentiation.
            custom_distributions: Optional distributions to follow.
            **model_kwargs: Parameters for TVAE and v1.2.0 logic:
                - differentiation_factor (float): latent space separation factor.
                - clipping_mode (str): 'strict', 'permissive', or 'none'.
                - clipping_factor (float): tolerance for permissive clipping.
                - epochs (int): number of training epochs.
        """
        self.logger.info("Starting RTVAE synthesis via Synthcity...")
        self._patch_synthcity_encoder()  # Apply patch
        model_kwargs = self._normalize_epoch_params(model_kwargs)
        differentiation_factor = model_kwargs.pop("differentiation_factor", 0.0)
        clipping_mode = model_kwargs.pop("clipping_mode", "strict")
        clipping_factor = model_kwargs.pop("clipping_factor", 0.1)

        syn = self._get_synthesizer("rtvae", **model_kwargs)
        _fit_kw = {"cond": cond} if cond is not None else {}
        try:
            syn.fit(data, **_fit_kw)
        except Exception as e:
            self.logger.error(f"RTVAE training failed: {e}")
            return None

        self.synthesizer = syn
        self.method = "rtvae"
        self.metadata = {"columns": data.columns.tolist(), "target_col": target_col}

        if differentiation_factor > 0.0 and target_col and target_col in data.columns:
            synth_df = self.apply_latent_differentiation(
                        syn=syn,
                        data=data,
                        n_samples=n_samples,
                        target_col=target_col,
                        differentiation_factor=differentiation_factor,
                        method="rtvae",
                        clipping_mode=clipping_mode,
                        clipping_factor=clipping_factor)
        else:
            # Custom distributions
            col = None
            if custom_distributions and target_col and target_col in custom_distributions:
                col = target_col
            elif custom_distributions:
                col = next(iter(custom_distributions))

            if col and col in data.columns:
                dist = custom_distributions[col]
                self.logger.info(
                    f"RTVAE: generating conditionally per class on '{col}' — {dist}"
                )
                frames = []
                for cls, proportion in dist.items():
                    n_cls = max(1, round(n_samples * proportion))
                    cls_df = syn.generate(count=n_cls, random_state=self.random_state).dataframe()
                    cls_df[col] = cls
                    frames.append(cls_df)
                synth_df = self._concat_and_shuffle(frames)
            else:
                gen_kwargs = {"count": n_samples}
                if cond is not None:
                    gen_kwargs["cond"] = cond
                synth_df = syn.generate(**gen_kwargs, random_state=self.random_state).dataframe()

        return synth_df

    def apply_latent_differentiation(
        self,
        syn: Any,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        differentiation_factor: float,
        method: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Unified implementation of latent space differentiation for TVAE and scVI.

        Follows a 5-step process:
        1. Generate latent space z(d) for samples.
        2. Calculate centers of mass (centroids) for cases (cmca) and controls (cmco).
        3. Calculate distance vector c = cmca - cmco (for case) or c = cmco - cmca (for control).
        4. Generate synthetic latent z'(d) = z(d) + a * c.
        5. Decode z'(d) and apply flexible clipping.
        """
        self.logger.info(f"Applying unified latent differentiation for {method} (factor: {differentiation_factor})")

        clipping_mode = kwargs.get("clipping_mode", "strict")
        clipping_factor = kwargs.get("clipping_factor", None)
        clip_min = kwargs.get("clip_min", None)
        clip_max = kwargs.get("clip_max", None)

        try:
            # Step 1: Generate Latent Space z(d)
            if method == "tvae":
                tabular_model = syn.model
                pytorch_model = tabular_model.model
                pytorch_model.eval()

                orig_target_dtype = data[target_col].dtype
                data_for_encode = data.copy()
                data_for_encode[target_col] = data_for_encode[target_col].astype(str)
                data_encoded = tabular_model.encode(data_for_encode)
                data_tensor = torch.tensor(data_encoded.values, dtype=torch.float32).to(pytorch_model.device)

                target_series = data[target_col]
                unique_classes = target_series.unique()
                if len(unique_classes) != 2:
                    self.logger.warning(f"Differentiation works best with 2 classes, found {len(unique_classes)}. Using first two.")

                control_val = unique_classes[0]
                case_val = unique_classes[1]

                # The TVAE encoder was trained on encoded features + conditioning dims.
                # If the model has conditional dimensions, we must append them
                # so the input size matches what the encoder expects.
                cond_dim = getattr(pytorch_model, "n_units_conditional", 0)
                with torch.no_grad():
                    cond_tensor_enc = None
                    if cond_dim > 0:
                        label_to_idx = {str(control_val): 0, str(case_val): 1}
                        labels_enc = target_series.values
                        label_indices = torch.tensor(
                            [label_to_idx.get(str(l), 0) for l in labels_enc],
                            dtype=torch.long,
                            device=pytorch_model.device,
                        )
                        cond_tensor_enc = torch.zeros(
                            len(labels_enc), cond_dim, device=pytorch_model.device
                        )
                        cond_tensor_enc.scatter_(1, label_indices.unsqueeze(1).clamp(max=cond_dim - 1), 1.0)
                        data_tensor = torch.cat([data_tensor, cond_tensor_enc], dim=1)

                    encoder_out = pytorch_model.encoder(data_tensor)
                    mu = encoder_out[0] if isinstance(encoder_out, (tuple, list)) else encoder_out
                    z = mu

            elif method == "scvi":
                model = syn
                device = model.device if hasattr(model, "device") else model.module.device
                z = torch.tensor(model.get_latent_representation(), dtype=torch.float32).to(device)
                # Support both AnnData and DataFrame as input
                if hasattr(data, "obs"):
                    unique_classes = data.obs[target_col].unique()
                else:
                    unique_classes = data[target_col].unique()
                control_val = unique_classes[0]
                case_val = unique_classes[1]
            else:
                return syn.generate(count=n_samples, random_state=self.random_state).dataframe()

            # Step 2: Calculate Centroids
            if method == "scvi" and hasattr(data, "obs"):
                labels = data.obs[target_col].astype(str).values
                mask_case = labels == str(case_val)
                mask_control = labels == str(control_val)
            elif method == "scvi":
                labels = data[target_col].astype(str).values
                mask_case = labels == str(case_val)
                mask_control = labels == str(control_val)
            else:
                labels = data[target_col].values
                mask_case = labels == case_val
                mask_control = labels == control_val

            self.logger.debug(
                f"[Differentiation] unique_classes={unique_classes}, "
                f"case_val={case_val!r} ({type(case_val).__name__}), "
                f"control_val={control_val!r} ({type(control_val).__name__}), "
                f"labels sample={labels[:5]}, mask_case sum={mask_case.sum()}, mask_control sum={mask_control.sum()}"
            )

            if mask_case.sum() == 0 or mask_control.sum() == 0:
                raise ValueError(
                    f"Empty mask for differentiation: mask_case={mask_case.sum()}, "
                    f"mask_control={mask_control.sum()}. "
                    f"unique_classes={unique_classes}, labels sample={labels[:5]}"
                )

            cmca = z[mask_case].mean(dim=0)
            cmco = z[mask_control].mean(dim=0)

            if torch.isnan(cmca).any() or torch.isnan(cmco).any():
                raise ValueError("Centroid calculation returned NaNs after masking.")

            # Step 3: Calculate distance vector
            distance_vector_case = cmca - cmco       # direction to push case samples
            distance_vector_control = cmco - cmca    # direction to push control samples

            self.logger.info(f"[Differentiation] Centroid '{case_val}' (case):    {cmca.cpu().numpy().round(4)}")
            self.logger.info(f"[Differentiation] Centroid '{control_val}' (control): {cmco.cpu().numpy().round(4)}")
            self.logger.info(f"[Differentiation] Distance vector (case→control):  {distance_vector_case.cpu().numpy().round(4)}")
            self.logger.info(f"[Differentiation] ||distance||: {torch.norm(distance_vector_case).item():.4f}")
            self.logger.info(f"[Differentiation] Shift applied: factor={differentiation_factor} × ||d||={torch.norm(distance_vector_case).item():.4f} = {differentiation_factor * torch.norm(distance_vector_case).item():.4f}")

            # Step 4: Enhance Latent Space — push classes apart
            z_prime = z.clone()
            z_prime[mask_case] += differentiation_factor * distance_vector_case
            z_prime[mask_control] += differentiation_factor * distance_vector_control

            cmca_prime = z_prime[mask_case].mean(dim=0)
            cmco_prime = z_prime[mask_control].mean(dim=0)
            self.logger.info(f"[Differentiation] New centroid '{case_val}':    {cmca_prime.cpu().numpy().round(4)}")
            self.logger.info(f"[Differentiation] New centroid '{control_val}': {cmco_prime.cpu().numpy().round(4)}")
            self.logger.info(f"[Differentiation] New ||distance||: {torch.norm(cmca_prime - cmco_prime).item():.4f}")

            # Step 5: Decode
            if method == "tvae":
                with torch.no_grad():

                    # Synthcity's Decoder.forward(X, cond) expects latent and
                    # conditioning SEPARATELY — it concatenates them internally.
                    # n_units_conditional lives on the VAE (pytorch_model), not on the Decoder.
                    cond_dim = getattr(pytorch_model, "n_units_conditional", 0)

                    cond_tensor = None
                    if cond_dim > 0:
                        label_to_idx = {str(control_val): 0, str(case_val): 1}
                        label_indices = torch.tensor(
                            [label_to_idx.get(str(l), 0) for l in labels],
                            dtype=torch.long,
                            device=pytorch_model.device,
                        )
                        cond_tensor = torch.zeros(
                            len(labels), cond_dim, device=pytorch_model.device
                        )
                        cond_tensor.scatter_(1, label_indices.unsqueeze(1).clamp(max=cond_dim - 1), 1.0)

                    reconstructed_tensor = pytorch_model.decoder(z_prime, cond_tensor)

                reconstructed_df = pd.DataFrame(
                    reconstructed_tensor.cpu().numpy(),
                    columns=data_encoded.columns
                )
                synth_df = syn.model.decode(reconstructed_df)
                synth_df[target_col] = data[target_col].values
                try:
                    synth_df[target_col] = synth_df[target_col].astype(orig_target_dtype)
                except (ValueError, TypeError):
                    pass

            elif method == "scvi":
                with torch.no_grad():
                    # Preserve library size logic
                    if hasattr(data, "X"):
                        orig_lib_size = self._safe_to_dense(data.X).sum(axis=1)
                        orig_log_lib = np.log(orig_lib_size + 1e-8)
                    else:
                        orig_log_lib = np.zeros(len(data))

                    library_tensor = torch.tensor(orig_log_lib, dtype=torch.float32).unsqueeze(1).to(model.device)
                    batch_index = torch.zeros(len(z), 1, dtype=torch.long).to(model.device)

                    y_tensor = None
                    if getattr(model.module, "dispersion", "gene") == "gene-label":
                        label_registry = model.adata_manager.get_state_registry("labels")
                        cat_mapping = label_registry.categorical_mapping
                        label_map = {str(cat): i for i, cat in enumerate(cat_mapping)}
                        y_tensor = torch.tensor(
                            [label_map[str(m)] for m in labels],
                            dtype=torch.long
                        ).unsqueeze(1).to(model.device)

                    gen_outputs = model.module.generative(
                        z=z_prime,
                        library=library_tensor,
                        batch_index=batch_index,
                        y=y_tensor,
                    )
                    px_dist = gen_outputs["px"]
                    synth_values = px_dist.sample().cpu().numpy() if hasattr(px_dist, "sample") else px_dist.mean.cpu().numpy()

                # Use var_names from adata if possible, or columns from data
                col_names = data.var_names if hasattr(data, "var_names") else [c for c in data.columns if c != target_col]
                synth_df = pd.DataFrame(synth_values, columns=col_names)
                synth_df[target_col] = labels

            # Sample/Match size
            if len(synth_df) != n_samples:
                synth_df = synth_df.sample(
                    n=n_samples,
                    replace=(n_samples > len(synth_df)),
                    random_state=self.random_state
                ).reset_index(drop=True)

            # Clipping and Integer Preservation
            numeric_cols = synth_df.select_dtypes(include=[np.number]).columns
            # Resolve column access for both DataFrame and AnnData
            data_columns = data.var_names if hasattr(data, "var_names") else data.columns
            for col in numeric_cols:
                if col in data_columns:
                    col_data = data[:, col].X if hasattr(data, "var_names") else data[col]
                    if hasattr(col_data, "toarray"):
                        col_data = col_data.toarray().ravel()
                    elif hasattr(col_data, "A1"):
                        col_data = col_data.A1
                    col_series = pd.Series(col_data).dropna()
                    orig_min = col_series.min()
                    orig_max = col_series.max()

                    if clipping_mode == "none":
                        # No clipping, only preserve integer dtype
                        if pd.api.types.is_integer_dtype(col_series):
                            synth_df[col] = synth_df[col].round().astype(int)
                        continue

                    col_clip_min = clip_min.get(col) if isinstance(clip_min, dict) else clip_min
                    col_clip_max = clip_max.get(col) if isinstance(clip_max, dict) else clip_max

                    if col_clip_min is None and col_clip_max is None:
                        if clipping_mode == "permissive" and clipping_factor is not None:
                            col_range = orig_max - orig_min
                            low = orig_min - (clipping_factor * col_range)
                            high = orig_max + (clipping_factor * col_range)
                        else:  # strict: original range
                            low, high = orig_min, orig_max
                        col_clip_min, col_clip_max = low, high

                    synth_df[col] = synth_df[col].clip(lower=col_clip_min, upper=col_clip_max)

                    if pd.api.types.is_integer_dtype(col_series):
                        synth_df[col] = synth_df[col].round().astype(int)

            return synth_df

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Unified differentiation failed for {method}: {e}. Falling back to standard generation.")
            if method == "scvi":
                # Fallback: sample from prior without differentiation
                self.logger.warning("scVI fallback: generating from standard latent prior (no differentiation).")
                import scvi as _scvi_mod  # noqa: F401
                n_latent = syn.module.n_latent
                rng = np.random.default_rng(self.random_state)
                latent_samples = rng.standard_normal((n_samples, n_latent)).astype(np.float32)
                latent_tensor = torch.tensor(latent_samples).to(syn.device)
                mean_lib = float(np.log(10000))
                library_tensor = torch.full((n_samples, 1), mean_lib, dtype=torch.float32).to(syn.device)
                batch_index = torch.zeros(n_samples, 1, dtype=torch.long).to(syn.device)
                with torch.no_grad():
                    y_fallback = None
                    if getattr(syn.module, "dispersion", "gene") == "gene-label":
                        y_fallback = torch.zeros(n_samples, 1, dtype=torch.long).to(syn.device)
                    gen_out = syn.module.generative(z=latent_tensor, library=library_tensor, batch_index=batch_index, y=y_fallback)
                    px_dist = gen_out["px"]
                    vals = px_dist.sample().cpu().numpy() if hasattr(px_dist, "sample") else px_dist.mean.cpu().numpy()
                col_names = syn.adata.var_names.tolist()
                return pd.DataFrame(vals, columns=col_names)
            return syn.generate(count=n_samples, random_state=self.random_state).dataframe()


    def _synthesize_bn(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using Bayesian Networks via Synthcity.

        Bayesian Networks model conditional dependencies between variables,
        making them especially useful for clinical and structured tabular data
        where causal relationships between features matter.

        Args:
            data: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            target_col: Optional target column name (informational).
            **model_kwargs: Additional parameters passed to the Synthcity BN plugin.
                - n_iter: int = 1000 - Training iterations.
                - struct_learning_n_iter: int = 1000 - Structure learning iterations.
                - struct_learning_search_method: str = 'tree_search' - Structure learning method.

        Returns:
            Synthetic DataFrame.
        """
        self.logger.info("Starting Bayesian Network synthesis via Synthcity...")
        try:
            from synthcity.plugins import Plugins  # noqa: F401
        except ImportError:
            raise ImportError(
                "Bayesian Network synthesis requires synthcity. "
                "Install with: pip install synthcity"
            )

        self._patch_synthcity_encoder()
        model_kwargs = self._normalize_epoch_params(model_kwargs)

        syn = self._get_synthesizer("bayesian_network", **model_kwargs)
        syn.fit(data)
        self.synthesizer = syn
        self.method = "bayesian_network"
        self.metadata = {"columns": data.columns.tolist()}
        return syn.generate(count=n_samples, random_state=self.random_state).dataframe()


    def _synthesize_windowed_copula(
        self,
        data: pd.DataFrame,
        n_samples: int,
        time_col: Optional[str] = None,
        n_windows: int = 5,
        generate_at: Optional[List[float]] = None,
        **kwargs
    ) -> pd.DataFrame:

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Windowed copula required at least one numeric column")
        df = data.sort_values(time_col)[numeric_cols].copy() if time_col and time_col in data.columns else data[numeric_cols].copy()
        windowed_size = len(df) // n_windows
        windows = [df.iloc[i * windowed_size:(i + 1) * windowed_size] for i in range(n_windows)]
        windows[-1] = df.iloc[(n_windows -1) * windowed_size:]

        from copulae import GaussianCopula
        from sklearn.preprocessing import MinMaxScaler

        fitted_copulas = []
        scalers= []
        for window_df in windows:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(window_df.values)
            cop = GaussianCopula(dim=len(numeric_cols))
            cop.fit(scaled)
            fitted_copulas.append(cop)
            scalers.append(scaler)

        if generate_at is None:
            generate_at = [i / max(n_windows - 1, 1) for i in range(n_windows)]

        parts = []
        samples_per_point = max(1, n_samples // len(generate_at))

        for t in generate_at:
            # Ventanas vecinas
            idx = t * (n_windows - 1)
            i_low = min(int(idx), n_windows - 2)
            i_high = i_low + 1
            alpha = idx - i_low  # peso hacia la ventana superior

            n_low = max(1, round(samples_per_point * (1 - alpha)))
            n_high = max(1, samples_per_point - n_low)

            raw_low = fitted_copulas[i_low].random(n_low)
            if raw_low.ndim == 1:
                raw_low = raw_low.reshape(1, -1)
            raw_high = fitted_copulas[i_high].random(n_high)
            if raw_high.ndim == 1:
                raw_high = raw_high.reshape(1, -1)
            part_low = pd.DataFrame(scalers[i_low].inverse_transform(raw_low), columns=numeric_cols)
            part_high = pd.DataFrame(scalers[i_high].inverse_transform(raw_high), columns=numeric_cols)

            parts.append(pd.concat([part_low, part_high], ignore_index=True))

        result = pd.concat(parts, ignore_index=True)[:n_samples].reset_index(drop=True)

        self.synthesizer = fitted_copulas
        self.method = "windowed_copula"
        self.metadata = {
            "numeric_cols": numeric_cols,
            "n_windows": n_windows,
            "generate_at": generate_at,
            "scalers" : scalers,
            "columns": data.columns.tolist()
        }

        return result

    def _synthesize_copula(
        self,
        data: pd.DataFrame,
        n_samples: int,
        method: str,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using Copulae (Gaussian Copula)."""
        self.logger.info("Starting synthesis using Gaussian Copula...")
        try:
            from copulae import GaussianCopula
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            raise ImportError(
                "copulae and scikit-learn are required for the 'copula' method."
            )

        # Preprocessing: Copulas work on [0, 1] margins
        # We'll use a simple MinMax scaler for now to get to [0, 1],
        # but true copulas usually use empirical CDF transofmration.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("Copula synthesis requires at least some numeric columns.")

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data[numeric_cols])

        # Fit copula
        cop = GaussianCopula(dim=len(numeric_cols))
        cop.fit(X_scaled)

        # Store state for persistence
        self.synthesizer = cop
        self.method = "copula"
        self.metadata = {
            "columns": data.columns.tolist(),
            "numeric_cols": numeric_cols,
            "scaler": scaler,
        }

        # Sample
        samples = cop.random(n_samples)

        # Inverse transform
        synth_numeric = pd.DataFrame(
            scaler.inverse_transform(samples), columns=numeric_cols
        )

        # Handle non-numeric columns by simple resampling (naive approach for consistency)
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            self.logger.warning(
                "Copula method currently only models numeric correlations. Non-numeric columns will be resampled independently."
            )
            synth_non_numeric = (
                data[non_numeric_cols]
                .sample(n=n_samples, replace=True)
                .reset_index(drop=True)
            )
            synth = pd.concat([synth_numeric, synth_non_numeric], axis=1)
        else:
            synth = synth_numeric

        return synth[data.columns]  # Restore original order

    def _synthesize_resample(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data by resampling from the original dataset, with optional weighting."""
        self.logger.info("Starting synthesis by resampling...")
        self.synthesizer = None
        self.method = "resample"
        if not custom_distributions:
            return data.sample(
                n=n_samples, replace=True, random_state=self.random_state
            )
        self.logger.info(
            f"Applying custom distributions via weighted resampling: {custom_distributions}"
        )
        self.logger.warning(
            "The 'resample' method with custom distributions changes proportions but does not generate new data."
        )
        col_to_condition = (
            target_col
            if target_col and target_col in custom_distributions
            else next(iter(custom_distributions))
        )
        dist = custom_distributions[col_to_condition]
        weights = pd.Series(0.0, index=data.index)
        for category, proportion in dist.items():
            weights[data[col_to_condition] == category] = proportion
        if weights.sum() == 0:
            self.logger.warning(
                "Weights are all zero. Falling back to uniform resampling."
            )
            return data.sample(
                n=n_samples, replace=True, random_state=self.random_state
            )
        return data.sample(
            n=n_samples, replace=True, random_state=self.random_state, weights=weights
        )

    def _synthesize_smote(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        custom_distributions: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using SMOTE (Synthetic Minority Over-sampling Technique).

        When ``custom_distributions`` contains the ``target_col``, the
        proportions are translated into absolute counts and passed as
        ``sampling_strategy`` to SMOTE so it generates the exact class
        distribution requested instead of always balancing 50/50.
        """
        self.logger.info("Starting SMOTE synthesis...")
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for SMOTE. Please install it."
            )

        if not target_col:
            raise ValueError("target_col is required for SMOTE synthesis.")

        X = data.drop(columns=target_col)
        y = data[target_col]

        if not X.select_dtypes(exclude=np.number).empty:
            self.logger.warning(
                "Standard SMOTE does not handle categorical features well. Use SMOTE-NC or encode first."
            )

        try:
            k_neighbors = kwargs.get("k_neighbors", 5)

            # Build sampling_strategy from custom_distributions if provided
            sampling_strategy: Any = "auto"
            if custom_distributions and target_col in custom_distributions:
                dist = custom_distributions[target_col]
                original_counts = y.value_counts().to_dict()
                # SMOTE only oversamples — target counts must be >= original
                sampling_strategy = {}
                for cls, proportion in dist.items():
                    target_count = max(
                        original_counts.get(cls, 0),
                        round(n_samples * proportion),
                    )
                    sampling_strategy[cls] = target_count
                self.logger.info(
                    f"SMOTE sampling_strategy from custom_distributions: "
                    f"{sampling_strategy}"
                )

            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state,
            )
            X_res, y_res = smote.fit_resample(X, y)
            self.synthesizer = smote
            self.method = "smote"
            data_res = pd.concat([X_res, y_res], axis=1)

            if len(data_res) < n_samples:
                return data_res.sample(
                    n=n_samples, replace=True, random_state=self.random_state
                )
            else:
                return data_res.sample(
                    n=n_samples, replace=False, random_state=self.random_state
                )

        except Exception as e:
            self.logger.error(f"SMOTE synthesis failed: {e}")
            raise e

    def _synthesize_adasyn(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        custom_distributions: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using ADASYN (Adaptive Synthetic Sampling).

        When ``custom_distributions`` contains the ``target_col``, the
        proportions are translated into absolute counts and passed as
        ``sampling_strategy`` to ADASYN.
        """
        self.logger.info("Starting ADASYN synthesis...")
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for ADASYN. Please install it."
            )

        if not target_col:
            raise ValueError("target_col is required for ADASYN synthesis.")

        X = data.drop(columns=target_col)
        y = data[target_col]

        try:
            n_neighbors = kwargs.get("n_neighbors", 5)

            # Build sampling_strategy from custom_distributions if provided
            sampling_strategy: Any = "auto"
            if custom_distributions and target_col in custom_distributions:
                dist = custom_distributions[target_col]
                original_counts = y.value_counts().to_dict()
                sampling_strategy = {}
                for cls, proportion in dist.items():
                    target_count = max(
                        original_counts.get(cls, 0),
                        round(n_samples * proportion),
                    )
                    sampling_strategy[cls] = target_count
                self.logger.info(
                    f"ADASYN sampling_strategy from custom_distributions: "
                    f"{sampling_strategy}"
                )

            adasyn = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=n_neighbors,
                random_state=self.random_state,
            )
            X_res, y_res = adasyn.fit_resample(X, y)
            self.synthesizer = adasyn
            self.method = "adasyn"
            data_res = pd.concat([X_res, y_res], axis=1)

            if len(data_res) < n_samples:
                return data_res.sample(
                    n=n_samples, replace=True, random_state=self.random_state
                )
            else:
                return data_res.sample(
                    n=n_samples, replace=False, random_state=self.random_state
                )
        except Exception as e:
            self.logger.error(f"ADASYN synthesis failed: {e}")
            raise e

    # REMOVED: _synthesize_timegan and _synthesize_dgan methods
    # These methods required ydata-synthetic library which is not used in this project.
    # For time series synthesis, use Synthcity's time series models or other alternatives.



    def _synthesize_ddpm(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes data using Synthcity's TabDDPM (Tabular Denoising Diffusion).

        TabDDPM is a more advanced diffusion model specifically designed for tabular data.
        It supports multiple architectures (MLP, ResNet, TabNet) and advanced schedulers.

        Args:
            data: Input DataFrame
            n_samples: Number of samples to generate
            **kwargs: Additional parameters for TabDDPM:
                - n_iter: int = 1000 - Training epochs
                - lr: float = 0.002 - Learning rate
                - batch_size: int = 1024 - Batch size
                - num_timesteps: int = 1000 - Diffusion timesteps
                - model_type: str = "mlp" - Model architecture ("mlp", "resnet", "tabnet")
                - scheduler: str = "cosine" - Beta scheduler ("cosine", "linear")
                - gaussian_loss_type: str = "mse" - Loss type ("mse", "kl")
                - is_classification: bool = False - Whether task is classification

        Returns:
            Synthetic DataFrame
        """
        self.logger.info("Starting TabDDPM synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
        except ImportError:
            raise ImportError(
                "Synthcity is required for diffusion/ddpm synthesis. "
                "Install with: pip install synthcity"
            )

        # Extract DDPM-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        lr = kwargs.get("lr", 0.002)
        batch_size = kwargs.get("batch_size", 1024)
        num_timesteps = kwargs.get("num_timesteps", 1000)
        model_type = kwargs.get("model_type", "mlp")
        scheduler = kwargs.get("scheduler", "cosine")
        gaussian_loss_type = kwargs.get("gaussian_loss_type", "mse")
        is_classification = kwargs.get("is_classification", False)

        # Load plugin
        plugin = Plugins().get(
            "ddpm",
            n_iter=n_iter,
            lr=lr,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            model_type=model_type,
            scheduler=scheduler,
            gaussian_loss_type=gaussian_loss_type,
            is_classification=is_classification,
        )

        # Prepare data
        loader = GenericDataLoader(data)

        # Train
        self.logger.info(f"Training TabDDPM for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "ddpm"

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic samples...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TabDDPM synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_timegan(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Synthcity's TimeGAN.

        TimeGAN is designed for sequential/temporal data with multiple entities.
        It learns both temporal dynamics and feature distributions.

        Args:
            data: Input DataFrame with temporal structure
            n_samples: Number of sequences to generate
            **kwargs: Additional parameters for TimeGAN:
                - n_iter: int = 1000 - Training epochs
                - n_units_hidden: int = 100 - Hidden units
                - batch_size: int = 128 - Batch size
                - lr: float = 0.001 - Learning rate

        Returns:
            Synthetic DataFrame with temporal structure
        """
        self.logger.info("Starting TimeGAN synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            self.logger.error("Synthcity not available for TimeGAN.")
            raise ImportError(
                "TimeGAN requires synthcity. Install with: pip install synthcity"
            )

        # Extract TimeGAN-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        n_units_hidden = kwargs.get("n_units_hidden", 100)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        # Load plugin
        plugin = Plugins().get(
            "timegan",
            n_iter=n_iter,
            n_units_hidden=n_units_hidden,
            batch_size=batch_size,
            lr=lr,
        )

        # Prepare time series data for TimeSeriesDataLoader.
        # synthcity expects:
        #   temporal_data: list of DataFrames, one per sequence (numeric features only)
        #   observation_times: list of numeric arrays, one per sequence
        #   outcome: optional per-sequence outcome Series
        sequence_key = kwargs.get("sequence_key", None)
        time_key = kwargs.get("time_key", None)
        target_col = kwargs.get("target_col", None)

        def _to_numeric_times(times_array):
            """Convert datetime arrays to numeric seconds from sequence start."""
            s = pd.Series(times_array)
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    s = pd.to_datetime(s)
                    s = (s - s.iloc[0]).dt.total_seconds().astype(float)
                except Exception as e:
                    self.logger.debug(f"Could not parse datetime column; falling back to integer index. Reason: {e}")
                    s = pd.Series(range(len(times_array)), dtype=float)
            return s.tolist()

        if sequence_key and sequence_key in data.columns:
            exclude = {sequence_key}
            if time_key:
                exclude.add(time_key)
            # Separate target (static per sequence) from temporal features
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                # Outcome: one value per sequence (first value, since it's constant)
                outcome_series = (
                    data.groupby(sequence_key)[target_col]
                    .first()
                    .reset_index(drop=True)
                )
            else:
                outcome_series = None

            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = []
            observation_times = []
            for _, grp in data.groupby(sequence_key, sort=False):
                grp = grp.reset_index(drop=True)
                if time_key and time_key in grp.columns:
                    obs_times = _to_numeric_times(grp[time_key].values)
                else:
                    obs_times = list(range(len(grp)))
                temporal_data.append(grp[feature_cols].reset_index(drop=True))
                observation_times.append(obs_times)

            # synthcity requires outcome as DataFrame, not Series
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0] * len(temporal_data)})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )
        else:
            # Fallback: treat the entire DataFrame as ONE multi-step sequence.
            # This is the correct approach for a flat time series with no sequence_key.
            exclude = set()
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = (
                    data[target_col].iloc[[0]].reset_index(drop=True)
                )  # one outcome per sequence
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = [data[feature_cols].reset_index(drop=True)]
            observation_times = [list(range(len(data)))]
            # synthcity requires outcome as DataFrame, not Series
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0]})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )

        # Train
        self.logger.info(f"Training TimeGAN for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "timegan"

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TimeGAN synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_timevae(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Synthcity's TimeVAE.

        TimeVAE is a variational autoencoder designed for temporal data.
        It's generally faster than TimeGAN and works well for regular time series.

        Args:
            data: Input DataFrame with temporal structure
            n_samples: Number of sequences to generate
            **kwargs: Additional parameters for TimeVAE:
                - n_iter: int = 1000 - Training epochs
                - decoder_n_layers_hidden: int = 2 - Decoder layers
                - decoder_n_units_hidden: int = 100 - Decoder units
                - batch_size: int = 128 - Batch size
                - lr: float = 0.001 - Learning rate

        Returns:
            Synthetic DataFrame with temporal structure
        """
        self.logger.info("Starting TimeVAE synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            self.logger.error("Synthcity not available for TimeVAE.")
            raise ImportError(
                "TimeVAE requires synthcity. Install with: pip install synthcity"
            )

        # Extract TimeVAE-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        decoder_n_layers_hidden = kwargs.get("decoder_n_layers_hidden", 2)
        decoder_n_units_hidden = kwargs.get("decoder_n_units_hidden", 100)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        # Load plugin
        plugin = Plugins().get(
            "timevae",
            n_iter=n_iter,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            batch_size=batch_size,
            lr=lr,
        )

        # Prepare time series data for TimeSeriesDataLoader — same logic as _synthesize_timegan.
        # synthcity requires temporal_data as list of DataFrames, observation_times as list of
        # numeric arrays, and outcome as a DataFrame (not a flat input DataFrame).
        sequence_key = kwargs.get("sequence_key", None)
        time_key = kwargs.get("time_key", None)
        target_col = kwargs.get("target_col", None)

        def _to_numeric_times(times_array):
            s = pd.Series(times_array)
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    s = pd.to_datetime(s)
                    s = (s - s.iloc[0]).dt.total_seconds().astype(float)
                except Exception as e:
                    self.logger.debug(f"Could not parse datetime column; falling back to integer index. Reason: {e}")
                    s = pd.Series(range(len(times_array)), dtype=float)
            return s.tolist()

        if sequence_key and sequence_key in data.columns:
            exclude = {sequence_key}
            if time_key:
                exclude.add(time_key)
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = (
                    data.groupby(sequence_key)[target_col]
                    .first()
                    .reset_index(drop=True)
                )
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = []
            observation_times = []
            for _, grp in data.groupby(sequence_key, sort=False):
                grp = grp.reset_index(drop=True)
                obs_times = (
                    _to_numeric_times(grp[time_key].values)
                    if time_key and time_key in grp.columns
                    else list(range(len(grp)))
                )
                temporal_data.append(grp[feature_cols].reset_index(drop=True))
                observation_times.append(obs_times)
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0] * len(temporal_data)})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )
        else:
            # Fallback: treat the entire DataFrame as ONE multi-step sequence.
            exclude = set()
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = data[target_col].iloc[[0]].reset_index(drop=True)
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = [data[feature_cols].reset_index(drop=True)]
            observation_times = [list(range(len(data)))]
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0]})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )

        # Train
        self.logger.info(f"Training TimeVAE for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "timevae"

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TimeVAE synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_fflows(
        self,
        data: pd.DataFrame,
        n_samples: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes time series data using Synthcity's FourierFlows (fflows).

        FourierFlows uses normalizing flows in the frequency domain to generate
        realistic temporal sequences. Generally more stable than TimeGAN and
        particularly effective for periodic or quasi-periodic time series.

        Args:
            data: Input DataFrame with temporal structure.
            n_samples: Number of synthetic sequences to generate.
            **kwargs:
                - sequence_key: str - Column identifying each sequence (required for multi-sequence data).
                - time_key: str - Column with timestamps/time indices.
                - target_col: str - Target column to use as outcome (separated from features).
                - n_iter: int = 1000 - Training epochs.
                - batch_size: int = 128 - Batch size.
                - lr: float = 0.001 - Learning rate.

        Returns:
            Synthetic DataFrame with temporal structure.
        """
        self.logger.info("Starting FourierFlows (fflows) synthesis via Synthcity...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            raise ImportError(
                "FourierFlows requires synthcity. Install with: pip install synthcity"
            )

        # Extract fflows-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        plugin = Plugins().get("fflows", n_iter=n_iter, batch_size=batch_size, lr=lr)

        # Reuse the same TimeSeriesDataLoader preparation logic as timegan/timevae
        sequence_key = kwargs.get("sequence_key", None)
        time_key = kwargs.get("time_key", None)
        target_col = kwargs.get("target_col", None)

        def _to_numeric_times(times_array):
            s = pd.Series(times_array)
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    s = pd.to_datetime(s)
                    s = (s - s.iloc[0]).dt.total_seconds().astype(float)
                except Exception as e:
                    self.logger.debug(f"Could not parse datetime column; falling back to integer index. Reason: {e}")
                    s = pd.Series(range(len(times_array)), dtype=float)
            return s.tolist()

        if sequence_key and sequence_key in data.columns:
            exclude = {sequence_key}
            if time_key:
                exclude.add(time_key)
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = (
                    data.groupby(sequence_key)[target_col]
                    .first()
                    .reset_index(drop=True)
                )
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data, observation_times = [], []
            for _, grp in data.groupby(sequence_key, sort=False):
                grp = grp.reset_index(drop=True)
                obs_times = (
                    _to_numeric_times(grp[time_key].values)
                    if time_key and time_key in grp.columns
                    else list(range(len(grp)))
                )
                temporal_data.append(grp[feature_cols].reset_index(drop=True))
                observation_times.append(obs_times)
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0] * len(temporal_data)})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )
        else:
            # Fallback: treat the entire DataFrame as one multi-step sequence
            exclude = set()
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = data[target_col].iloc[[0]].reset_index(drop=True)
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = [data[feature_cols].reset_index(drop=True)]
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0]})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=[list(range(len(data)))],
                outcome=_outcome,
            )

        self.logger.info(f"Training FourierFlows for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "fflows"
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()
        self.logger.info(
            f"FourierFlows synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_kde(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using Kernel Density Estimation. Only supports numeric data."""
        self.logger.info("Starting KDE synthesis...")
        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            raise ImportError("scikit-learn is required for KDE synthesis.")

        non_numeric_cols = data.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            raise ValueError(
                f"The 'kde' method only supports numeric data, but found non-numeric columns: {list(non_numeric_cols)}."
            )

        model_params = {
            "bandwidth": kwargs.get("bandwidth", 1.0),
            "kernel": kwargs.get("kernel", "gaussian"),
        }

        kde = KernelDensity(**model_params)
        kde.fit(data.values)
        self.synthesizer = kde
        self.method = "kde"
        self.metadata = {"columns": data.columns.tolist()}

        synth_data = kde.sample(n_samples, random_state=self.random_state)
        synth = pd.DataFrame(synth_data, columns=data.columns)

        if custom_distributions:
            synth = self._apply_postprocess_distribution(
                synth, custom_distributions, target_col, n_samples
            )
        return synth

    def _synthesize_gmm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using Gaussian Mixture Models. Only supports numeric data."""
        self.logger.info("Starting GMM synthesis...")
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError("scikit-learn is required for GMM synthesis.")

        non_numeric_cols = data.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            raise ValueError(
                f"The 'gmm' method only supports numeric data, but found non-numeric columns: {list(non_numeric_cols)}."
            )
        model_params = {
            "n_components": kwargs.get("n_components", 5),
            "covariance_type": kwargs.get("covariance_type", "full"),
            "random_state": self.random_state,
        }
        # Filter kwargs to only include what GaussianMixture expects if necessary,
        # but for now let's just update with what's provided.
        model_p = model_params.copy()
        model_p.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in ["n_components", "covariance_type"]
            }
        )

        gmm = GaussianMixture(**model_p)
        gmm.fit(data)
        self.synthesizer = gmm
        self.method = "gmm"
        self.metadata = {"columns": data.columns.tolist()}
        synth_data, _ = gmm.sample(n_samples)
        synth = pd.DataFrame(synth_data, columns=data.columns)

        # If the target is supposed to be classification, round the results
        if target_col and target_col in synth.columns:
            unique_values = data[target_col].nunique()
            if unique_values < 25 or (unique_values / len(data)) < 0.05:
                self.logger.info(
                    f"Rounding GMM results for target column '{target_col}' to nearest integer."
                )
                synth[target_col] = synth[target_col].round().astype(int)

        if custom_distributions:
            synth = self._apply_postprocess_distribution(
                synth, custom_distributions, target_col, n_samples
            )
        return synth

    def _report_scvi_history(self, model):
        """Log and store the training history of scVI/scANVI models."""
        if hasattr(model, "history") and isinstance(model.history, dict):
            try:
                hist = model.history
                # Store for programmatic access
                self.training_history = {k: v.copy() for k, v in hist.items() if hasattr(v, 'copy')}
                if "train_loss_epoch" in hist and not hist["train_loss_epoch"].empty:
                    final_loss = hist["train_loss_epoch"].iloc[-1].values[0]
                    self.logger.info(f"Final training loss: {final_loss:.4f}")
                    if self.verbose_training:
                        self.logger.info("Training loss evolution:")
                        for key in ["train_loss_epoch", "elbo_train", "reconstruction_loss_train"]:
                            if key in hist and not hist[key].empty:
                                self.logger.info(f"  {key}:\n{hist[key]}")
            except Exception as e:
                self.logger.debug(f"Failed to report scVI history: {e}")

    def _synthesize_scanvi(
        self,
        data: Union[pd.DataFrame, Any],
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes single-cell data using scANVI (Semi-supervised VAE).

        Requires a labeled target column. First trains a base scVI model,
        then fine-tunes scANVI for label-aware generation.

        Args:
            data: DataFrame or AnnData with numeric expression values.
            n_samples: Number of synthetic samples to generate.
            target_col: Column with cell type labels — required for scANVI.
            custom_distributions: Dict with proportions per class.
            **kwargs:
                - n_latent (int): Latent space size (default: 30).
                - n_layers (int): Number of layers (default: 1).
                - epochs (int): scVI base training epochs (default: 200).
                - scanvi_epochs (int): scANVI fine-tuning epochs (default: 20).
                - unlabeled_category (str): Label for unlabeled cells (default: 'Unknown').
                - early_stopping (bool): Use early stopping (default: True).
                - use_latent_sampling (bool): Sample from real latent space (default: True).
        """
        self.logger.info("Starting scANVI synthesis for single-cell data...")

        if not target_col:
            raise ValueError("scANVI requires target_col — the cell type label column.")

        try:
            import torch
            try:
                import torchvision  # noqa: F401
            except Exception as e:
                self.logger.debug(f"torchvision not available (optional): {e}")
            import anndata
            import scvi
        except ImportError as e:
            raise ImportError(
                f"scvi-tools, anndata and torch are required for scANVI synthesis. "
                f"Install with: pip install scvi-tools anndata torch. Original error: {e}"
            )

        # --- Build AnnData (igual que scVI) ---
        if hasattr(data, "obs") and hasattr(data, "X") and not isinstance(data, pd.DataFrame):
            adata = data.copy()
            if target_col not in adata.obs.columns:
                raise ValueError(f"target_col '{target_col}' not found in AnnData.obs.")
            adata.obs[target_col] = adata.obs[target_col].astype(str)
        else:
            if target_col not in data.columns:
                raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
            expr_cols = [c for c in data.columns if c != target_col]
            expr_data = data[expr_cols].select_dtypes(include=[np.number])
            if expr_data.empty:
                raise ValueError("No numeric columns found for scANVI synthesis.")
            adata = anndata.AnnData(X=expr_data.values.astype(np.float32))
            adata.obs_names = [f"cell_{i}" for i in range(len(data))]
            adata.var_names = list(expr_data.columns)
            adata.obs[target_col] = data[target_col].astype(str).values

        # --- Parámetros ---
        n_latent = kwargs.get("n_latent", 30)
        n_layers = kwargs.get("n_layers", 1)
        epochs = kwargs.get("epochs", kwargs.get("max_epochs", 200))
        scanvi_epochs = kwargs.get("scanvi_epochs", 20)
        unlabeled_category = kwargs.get("unlabeled_category", "Unknown")
        early_stopping = kwargs.get("early_stopping", True)
        use_latent_sampling = kwargs.get("use_latent_sampling", True)

        # --- Paso 1: entrenar scVI base (unsupervised) ---
        self.logger.info("Training base scVI model...")
        scvi.model.SCVI.setup_anndata(adata)
        scvi_model = scvi.model.SCVI(adata, n_latent=n_latent, n_layers=n_layers)
        try:
            scvi_model.train(max_epochs=epochs, early_stopping=early_stopping)
        except Exception as e:
            self.logger.error(f"scVI base training failed: {e}")
            return None

        # --- Paso 2: crear y entrenar scANVI ---
        self.logger.info(f"Fine-tuning scANVI for {scanvi_epochs} epochs...")
        try:
            scanvi_model = scvi.model.SCANVI.from_scvi_model(
                scvi_model,
                labels_key=target_col,
                unlabeled_category=unlabeled_category,
            )
            scanvi_model.train(max_epochs=scanvi_epochs, early_stopping=early_stopping)
            self._report_scvi_history(scanvi_model)
        except Exception as e:
            self.logger.error(f"scANVI training failed: {e}")
            return None

        scanvi_model.module.eval()
        self.synthesizer = scanvi_model
        self.method = "scanvi"
        self.metadata = {"columns": adata.var_names.tolist(), "target_col": target_col}

        # --- Paso 3: generar ---
        # Determinar distribución por clase
        if custom_distributions and target_col in custom_distributions:
            dist = custom_distributions[target_col]
        elif custom_distributions:
            dist = next(iter(custom_distributions.values()))
        else:
            # Proporcional a clases originales
            counts = adata.obs[target_col].value_counts(normalize=True)
            dist = counts.to_dict()

        self.logger.info(f"Generating {n_samples} samples with distribution: {dist}")

        frames = []
        for label, proportion in dist.items():
            n_cls = max(1, round(n_samples * proportion))
            cell_indices = np.where(adata.obs[target_col].values == str(label))[0]

            if len(cell_indices) == 0:
                self.logger.warning(f"No cells found for label '{label}', skipping.")
                continue

            if use_latent_sampling:
                latent = scanvi_model.get_latent_representation(indices=cell_indices)
                rng = np.random.default_rng(self.random_state)
                sample_idx = rng.choice(len(latent), size=n_cls, replace=True)
                latent_samples = latent[sample_idx].astype(np.float32)
            else:
                rng = np.random.default_rng(self.random_state)
                latent_samples = rng.standard_normal((n_cls, n_latent)).astype(np.float32)

            with torch.no_grad():
                latent_tensor = torch.tensor(latent_samples).to(scanvi_model.device)

                if use_latent_sampling:
                    orig_lib = np.sum(adata.X[cell_indices], axis=1)
                    if hasattr(orig_lib, "A1"):
                        orig_lib = orig_lib.A1
                    sampled_lib = orig_lib[sample_idx]
                    library_tensor = torch.tensor(
                        np.log(sampled_lib + 1e-8), dtype=torch.float32
                    ).unsqueeze(1).to(scanvi_model.device)
                else:
                    mean_lib = adata.X.sum(axis=1).mean()
                    library_tensor = torch.full(
                        (n_cls, 1), float(np.log(mean_lib + 1e-8)), dtype=torch.float32
                    ).to(scanvi_model.device)

                batch_index = torch.zeros(n_cls, 1, dtype=torch.long).to(scanvi_model.device)

                # Obtener label index para scANVI
                label_registry = scanvi_model.adata_manager.get_state_registry("labels")
                cat_mapping = label_registry.categorical_mapping
                label_map = {str(cat): i for i, cat in enumerate(cat_mapping)}
                label_idx = label_map.get(str(label), 0)
                y_tensor = torch.full((n_cls, 1), label_idx, dtype=torch.long).to(scanvi_model.device)

                generative_outputs = scanvi_model.module.generative(
                    z=latent_tensor,
                    library=library_tensor,
                    batch_index=batch_index,
                    y=y_tensor,
                )
                px_dist = generative_outputs["px"]
                try:
                    synth_expr = px_dist.sample() if hasattr(px_dist, "sample") else px_dist.mean
                except Exception as e:
                    self.logger.warning(f"scANVI px_dist sampling failed; using zero tensor as fallback. Reason: {e}")
                    synth_expr = torch.zeros((n_cls, adata.n_vars), dtype=torch.float32)

                if hasattr(synth_expr, "cpu"):
                    synth_expr = synth_expr.cpu()

            cls_df = pd.DataFrame(synth_expr.numpy(), columns=adata.var_names)
            cls_df[target_col] = str(label)
            frames.append(cls_df)

        if not frames:
            self.logger.error("No samples generated.")
            return None

        synth_df = self._concat_and_shuffle(frames)
        self.logger.info(f"scANVI synthesis complete. Generated {len(synth_df)} samples.")
        return synth_df

    def _synthesize_scvi(
        self,
        data: Union[pd.DataFrame, Any],
        n_samples: int,
        target_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes single-cell-like data using scVI (Variational Autoencoder).

        This method treats the input as a gene expression matrix where:
        - Rows are cells/samples
        - Columns are genes/features

        Args:
            data: DataFrame or AnnData with numeric expression values
            n_samples: Number of synthetic samples to generate
            target_col: Optional column to preserve as metadata (will be excluded from training)
            **kwargs: Additional parameters passed to scVI model:
                - differentiation_factor (float): Factor for latent space separation (v1.2.0).
                - clipping_mode (str): 'strict', 'permissive', or 'none'.
                - clipping_factor (float): % of range for permissive clipping.
                - use_latent_sampling (bool): If True, samples from real data latent space.
                - n_latent (int): Size of latent space (default: 30).

        Returns:
            DataFrame with synthetic samples
        """
        self.logger.info("Starting scVI synthesis for single-cell data...")
        self.logger.debug(f"Entering _synthesize_scvi with data type: {type(data)}")

        differentiation_factor = kwargs.pop("differentiation_factor", 0.0)
        clipping_mode = kwargs.pop("clipping_mode", "strict")
        clipping_factor = kwargs.pop("clipping_factor", 0.1)

        try:
            import torch
            # Pre-import torchvision before scvi/lightning to avoid circular import
            # (torchvision._meta_registrations depends on torchvision.extension being
            # fully initialized before lightning/torchmetrics trigger a re-import)
            try:
                import torchvision  # noqa: F401
            except Exception as e:
                self.logger.debug(f"torchvision not available (optional): {e}")
            import anndata
            import scvi
        except ImportError as e:
            raise ImportError(
                f"scvi-tools, anndata and torch are required for scVI synthesis. "
                f"Install with: pip install scvi-tools anndata torch. Original error: {e}"
            )

        # Create or use AnnData object
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            adata = data
            # Ensure target_col is in obs if provided and is string
            if target_col:
                if target_col not in adata.obs.columns:
                    self.logger.warning(
                        f"target_col '{target_col}' not found in AnnData.obs."
                    )
                else:
                    adata.obs[target_col] = adata.obs[target_col].astype(str)
        else:
            # Separate metadata from expression data
            metadata_cols = []
            if target_col and target_col in data.columns:
                metadata_cols.append(target_col)

            # Get expression columns (numeric only, excluding metadata)
            expr_cols = [c for c in data.columns if c not in metadata_cols]
            expr_data = data[expr_cols].select_dtypes(include=[np.number])

            if expr_data.empty:
                raise ValueError("No numeric columns found for scVI synthesis.")

            # Create AnnData object
            adata = anndata.AnnData(X=expr_data.values.astype(np.float32))
            adata.obs_names = [f"cell_{i}" for i in range(len(data))]
            adata.var_names = list(expr_data.columns)

            # Add metadata to obs if present
            if metadata_cols:
                for col in metadata_cols:
                    if col == target_col:
                        # Ensure labels are always strings from the start to maintain registry consistency
                        adata.obs[col] = data[col].astype(str).values
                    else:
                        adata.obs[col] = data[col].values

        # Separate parameters for model initialization and training
        n_latent = kwargs.get("n_latent", 30)
        model_setup_params = [
            "n_hidden", "n_latent", "n_layers", "dropout_rate",
            "dispersion", "gene_likelihood", "use_observed_lib_size",
            "latent_distribution"
        ]

        model_init_kwargs = {
            "n_latent": n_latent,
            "n_layers": kwargs.get("n_layers", 1),
        }

        # Pull any other model setup params from kwargs
        for param in model_setup_params:
            if param in kwargs:
                model_init_kwargs[param] = kwargs[param]

        # Use target_col for dispersion if provided and not explicitly overridden
        # CRITICAL: If use_scanvi is True, we keep scVI unsupervised to avoid size mismatches
        use_scanvi = kwargs.get("use_scanvi", False)
        if target_col and not use_scanvi and "dispersion" not in model_init_kwargs:
            model_init_kwargs["dispersion"] = "gene-label"

        # Work on a copy of adata to avoid modifying the original if passed
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            # 'adata' comes from the logic above (either original or converted)
            adata_to_train = adata.copy()
        else:
            adata_to_train = adata

        # Setup and train scVI model
        # CRITICAL: If scANVI is requested, base scVI MUST be unsupervised (no labels_key)
        if target_col and target_col in adata_to_train.obs.columns and not use_scanvi:
            self.logger.info(f"Setting up supervised scVI with labels: {target_col}")
            scvi.model.SCVI.setup_anndata(adata_to_train, labels_key=target_col)
        else:
            self.logger.info("Setting up unsupervised scVI base...")
            scvi.model.SCVI.setup_anndata(adata_to_train)

        model = scvi.model.SCVI(adata_to_train, **model_init_kwargs)

        # Map training parameters
        epochs = kwargs.get("epochs", kwargs.get("max_epochs", 200))
        early_stopping = kwargs.get("early_stopping", True)

        self.logger.info(f"Training scVI model with {epochs} epochs (Early stopping: {early_stopping})...")

        # Comprehensive list of training parameters for scVI (Trainer parameters)
        train_params = [
            "accelerator", "devices", "train_size", "validation_size",
            "shuffle_set_split", "load_sparse_tensor", "batch_size",
            "early_stopping", "datasplitter_kwargs", "plan_config",
            "plan_kwargs", "datamodule", "trainer_config"
        ]

        train_kwargs = {
            "max_epochs": epochs,
            "early_stopping": early_stopping,
        }

        # Add early_stopping_patience if provided in kwargs
        if "early_stopping_patience" in kwargs:
            train_kwargs["early_stopping_patience"] = kwargs["early_stopping_patience"]
        if "check_val_every_n_epoch" in kwargs:
            train_kwargs["check_val_every_n_epoch"] = kwargs["check_val_every_n_epoch"]

        # Add any user-provided parameters that match scVI training signature
        for param in train_params:
            if param in kwargs:
                train_kwargs[param] = kwargs[param]

        # Generation-specific parameters that should NEVER go to model.init or model.train
        gen_params = [
            "differentiation_factor", "use_scanvi",
            "use_contrastivevi", "scanvi_epochs", "scanvi_unlabeled_category",
            "use_latent_sampling", "preserve_library_size", "epochs", "max_epochs"
        ]

        # Also allow any additional kwargs to pass through to train(),
        # BUT EXCLUDE those used for model initialization OR generation logic to avoid TypeErrors.
        for k, v in kwargs.items():
            if (k not in train_kwargs and
                k not in model_setup_params and
                k not in gen_params):
                train_kwargs[k] = v

        try:
            model.train(**train_kwargs)
            self._report_scvi_history(model)
        except Exception as e:
            self.logger.error(f"scVI training failed: {e}")
            return None

        model.module.eval()  # Ensure model is in eval mode for generation
        self.synthesizer = model
        self.method = "scvi"
        # Decide generation strategy based on parameters
        use_latent_sampling = kwargs.get("use_latent_sampling", True)
        synthetic_metadata = None

        if differentiation_factor > 0.0 and target_col:
            return self.apply_latent_differentiation(
                syn=model,
                data=data,
                n_samples=n_samples,
                target_col=target_col,
                differentiation_factor=differentiation_factor,
                method="scvi",
                clipping_mode=clipping_mode,
                clipping_factor=clipping_factor,
                **kwargs
            )

        if use_latent_sampling:
            self.logger.info("Using latent sampling for more realistic generation...")
            # Use get_latent_representation to get the full latent space
            orig_latent = model.get_latent_representation()

            # Sample indices to use for generation
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(orig_latent), size=n_samples, replace=True)
            latent_samples = orig_latent[indices].astype(np.float32)

            synthetic_metadata = None
            if target_col and target_col in (
                adata_to_train.obs.columns
            ):
                source_metadata = adata_to_train.obs[target_col].values
                synthetic_metadata = source_metadata[indices]
        else:
            self.logger.info("Using standard prior sampling (random normal).")
            # latent_samples = np.random.randn(n_samples, n_latent).astype(np.float32)
            # Fetch n_latent from model
            n_latent = model.module.n_latent
            rng = np.random.default_rng(self.random_state)
            latent_samples = rng.standard_normal((n_samples, n_latent)).astype(np.float32)
            synthetic_metadata = None
            indices = np.arange(n_samples) # Dummy indices if not latent sampling

        with torch.no_grad():
            latent_tensor = torch.tensor(latent_samples).to(model.device)

            if use_latent_sampling and kwargs.get('preserve_library_size', True):
                orig_lib_size = self._safe_to_dense(adata_to_train.X).sum(axis=1)
                orig_log_lib = np.log(orig_lib_size + 1e-8)
                sampled_log_lib = orig_log_lib[indices]
                library_tensor = torch.tensor(
                    sampled_log_lib, dtype=torch.float32
                ).unsqueeze(1).to(model.device)
            else:
                mean_lib_size = adata_to_train.X.sum(axis=1).mean()
                log_library_size = np.log(mean_lib_size + 1e-8)
                self.logger.debug(
                    f"scVI Mean library size: {mean_lib_size}, log: {log_library_size}"
                )
                library_tensor = torch.full(
                    (n_samples, 1), log_library_size, dtype=torch.float32
                ).to(model.device)

            batch_index = torch.zeros(n_samples, 1, dtype=torch.long).to(model.device)

            # Handle labels for gene-label dispersion
            y_tensor = None
            if getattr(model.module, "dispersion", "gene") == "gene-label":
                try:
                    label_registry = model.adata_manager.get_state_registry("labels")
                    cat_mapping = label_registry.categorical_mapping
                    label_map = {str(cat): i for i, cat in enumerate(cat_mapping)}
                    y_tensor = torch.tensor(
                        [label_map[str(m)] for m in synthetic_metadata],
                        dtype=torch.long
                    ).unsqueeze(1).to(model.device)
                except Exception as e:
                    self.logger.warning(f"Could not map labels for gene-label dispersion (scVI): {e}")

            generative_outputs = model.module.generative(
                z=latent_tensor,
                library=library_tensor,
                batch_index=batch_index,
                y=y_tensor,
            )

            # Sample from the distribution
            # In newer scvi-tools, 'px' is a Distribution object (e.g. ZINB)
            # We sample from it to get synthetic counts
            px_dist = generative_outputs["px"]

            # Always use stochastic sampling for realistic count data.
            # While mean decoding preserves differentiation signals cleanly,
            # it produces fractional denoised values that fail structural similarity metrics
            # expecting sparsity and count distributions.
            try:
                if hasattr(px_dist, "sample"):
                    synthetic_expression = px_dist.sample()
                else:
                    synthetic_expression = px_dist.mean
            except Exception as e:
                self.logger.warning(f"scVI px_dist sampling failed; using zero tensor as fallback. Reason: {e}")
                synthetic_expression = torch.zeros(
                    (n_samples, adata.n_vars), dtype=torch.float32
                )

            if hasattr(synthetic_expression, "cpu"):
                synthetic_expression = synthetic_expression.cpu()

            synth_values = synthetic_expression.numpy()
            self.logger.debug(
                f"scVI synthesis produced values with shape: {synth_values.shape}"
            )

        # Use adata.var_names for column names (works for both DataFrame and AnnData input)
        synth_df = pd.DataFrame(synth_values, columns=adata.var_names)

        # Add metadata column with random sampling from original if present
        if target_col and target_col in (
            data.obs.columns if hasattr(data, "obs") else data.columns
        ):
            if synthetic_metadata is not None:
                synth_df[target_col] = synthetic_metadata
            else:
                # metadata_cols might be empty if adata was passed
                source_metadata = (
                    data.obs[target_col].values
                    if hasattr(data, "obs")
                    else data[target_col].values
                )
                synth_df[target_col] = np.random.default_rng(self.random_state).choice(
                    source_metadata, size=n_samples, replace=True
                )

        self.logger.info(f"scVI synthesis complete. Generated {len(synth_df)} samples.")
        return synth_df


    def _synthesize_gears(
        self,
        data: Union[pd.DataFrame, Any],
        n_samples: int,
        target_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes single-cell perturbation data using GEARS (Graph-based perturbation prediction).

        Args:
            data: DataFrame or AnnData with gene expression values
            n_samples: Number of samples to generate
            target_col: Optional column to preserve as metadata
            **kwargs: Additional parameters:
                - perturbations: List of genes to perturb (required)
                - ctrl: Control condition name (default: 'ctrl')
                - epochs: Training epochs (default: 20)
                - batch_size: Batch size (default: 32)
                - device: Device to use (default: 'cpu')

        Returns:
            DataFrame with synthetic perturbation predictions
        """
        self.logger.info(
            "Starting GEARS synthesis for single-cell perturbation data..."
        )

        try:
            import anndata
            from gears import GEARS, PertData
        except ImportError:
            raise ImportError(
                "gears and anndata are required for GEARS synthesis. "
                "Install with: pip install gears anndata"
            )

        # Get perturbations parameter (required)
        perturbations = kwargs.get("perturbations")
        if not perturbations:
            raise ValueError(
                "GEARS requires 'perturbations' parameter: list of genes to perturb. "
                "Example: perturbations=['GENE1', 'GENE2']"
            )

        # Create or use AnnData object
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            adata = data
        else:
            # Separate metadata from expression data
            metadata_cols = []
            if target_col and target_col in data.columns:
                metadata_cols.append(target_col)

            # Get expression columns (numeric only)
            expr_cols = [c for c in data.columns if c not in metadata_cols]
            expr_data = data[expr_cols].select_dtypes(include=[np.number])

            if expr_data.empty:
                raise ValueError("No numeric columns found for GEARS synthesis.")

            # Create AnnData object
            adata = anndata.AnnData(X=expr_data.values.astype(np.float32))
            adata.obs_names = [f"cell_{i}" for i in range(len(data))]
            adata.var_names = list(expr_data.columns)
            adata.var["gene_name"] = list(expr_data.columns)

            # Add condition column if not present
            ctrl_name = kwargs.get("ctrl", "ctrl")
            if "condition" not in adata.obs.columns:
                adata.obs["condition"] = ctrl_name
            adata.obs["cell_type"] = "default"

            if metadata_cols:
                for col in metadata_cols:
                    adata.obs[col] = data[col].values

        # Setup GEARS parameters
        epochs = kwargs.get("epochs", 20)
        batch_size = kwargs.get("batch_size", 32)
        device = kwargs.get("device", "cpu")
        hidden_size = kwargs.get("hidden_size", 64)

        self.logger.info(
            f"Training GEARS model with {epochs} epochs, batch_size={batch_size}..."
        )

        try:
            # Create temporary PertData object
            import os
            import tempfile

            from scipy import sparse

            # GEARS expects sparse matrix for co-expression calculation
            if not sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(adata.X)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Process data for GEARS
                pert_data = PertData(tmpdir)
                pert_data.new_data_process(
                    dataset_name="custom", adata=adata, skip_calc_de=True
                )
                pert_data.load(data_path=os.path.join(tmpdir, "custom"))

                # Inject dummy DE genes metadata if missing (required by GEARS)
                # Must be done AFTER load() because load() reads from disk
                if "non_zeros_gene_idx" not in pert_data.adata.uns:
                    # Create a mapping where each condition maps to all gene indices
                    all_gene_indices = list(range(len(pert_data.adata.var_names)))
                    pert_data.adata.uns["non_zeros_gene_idx"] = {
                        cond: all_gene_indices
                        for cond in pert_data.adata.obs["condition"].unique()
                    }

                # Prepare data split (use all data for training in this case)
                pert_data.prepare_split(split="simulation", seed=1)
                pert_data.get_dataloader(
                    batch_size=batch_size, test_batch_size=batch_size
                )

                # Initialize and train GEARS model
                gears_model = GEARS(pert_data, device=device)
                gears_model.model_initialize(hidden_size=hidden_size)

                try:
                    gears_model.train(epochs=epochs)
                except ValueError as ve:
                    # Ignore pearsonr error during validation on synthetic/small datasets
                    if "at least 2" in str(ve):
                        self.logger.warning(
                            f"GEARS validation metric calculation failed ({ve}). "
                            "Continuing as model training likely completed an epoch."
                        )
                    else:
                        raise ve

                self.synthesizer = gears_model
                self.method = "gears"

                # Generate predictions for specified perturbations
                # Format perturbations as list of lists
                if isinstance(perturbations[0], str):
                    # Single perturbation per prediction
                    pert_list = [[p] for p in perturbations]
                else:
                    # Already formatted as list of lists
                    pert_list = perturbations

                # Predict outcomes
                predictions = gears_model.predict(pert_list)

                # Convert predictions dict to DataFrame.
                # gears_model.predict() returns a dict:
                #   {('GENE', 'ctrl'): np.ndarray(shape=(n_genes,)), ...}
                if isinstance(predictions, dict):
                    # Stack all predicted arrays (one per perturbation condition)
                    pred_arrays = [v for v in predictions.values()]
                    pred_values = np.stack(
                        pred_arrays, axis=0
                    )  # shape: (n_perts, n_genes)
                elif hasattr(predictions, "cpu"):
                    pred_values = predictions.detach().cpu().numpy()
                else:
                    pred_values = np.array(predictions)

                # Generate n_samples by repeating/sampling predictions
                _rng = np.random.default_rng(self.random_state)
                if len(pred_values) >= n_samples:
                    indices = _rng.choice(
                        len(pred_values), size=n_samples, replace=False
                    )
                else:
                    indices = _rng.choice(
                        len(pred_values), size=n_samples, replace=True
                    )

                synth_values = pred_values[indices]
                synth_df = pd.DataFrame(synth_values, columns=adata.var_names)

                # Add metadata back
                if target_col and target_col in adata.obs.columns:
                    synth_df[target_col] = _rng.choice(
                        adata.obs[target_col], size=n_samples, replace=True
                    )

                self.logger.info(
                    f"GEARS synthesis complete. Generated {len(synth_df)} samples."
                )
                return synth_df

        except Exception as e:
            self.logger.error(f"GEARS synthesis failed: {e}")
            raise e

    def _synthesize_fcs_generic(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict],
        model_factory_func,
        method_name: str,
        iterations: int,
        target_col: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Generic helper for Fully Conditional Specification (FCS) synthesis.

        Args:
            data: Original dataframe.
            n_samples: Number of samples to generate.
            custom_distributions: Optional customs distribution dict.
            model_factory_func: Callable(is_classification: bool, model_params: dict) -> model instance.
            method_name: Name of the method for logging.
            iterations: Number of FCS iterations.
        """
        self.logger.info(f"Starting {method_name} (FCS-style) synthesis...")

        if custom_distributions:
            self.logger.warning(
                f"For '{method_name}' method, custom distributions are handled by resampling the training data."
            )

        # Prepare initial synthetic data (bootstrap)
        X_real = data.copy()

        # Convert datetime columns to numeric (int64 unix nanoseconds) so sklearn can handle them
        datetime_cols = X_real.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()
        for _dc in datetime_cols:
            X_real[_dc] = X_real[_dc].astype(np.int64)

        # If target column is specified and has balanced distribution requested,
        # we balance the STARTING point (bootstrap) so FCS doesn't have to work as hard
        # to move the distribution, and to avoid bias from the starting features.
        X_bootstrap_source = X_real
        if target_col and custom_distributions and target_col in custom_distributions:
            self.logger.info(f"Balancing bootstrap source for column '{target_col}'...")
            X_res, y_res = self._apply_resampling_strategy(
                X_real.drop(columns=target_col),
                X_real[target_col],
                custom_distributions[target_col],
                len(X_real),
            )
            # Reconstruct the full balanced dataframe
            X_bootstrap_source = X_res.copy()
            X_bootstrap_source[target_col] = y_res

        # Ensure object columns are category for consistency
        for col in X_real.select_dtypes(include=["object"]).columns:
            X_real[col] = X_real[col].astype("category")
            X_bootstrap_source[col] = X_bootstrap_source[col].astype("category")

        # Initial random sample
        # OPTIMIZATION: Instead of pure random sample (which might miss rare categories),
        # we repeat the original dataset as many times as possible, then sample the rest.
        n_real = len(X_bootstrap_source)
        if n_samples > n_real:
            n_repeats = n_samples // n_real
            remainder = n_samples % n_real
            X_synth_list = [X_bootstrap_source] * n_repeats
            if remainder > 0:
                X_synth_list.append(
                    X_bootstrap_source.sample(
                        n=remainder, replace=False, random_state=self.random_state
                    )
                )
            X_synth = pd.concat(X_synth_list, ignore_index=True)
            # Shuffle to break order
            X_synth = X_synth.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
        else:
            # If we need fewer samples than real data, standard sample is fine (or could be stratified)
            X_synth = X_bootstrap_source.sample(
                n=n_samples, replace=True, random_state=self.random_state
            ).reset_index(drop=True)

        # Align categories for LGBM/Categorical handling
        cat_cols = X_real.select_dtypes(include="category").columns
        for col in cat_cols:
            X_synth[col] = pd.Categorical(
                X_synth[col], categories=X_real[col].cat.categories
            )

        try:
            # Storage for persistence
            fitted_models = {}
            encoding_info = {col: X_real[col].cat.categories for col in cat_cols}

            # Marginals for initialization (using raw values for sampling)
            # We store the unique values and their counts to sample with replacement respecting distribution
            marginals = {}
            for col in X_real.columns:
                marginals[col] = X_real[
                    col
                ].values  # Storing full column values might be heavy?
                # Optimization: Store value_counts if cardinality is low, else sample?
                # Actually, storing values allows np.random.choice easily.
                # If data is huge, we might want to store just unique and counts.
                # For now let's store values (simplest for 'bootstrap' init).

            # Or better, just store the bootstrap source we created?
            # X_bootstrap_source is balanced.
            # But the persistent model generates from scratch.
            # Let's populate marginals from X_bootstrap_source (balanced if requested).
            marginals = {
                c: X_bootstrap_source[c].values for c in X_bootstrap_source.columns
            }

            for it in tqdm(range(iterations), desc=f"{method_name} Iterations"):
                # self.logger.info(f"{method_name} iteration {it + 1}/{iterations}")
                for col in X_real.columns:
                    y_real_train = X_real[col]
                    Xr_real_train = X_real.drop(columns=col)
                    Xs_synth = X_synth.drop(columns=col)

                    # Determine task type
                    is_classification = False
                    if not pd.api.types.is_numeric_dtype(y_real_train):
                        is_classification = True
                    # Heuristic for low-cardinality numeric targets (treat as class)
                    elif col == target_col:
                        unique_values = y_real_train.nunique()
                        if (
                            unique_values < 25
                            or (unique_values / len(y_real_train)) < 0.05
                        ):
                            is_classification = True

                    # Model instantiation via factory
                    model = model_factory_func(is_classification)

                    # Prepare training data (potentially applying custom distributions)
                    y_to_fit, X_to_fit = (y_real_train, Xr_real_train)
                    if custom_distributions and col in custom_distributions:
                        X_to_fit, y_to_fit = self._apply_resampling_strategy(
                            Xr_real_train,
                            y_real_train,
                            custom_distributions[col],
                            len(Xr_real_train),
                        )

                    # Encode categorical features for sklearn-based models (CART/RF)
                    # LGBM handles categories natively, but generic sklearn trees do not.
                    # We check if the model is LGBM-like by class name string check or attribute.
                    is_lgbm = "LGBM" in model.__class__.__name__

                    if not is_lgbm:
                        # Sklearn encoding — use assign() to avoid full DataFrame copy
                        cat_cols = [
                            c for c in X_to_fit.columns
                            if pd.api.types.is_categorical_dtype(X_to_fit[c])
                            or isinstance(X_to_fit[c].dtype, pd.CategoricalDtype)
                        ]
                        fit_updates = {}
                        synth_updates = {}
                        for c in cat_cols:
                            fit_cat = X_to_fit[c].astype("category")
                            try:
                                synth_cast = Xs_synth[c].astype(fit_cat.dtype)
                            except Exception as e:
                                self.logger.debug(f"dtype cast failed for column '{c}'; forcing category alignment. Reason: {e}")
                                synth_cast = pd.Categorical(
                                    Xs_synth[c],
                                    categories=fit_cat.cat.categories,
                                )
                            fit_updates[c] = fit_cat.cat.codes
                            # synth_cast is a Series (has .cat) or a pd.Categorical (has .codes directly)
                            synth_updates[c] = synth_cast.cat.codes if isinstance(synth_cast, pd.Series) else synth_cast.codes
                        X_to_fit = X_to_fit.assign(**fit_updates)
                        Xs_synth_input = Xs_synth.assign(**synth_updates)
                    else:
                        # LGBM input
                        Xs_synth_input = Xs_synth

                    try:
                        model.fit(X_to_fit, y_to_fit)
                    except ValueError as e:
                        if "Input contains NaN" in str(e):
                            raise ValueError(
                                f"The '{method_name}' method failed due to NaNs. Please pre-clean data."
                            ) from e
                        raise e

                    # Store model if last iteration
                    if it == iterations - 1:
                        import copy

                        try:
                            fitted_models[col] = copy.deepcopy(model)
                        except Exception as e:
                            self.logger.warning(f"deepcopy failed for model on column '{col}'; storing reference instead. Reason: {e}")
                            fitted_models[col] = model  # Fallback if deepcopy fails

                    if (
                        is_classification
                        and hasattr(model, "predict_proba")
                        and not (custom_distributions and col in custom_distributions)
                    ):
                        # Probabilistic sampling for better distribution preservation
                        # but we skip it if we are already forcing a distribution via resampling
                        # to avoid double-amplification/overshooting.
                        probs = model.predict_proba(Xs_synth_input)
                        classes = model.classes_

                        # Sample for each row
                        _rng = np.random.default_rng(self.random_state)
                        y_synth_pred = np.array(
                            [_rng.choice(classes, p=p) for p in probs]
                        )
                    else:
                        # Regression, balancing via resampling, or fallback
                        y_synth_pred = model.predict(Xs_synth_input)

                    # Restore categorical type if needed
                    if y_real_train.dtype.name == "category":
                        y_synth_pred = pd.Categorical(
                            y_synth_pred, categories=y_real_train.cat.categories
                        )

                    X_synth[col] = y_synth_pred

            # End of loop
            # Instantiate FCSModel
            self.synthesizer = FCSModel(
                models=fitted_models,
                marginals=marginals,
                encoding_info=encoding_info,
                visit_order=list(X_real.columns),
                random_state=self.random_state,
            )
            self.method = method_name
            self.metadata = {"columns": data.columns.tolist()}

            return X_synth
        except Exception as e:
            self.logger.error(f"{method_name} synthesis failed: {e}", exc_info=True)
            return None

    def _apply_postprocess_distribution(
        self,
        synth: pd.DataFrame,
        custom_distributions: Dict,
        target_col: Optional[str],
        n_samples: int,
    ) -> pd.DataFrame:
        """
        Resamples a synthetic DataFrame so that the target column follows
        the requested distribution.  Used as a post-processing fallback for
        methods that cannot enforce the distribution during generation
        (TVAE, DDPM, Copula, BN, GMM, scVI …).

        Args:
            synth: Already-generated synthetic DataFrame.
            custom_distributions: ``{column: {class: proportion, …}, …}``
            target_col: Primary target column (checked first in the dict).
            n_samples: Desired total number of rows in the output.

        Returns:
            Resampled DataFrame with the requested distribution.
        """
        col = (
            target_col
            if target_col and target_col in custom_distributions
            else next(iter(custom_distributions))
        )
        dist = custom_distributions[col]

        self.logger.info(
            f"Applying post-process resampling on column '{col}' to match "
            f"requested distribution: {dist}"
        )

        frames = []
        for cls, proportion in dist.items():
            count = max(1, round(n_samples * proportion))
            subset = synth[synth[col] == cls]
            if len(subset) == 0:
                self.logger.warning(
                    f"No synthetic samples found for class '{cls}' in column "
                    f"'{col}'. Cannot enforce distribution for this class."
                )
                continue
            frames.append(
                subset.sample(n=count, replace=True, random_state=self.random_state)
            )

        if not frames:
            self.logger.warning(
                "Post-process distribution resampling produced no data. "
                "Returning original synthetic dataset."
            )
            return synth

        result = self._concat_and_shuffle(frames)
        self.logger.info(
            f"Post-process resampling complete: {len(result)} samples "
            f"(requested {n_samples})."
        )
        return result

    def _apply_resampling_strategy(self, X, y, custom_dist, n_samples):
        """Applies over/under-sampling to match a custom distribution before model training."""
        try:
            original_counts = y.value_counts().to_dict()

            # If "balanced", create a uniform distribution across all present classes
            if custom_dist == "balanced":
                unique_labels = list(original_counts.keys())
                if not unique_labels:
                    return X, y
                prob = 1.0 / len(unique_labels)
                custom_dist = {label: prob for label in unique_labels}

            # Ensure we have a dict
            if not isinstance(custom_dist, dict):
                return X, y

            target_total_size = n_samples
            target_counts = {
                k: int(v * target_total_size) for k, v in custom_dist.items()
            }
            oversampling_strategy = {
                k: v for k, v in target_counts.items() if v > original_counts.get(k, 0)
            }
            undersampling_strategy = {
                k: v for k, v in target_counts.items() if v < original_counts.get(k, 0)
            }
            steps = []

            try:
                from imblearn.over_sampling import RandomOverSampler
                from imblearn.pipeline import Pipeline as ImblearnPipeline
                from imblearn.under_sampling import RandomUnderSampler
            except ImportError:
                # Fallback if imblearn not available? Or raise.
                self.logger.warning(
                    "imbalanced-learn not installed. Skipping resampling strategy."
                )
                return X, y

            if oversampling_strategy:
                steps.append(
                    (
                        "o",
                        RandomOverSampler(
                            sampling_strategy=oversampling_strategy,
                            random_state=self.random_state,
                        ),
                    )
                )
            if undersampling_strategy:
                steps.append(
                    (
                        "u",
                        RandomUnderSampler(
                            sampling_strategy=undersampling_strategy,
                            random_state=self.random_state,
                        ),
                    )
                )
            if not steps:
                return X, y
            pipeline = ImblearnPipeline(steps=steps)
            self.logger.info(
                f"Applying resampling pipeline to match distribution for column '{y.name}'."
            )
            X_res, y_res = pipeline.fit_resample(X, y)
            return X_res, y_res
        except Exception as e:
            self.logger.warning(
                f"Could not apply resampling strategy for column '{y.name}': {e}. Using original distribution."
            )
            return X, y

    def _synthesize_cart(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        iterations: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Decision Trees."""

        def model_factory(is_classification):
            try:
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            except ImportError:
                raise ImportError("scikit-learn is required for CART synthesis.")

            model_params = {"random_state": self.random_state}
            model_params.update(kwargs)

            # Filter valid params for DecisionTree to avoid the garbage defaults bug
            valid_params = {
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "random_state",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "class_weight",
                "ccp_alpha",
            }
            filtered_params = {
                k: v for k, v in model_params.items() if k in valid_params
            }

            return (
                DecisionTreeClassifier(**filtered_params)
                if is_classification
                else DecisionTreeRegressor(**filtered_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "CART",
            iterations,
            target_col,
        )
    def _synthesize_xgboost(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        iterations: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with XGBoost."""

        def model_factory(is_classification):
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("xgboost is required for XGBoost synthesis.")

            model_params = {"random_state": self.random_state, "verbosity": 0}
            model_params.update(kwargs)
            return (
                xgb.XGBClassifier(**model_params)
                if is_classification
                else xgb.XGBRegressor(**model_params)
            )

        return self._synthesize_fcs_generic(
            data, n_samples, custom_distributions, model_factory, "XGBoost", iterations, target_col,
        )

    def _synthesize_hmm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        n_components: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        from hmmlearn import hmm
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("hmm requires at least one numeric column. ")

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(data[numeric_cols].values)
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=self.random_state,
        )
        model.fit(X)

        self.synthesizer = model
        self.method = "hmm"
        self.metadata = {
            "numeric_cols": numeric_cols,
            "scaler": scaler,
            "columns": data.columns.tolist(),
        }
        samples, _ = model.sample(n_samples)
        result = pd.DataFrame(
            scaler.inverse_transform(samples),
            columns=numeric_cols
        )
        return result


    def _synthesize_rf(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        iterations: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Random Forests."""

        def model_factory(is_classification):
            try:
                from sklearn.ensemble import (
                    RandomForestClassifier,
                    RandomForestRegressor,
                )
            except ImportError:
                raise ImportError("scikit-learn is required for RF synthesis.")

            model_params = {"random_state": self.random_state, "n_jobs": 1}
            model_params.update(kwargs)

            # Filter valid params for RandomForest
            valid_params = {
                "n_estimators",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "bootstrap",
                "oob_score",
                "n_jobs",
                "random_state",
                "verbose",
                "warm_start",
                "class_weight",
                "ccp_alpha",
                "max_samples",
            }
            filtered_params = {
                k: v for k, v in model_params.items() if k in valid_params
            }

            return (
                RandomForestClassifier(**filtered_params)
                if is_classification
                else RandomForestRegressor(**filtered_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "RF",
            iterations,
            target_col,
        )

    def _synthesize_lgbm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        iterations: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with LightGBM."""

        def model_factory(is_classification):
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("lightgbm is required for LGBM synthesis.")

            model_params = {"random_state": self.random_state, "verbose": -1}
            model_params.update(kwargs)
            return (
                lgb.LGBMClassifier(**model_params)
                if is_classification
                else lgb.LGBMRegressor(**model_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "LGBM",
            iterations,
            target_col,
        )

    def _inject_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        date_start: Optional[str],
        date_every: int,
        date_step: Optional[Dict[str, int]],
    ) -> pd.DataFrame:
        """Injects a date column into the DataFrame with specified frequency and step."""
        if date_start is None:
            return df
        if not isinstance(date_every, int) or date_every <= 0:
            raise ValueError(f"date_every must be a positive integer, got {date_every}")
        step = date_step or {"days": 1}
        valid_keys = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
            "nanoseconds",
        }
        if set(step.keys()) - valid_keys:
            raise ValueError(f"Invalid date_step keys: {set(step.keys()) - valid_keys}")
        try:
            start_ts = pd.to_datetime(date_start)
        except Exception as e:
            raise ValueError(f"Invalid date_start '{date_start}': {e}") from e
        total = len(df)
        if total == 0:
            df[date_col] = pd.Series(dtype="datetime64[ns]")
            return df
        periods = (total + date_every - 1) // date_every
        anchors = [start_ts + pd.DateOffset(**step) * i for i in range(periods)]
        series = (
            pd.Series(anchors).repeat(date_every).iloc[:total].reset_index(drop=True)
        )
        if date_col not in df.columns:
            df.insert(0, date_col, series)
        else:
            df[date_col] = series
        self.logger.info(
            f"[RealGenerator] Injected date column '{date_col}' starting at {start_ts}."
        )
        return df

    def _synthesize_dpgan(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        cond: Optional[Any] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes data using DPGAN (Differentially Private GAN) via Synthcity.

        Provides formal differential privacy guarantees during training.

        Args:
            data: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            target_col: Optional target column.
            custom_distributions: Optional class distributions for post-processing.
            **model_kwargs:
                - epochs (int): Training epochs (default: 300).
                - epsilon (float): Privacy budget — lower = more private (default: 1.0).
                - delta (float): Privacy delta parameter (default: 1e-5).
        """
        self.logger.info("Starting DPGAN synthesis via Synthcity...")
        self._patch_synthcity_encoder()
        model_kwargs = self._normalize_epoch_params(model_kwargs)

        syn = self._get_synthesizer("dpgan", **model_kwargs)
        _fit_kw = {"cond": cond} if cond is not None else {}
        try:
            syn.fit(data, **_fit_kw)
        except Exception as e:
            self.logger.error(f"DPGAN training failed: {e}")
            return None

        self.synthesizer = syn
        self.method = "dpgan"
        self.metadata = {"columns": data.columns.tolist()}

        gen_kwargs = {"count": n_samples}
        if cond is not None:
            gen_kwargs["cond"] = cond
        synth_df = syn.generate(**gen_kwargs, random_state=self.random_state).dataframe()

        if custom_distributions:
            synth_df = self._apply_postprocess_distribution(
                synth_df, custom_distributions, target_col, n_samples
            )
        return synth_df

    def _synthesize_pategan(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        cond: Optional[Any] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes data using PATE-GAN via Synthcity.

        Uses the PATE (Private Aggregation of Teachers' Ensembles) framework
        to provide differential privacy guarantees.

        Args:
            data: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            target_col: Optional target column.
            custom_distributions: Optional class distributions for post-processing.
            **model_kwargs:
                - epochs (int): Training epochs (default: 300).
                - epsilon (float): Privacy budget — lower = more private (default: 1.0).
                - delta (float): Privacy delta parameter (default: 1e-5).
                - teacher_iters (int): Number of teacher training iterations.
                - student_iters (int): Number of student training iterations.
        """
        self.logger.info("Starting PATE-GAN synthesis via Synthcity...")
        self._patch_synthcity_encoder()
        model_kwargs = self._normalize_epoch_params(model_kwargs)

        syn = self._get_synthesizer("pategan", **model_kwargs)
        _fit_kw = {"cond": cond} if cond is not None else {}
        try:
            syn.fit(data, **_fit_kw)
        except Exception as e:
            self.logger.error(f"PATE-GAN training failed: {e}")
            return None

        self.synthesizer = syn
        self.method = "pategan"
        self.metadata = {"columns": data.columns.tolist()}

        gen_kwargs = {"count": n_samples}
        if cond is not None:
            gen_kwargs["cond"] = cond
        synth_df = syn.generate(**gen_kwargs, random_state=self.random_state).dataframe()

        if custom_distributions:
            synth_df = self._apply_postprocess_distribution(
                synth_df, custom_distributions, target_col, n_samples
            )
        return synth_df

    def privatize(
        self,
        data: pd.DataFrame,
        epsilon: float = 1.0,
        delta: Optional[float] = None,
        numeric_sensitivity: float = 1.0,
        mechanism: str = "laplace",
        categorical_p: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Applies differential privacy mechanisms to an existing DataFrame.

        Numeric columns are perturbed using the Laplace or Gaussian mechanism.
        Categorical columns use Randomized Response.

        Args:
            data: Input DataFrame to privatize.
            epsilon (float): Privacy budget. Lower = more private (default: 1.0).
            delta (float): Required for Gaussian mechanism. Ignored for Laplace.
            numeric_sensitivity (float): Global sensitivity for numeric columns (default: 1.0).
            mechanism (str): 'laplace' or 'gaussian' for numeric columns (default: 'laplace').
            categorical_p (float): Probability of keeping the true category in randomized
                response. If None, derived from epsilon as p = e^epsilon / (e^epsilon + 1).

        Returns:
            DataFrame with privatized values.

        Example:
            >>> private_df = gen.privatize(df, epsilon=0.5)
            >>> private_df = gen.privatize(df, epsilon=1.0, mechanism='gaussian', delta=1e-5)
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if mechanism not in ("laplace", "gaussian"):
            raise ValueError("mechanism must be 'laplace' or 'gaussian'.")
        if mechanism == "gaussian" and delta is None:
            raise ValueError("delta is required for the Gaussian mechanism.")

        self.logger.info(
            f"Privatizing data with epsilon={epsilon}, mechanism={mechanism}..."
        )

        result = data.copy()
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
        _rng = np.random.default_rng(self.random_state)

        # --- Numeric columns ---
        for col in numeric_cols:
            if mechanism == "laplace":
                scale = numeric_sensitivity / epsilon
                noise = _rng.laplace(0, scale, size=len(data))
            else:
                # Gaussian mechanism
                sigma = numeric_sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                noise = _rng.normal(0, sigma, size=len(data))
            result[col] = data[col] + noise

        # --- Categorical columns: Randomized Response ---
        if categorical_p is None:
            exp_eps = np.exp(epsilon)
            p = exp_eps / (exp_eps + 1)
        else:
            p = categorical_p

        for col in categorical_cols:
            categories = data[col].dropna().unique().tolist()
            if len(categories) <= 1:
                continue

            # Vectorized randomized response: flip rows where random draw >= p
            col_arr = data[col].to_numpy()
            keep_mask = _rng.random(len(col_arr)) < p
            new_vals = col_arr.copy()
            flip_positions = np.where(~keep_mask)[0]
            if len(flip_positions) > 0:
                for uval in np.unique(col_arr[flip_positions]):
                    sub_pos = flip_positions[col_arr[flip_positions] == uval]
                    others = [c for c in categories if c != uval]
                    if others:
                        new_vals[sub_pos] = _rng.choice(others, size=len(sub_pos))
            result[col] = new_vals

        self.logger.info("Privatization complete.")
        return result

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
