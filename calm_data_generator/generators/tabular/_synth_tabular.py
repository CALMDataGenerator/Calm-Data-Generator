"""
Mixin providing tabular synthesis methods for RealGenerator.

Methods: _synthesize_ctgan, _synthesize_great, _synthesize_tvae,
         _synthesize_conditional_drift, _synthesize_rtvae, _synthesize_bn,
         _synthesize_windowed_copula, _synthesize_copula, _synthesize_resample,
         _synthesize_smote, _synthesize_adasyn, _synthesize_ddpm,
         _synthesize_kde, _synthesize_gmm, _synthesize_cart,
         _synthesize_xgboost, _synthesize_hmm, _synthesize_rf, _synthesize_lgbm.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class _TabularSynthMixin:

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
