"""
Mixin providing synthesis utility methods for RealGenerator.

Methods: _get_model_params, _validate_method, _get_synthesizer,
         _normalize_epoch_params, _concat_and_shuffle, _safe_to_dense,
         _build_cond_tensor, _validate_custom_distributions,
         _patch_synthcity_encoder, _synthesize_split_by_class,
         _apply_postprocess_distribution, _apply_resampling_strategy,
         _inject_dates, _synthesize_fcs_generic.
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class _SynthUtilsMixin:

    @staticmethod
    def _get_torch_device() -> str:
        """Returns 'cuda', 'mps', or 'cpu' — best device available."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _get_lightning_accelerator() -> str:
        """Returns the Lightning/scVI accelerator string for the best available device."""
        import torch
        if torch.cuda.is_available():
            return "gpu"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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

    def _inject_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        date_start: Optional[str],
        date_every: int,
        date_step: Optional[Dict],
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
        from calm_data_generator.generators.persistence_models import FCSModel

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
