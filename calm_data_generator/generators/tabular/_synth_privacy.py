"""
Mixin providing privacy-preserving synthesis methods for RealGenerator.

Methods: _synthesize_dpgan, _synthesize_pategan, privatize.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class _PrivacySynthMixin:

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
