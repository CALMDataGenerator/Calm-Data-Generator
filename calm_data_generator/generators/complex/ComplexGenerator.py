import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import eigh

from calm_data_generator.generators.base import BaseGenerator


class ComplexGenerator(BaseGenerator):
    """
    Abstract intermediate layer between BaseGenerator and domain generators.

    Provides three reusable mathematical engines:
      - _generate_correlated_module: Gaussian Copula (unconditional)
      - _generate_conditional_data: Gaussian Copula (conditional)
      - apply_stochastic_effects: stochastic per-entity effect framework

    Domain generators (Clinical, Finance, IoT, Insurance) should inherit from
    this class to reuse these engines without duplicating code.
    """

    def _generate_correlated_module(
        self,
        n_samples: int,
        marginals_list: list,
        sigma_module: np.ndarray,
    ) -> np.ndarray:
        """
        Generates correlated data for a module using Gaussian Copula.

        Args:
            n_samples (int): Number of samples to generate.
            marginals_list (list): List of marginal distributions (scipy frozen rv).
            sigma_module (np.ndarray): Correlation matrix for the module (n_vars x n_vars).

        Returns:
            np.ndarray: Generated data matrix (n_samples x n_vars).
        """
        n_mod_vars = len(marginals_list)

        # Ensure the matrix is Positive Semi-Definite (PSD)
        try:
            np.linalg.cholesky(sigma_module)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = eigh(sigma_module)
            eigvals[eigvals < 1e-6] = 1e-6
            sigma_module_psd = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)
            v = np.sqrt(np.diag(sigma_module_psd))
            sigma_module = sigma_module_psd / np.outer(v, v)

        # Copula algorithm
        mean_vec = np.zeros(n_mod_vars)
        Z_mod = np.random.multivariate_normal(
            mean=mean_vec, cov=sigma_module, size=n_samples
        )
        U_mod = np.clip(stats.norm.cdf(Z_mod), 1e-6, 1 - 1e-6)

        X_mod = np.zeros((n_samples, n_mod_vars))
        for i, marginal in enumerate(marginals_list):
            X_mod[:, i] = marginal.ppf(U_mod[:, i])

        return X_mod

    def _generate_conditional_data(
        self,
        n_samples: int,
        conditioning_data: np.ndarray,
        conditioning_marginals: list,
        target_marginals: list,
        full_covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Generates data for target variables conditioned on existing data using Gaussian Copula.

        Args:
            n_samples (int): Number of samples.
            conditioning_data (np.ndarray): Existing data matrix (n_samples x n_cond).
            conditioning_marginals (list): Marginals for conditioning variables.
            target_marginals (list): Marginals for target variables.
            full_covariance (np.ndarray): Full covariance matrix (n_cond + n_target) x (n_cond + n_target).

        Returns:
            np.ndarray: Generated target data (n_samples x n_target).
        """
        n_cond = len(conditioning_marginals)
        n_target = len(target_marginals)

        if conditioning_data.shape != (n_samples, n_cond):
            raise ValueError(
                f"conditioning_data shape {conditioning_data.shape} mismatch with "
                f"n_samples {n_samples} or n_cond {n_cond}."
            )

        if full_covariance.shape != (n_cond + n_target, n_cond + n_target):
            raise ValueError(
                f"full_covariance shape {full_covariance.shape} mismatch with "
                f"total variables {n_cond + n_target}."
            )

        # 1. Transform conditioning data to latent space Z_cond
        Z_cond = np.zeros_like(conditioning_data, dtype=float)
        for i, marginal in enumerate(conditioning_marginals):
            is_discrete = False
            if hasattr(marginal, "dist"):
                if hasattr(marginal.dist, "pmf") or marginal.dist.name in [
                    "binom",
                    "poisson",
                    "nbinom",
                    "randint",
                    "geom",
                    "hypergeom",
                    "logser",
                    "planck",
                    "boltzmann",
                    "zipf",
                    "dlaplace",
                    "skellam",
                ]:
                    is_discrete = True

            if is_discrete:
                # Randomized Quantile Residuals (Jittering)
                data = conditioning_data[:, i]
                u_high = marginal.cdf(data)
                u_low = marginal.cdf(data - 1)
                u_low = np.maximum(0, u_low)
                u_high = np.minimum(1, u_high)
                U = np.random.uniform(u_low, u_high)
            else:
                U = marginal.cdf(conditioning_data[:, i])

            U = np.clip(U, 1e-6, 1 - 1e-6)
            Z_cond[:, i] = stats.norm.ppf(U)

        # 2. Partition covariance matrix
        S_cc = full_covariance[:n_cond, :n_cond]
        S_ct = full_covariance[:n_cond, n_cond:]
        S_tc = full_covariance[n_cond:, :n_cond]
        S_tt = full_covariance[n_cond:, n_cond:]

        # 3. Compute conditional parameters
        try:
            S_cc_inv = np.linalg.inv(S_cc)
        except np.linalg.LinAlgError:
            S_cc_inv = np.linalg.inv(S_cc + np.eye(n_cond) * 1e-6)

        mu_cond = S_tc.dot(S_cc_inv).dot(Z_cond.T).T  # (n_samples, n_target)
        Sigma_cond = S_tt - S_tc.dot(S_cc_inv).dot(S_ct)

        # Ensure Sigma_cond is PSD
        try:
            np.linalg.cholesky(Sigma_cond)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = eigh(Sigma_cond)
            eigvals[eigvals < 1e-6] = 1e-6
            Sigma_cond = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)

        # 4. Sample Z_target from conditional multivariate normal
        Z_noise = np.random.multivariate_normal(
            mean=np.zeros(n_target), cov=Sigma_cond, size=n_samples
        )
        Z_target = mu_cond + Z_noise

        # 5. Transform Z_target to X_target using target marginals
        U_target = np.clip(stats.norm.cdf(Z_target), 1e-6, 1 - 1e-6)
        X_target = np.zeros((n_samples, n_target))
        for i, marginal in enumerate(target_marginals):
            X_target[:, i] = marginal.ppf(U_target[:, i])

        return X_target

    def apply_stochastic_effects(
        self,
        df: pd.DataFrame,
        entity_ids,
        effect_config: dict,
    ) -> None:
        """
        Applies a single stochastic effect to a subset of entities in-place.

        Supports 7 effect types: additive_shift, fold_change, power_transform,
        variance_scale, log_transform, polynomial_transform, sigmoid_transform.
        Also accepts simple_additive_shift as an alias for additive_shift.

        Args:
            df (pd.DataFrame): Data frame to modify in-place.
            entity_ids: Index labels of entities to apply the effect to.
            effect_config (dict): Effect configuration with keys:
                - index: column indices to affect
                - effect_type: one of the 7 supported types
                - effect_value: scalar, list [low, high], or dict (for sigmoid)
        """
        indices = effect_config["index"]
        effect_type = effect_config["effect_type"]
        effect_value = effect_config["effect_value"]
        cols_to_affect = df.columns[indices]
        n_entities = len(entity_ids)

        if n_entities == 0:
            return

        # simple_additive_shift is an alias for additive_shift (backward compat)
        if effect_type == "simple_additive_shift":
            effect_type = "additive_shift"

        if effect_type in ["additive_shift", "fold_change", "power_transform"]:
            if isinstance(effect_value, list) and len(effect_value) == 2:
                entity_offsets = np.random.uniform(
                    effect_value[0], effect_value[1], size=n_entities
                )
            else:
                scale = abs(effect_value) * 0.1 + 1e-6
                entity_offsets = np.random.normal(
                    loc=effect_value, scale=scale, size=n_entities
                )

            if effect_type == "additive_shift":
                df.loc[entity_ids, cols_to_affect] += entity_offsets[:, np.newaxis]
            elif effect_type == "fold_change":
                if np.any(entity_offsets <= 0):
                    entity_offsets[entity_offsets <= 0] = 1e-6
                df.loc[entity_ids, cols_to_affect] *= entity_offsets[:, np.newaxis]
            elif effect_type == "power_transform":
                df.loc[entity_ids, cols_to_affect] **= entity_offsets[:, np.newaxis]

        elif effect_type == "variance_scale":
            if isinstance(effect_value, list) and len(effect_value) == 2:
                scaling_factors = np.random.uniform(
                    effect_value[0], effect_value[1], size=len(cols_to_affect)
                )
            else:
                scaling_factors = effect_value

            X_mod = df.loc[entity_ids, cols_to_affect]
            mean = X_mod.mean(axis=0)
            std = X_mod.std(axis=0)
            Z = (X_mod - mean) / (std + 1e-8)
            X_new = Z * (std * scaling_factors) + mean
            df.loc[entity_ids, cols_to_affect] = X_new

        elif effect_type == "log_transform":
            epsilon = effect_value if isinstance(effect_value, (int, float)) else 1e-8
            df.loc[entity_ids, cols_to_affect] = np.log(
                df.loc[entity_ids, cols_to_affect] + epsilon
            )

        elif effect_type == "polynomial_transform":
            p = np.poly1d(effect_value)
            df.loc[entity_ids, cols_to_affect] = p(
                df.loc[entity_ids, cols_to_affect]
            )

        elif effect_type == "sigmoid_transform":
            X_mod = df.loc[entity_ids, cols_to_affect]
            k = effect_value.get("k", 1)
            x0 = effect_value.get("x0", X_mod.mean().mean())
            df.loc[entity_ids, cols_to_affect] = 1 / (
                1 + np.exp(-k * (X_mod - x0))
            )

        else:
            raise ValueError(f"Unsupported effect_type '{effect_type}'.")
