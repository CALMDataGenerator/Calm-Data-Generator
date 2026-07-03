"""Omics distribution-parameter design mixin for ClinicalDataGenerator.

Note: none of these three methods currently have any call sites in the
codebase (confirmed dead code as of the 2026-07 refactor) — kept here as-is
per the "pure move, zero behavior change" refactor policy.
"""

import numpy as np


class _OmicsParamsMixin:
    """Mixin providing distribution-parameter design helpers for ClinicalDataGenerator."""

    def _design_gene_params_rnaseq(self, n_genes, gene_mean_log_center):
        """
        Designs r (size) and p (prob) parameters for Negative Binomial distribution (RNA-Seq).
        """
        log_means = np.random.normal(loc=gene_mean_log_center, scale=1.5, size=n_genes)
        means = np.round(np.exp(log_means))

        dispersions = np.random.uniform(low=0.1, high=1.0, size=n_genes)
        sizes = 1 / dispersions

        probs = sizes / (sizes + means)

        valid_mask = (probs > 0) & (probs < 1)

        return sizes, probs, valid_mask

    def _design_protein_params(self, n_proteins):
        """
        Designs log_means and log_stds parameters for Log-Normal distribution (Proteins).
        """
        log_means = np.random.normal(loc=3.0, scale=0.8, size=n_proteins)
        log_stds = np.random.uniform(low=0.1, high=0.4, size=n_proteins)
        return log_stds, log_means  # Return log_stds and log_means, not log_scales

    def _design_gene_params_normal(self, n_genes, gene_mean_loc_center):
        """
        Designs loc (mean) and scale (std dev) parameters for Normal distribution (Microarray).
        """
        locs = np.random.normal(loc=gene_mean_loc_center, scale=1.0, size=n_genes)
        scales = np.random.uniform(low=0.5, high=2.0, size=n_genes)
        return locs, scales
