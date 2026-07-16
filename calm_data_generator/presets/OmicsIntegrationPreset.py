import logging

from calm_data_generator.generators.clinical import ClinicalDataGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class OmicsIntegrationPreset(GeneratorPreset):
    """
    Generates multi-omics data (Clinical + Gene Expression + Proteomics)
    with high correlation integrity between layers.
    """

    def generate(self, n_samples, n_genes=100, n_proteins=50, **kwargs):
        """Generate integrated multi-omics clinical data (gene + protein expression layers).

        Args:
            n_samples: Number of samples to generate.
            n_genes (int): Number of gene-expression features (default 100).
            n_proteins (int): Number of protein-expression features (default 50).
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame or Dict: The integrated multi-omics dataset(s).
        """
        gen = ClinicalDataGenerator(
            auto_report=kwargs.pop("auto_report", True), seed=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[OmicsIntegrationPreset] Simulating multi-omics data (Clinical + {n_genes} Genes + {n_proteins} Proteins)..."
            )

        # Forces generation of all layers
        return gen.generate(n_samples=n_samples, n_genes=n_genes, n_proteins=n_proteins)
