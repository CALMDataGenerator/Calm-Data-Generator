import logging

from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class ScenarioInjectorPreset(GeneratorPreset):
    """
    Directly leverages the ScenarioInjector to apply defined complex scenarios
    to an existing dataset, without necessarily generating new samples from scratch (unless desired).
    """

    def generate(self, data, scenario_config, **kwargs):
        """Apply a scenario configuration to existing data via the ScenarioInjector.

        Args:
            data: The dataset to transform.
            scenario_config: Scenario configuration (feature evolution / target construction).
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        injector = ScenarioInjector(seed=self.random_state)

        if self.verbose:
            logger.info("[ScenarioInjectorPreset] Applying scenario configuration...")

        return injector.evolve_features(
            df=data, evolution_config=scenario_config.get("evolve_features")
        )
