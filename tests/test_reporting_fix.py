import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.tabular import RealGenerator


def test_reporting_generation():
    """Explicitly test report generation with new dependencies."""
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.choice(["X", "Y"], 100),
            "C": np.random.randint(0, 100, 100),
        }
    )

    gen = RealGenerator(auto_report=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = gen.generate(df, n_samples=10, method="cart", target_col="B", output_dir=tmpdir)
        assert result is not None, "generate() returned None"
        assert len(result) == 10


if __name__ == "__main__":
    test_reporting_generation()
