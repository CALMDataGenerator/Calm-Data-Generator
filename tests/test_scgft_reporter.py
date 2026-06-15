import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import ReportConfig
from calm_data_generator.reports.QualityReporter import QualityReporter


def _make_dataframes():
    genes = [f"Gene_{i}" for i in range(50)]
    real_df = pd.DataFrame(
        np.random.poisson(lam=1.0, size=(100, 50)), columns=genes
    )
    real_df["cell_type"] = np.random.choice(["TypeA", "TypeB"], size=100)
    synth_df = pd.DataFrame(
        np.random.poisson(lam=0.9, size=(100, 50)), columns=genes
    )
    synth_df["cell_type"] = np.random.choice(["TypeA", "TypeB"], size=100)
    return real_df, synth_df


def _mock_scgft_modules(run_all_side_effect=None):
    """Build mocked anndata, scanpy, scgft_evaluator modules."""
    mock_ad = MagicMock()
    mock_sc = MagicMock()
    mock_scgft = MagicMock()

    mock_adata = MagicMock()
    mock_adata.obs.__getitem__.return_value.unique.return_value.tolist.return_value = [
        "TypeA", "TypeB"
    ]
    mock_ad.AnnData.return_value = mock_adata

    if run_all_side_effect is not None:
        mock_scgft.ScGFT_Evaluator.run_all.side_effect = run_all_side_effect

    return mock_ad, mock_sc, mock_scgft


class TestScGFTReporter(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name
        self.real_df, self.synth_df = _make_dataframes()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_scgft_integration(self):
        def mock_run_all(a1, a2, genes_top, col_grupo, grupo_a, grupo_b, **kwargs):
            return pd.DataFrame([{"ARI_TypeA": 0.85}])

        mock_ad, mock_sc, mock_scgft = _mock_scgft_modules(mock_run_all)

        with patch.dict(sys.modules, {
            "anndata": mock_ad,
            "scanpy": mock_sc,
            "scgft_evaluator": mock_scgft,
        }):
            reporter = QualityReporter(verbose=True)
            reporter._run_scgft_evaluation(
                real_df=self.real_df,
                synthetic_df=self.synth_df,
                output_dir=self.output_dir,
                target_col="cell_type",
            )

        mock_scgft.ScGFT_Evaluator.run_all.assert_called_once()

        report_file = os.path.join(self.output_dir, "scgft_report.html")
        self.assertTrue(os.path.exists(report_file), "scGFT HTML report was not created")

        with open(report_file) as f:
            content = f.read()
        self.assertIn("scGFT Single-Cell Evaluation Report", content)

    def test_scgft_not_available_graceful_exit(self):
        # Setting a sys.modules entry to None causes ImportError on import
        with patch.dict(sys.modules, {
            "anndata": None,
            "scanpy": None,
            "scgft_evaluator": None,
        }):
            reporter = QualityReporter(verbose=True)
            reporter._run_scgft_evaluation(
                real_df=self.real_df,
                synthetic_df=self.synth_df,
                output_dir=self.output_dir,
                target_col="cell_type",
            )

        report_file = os.path.join(self.output_dir, "scgft_report.html")
        self.assertFalse(os.path.exists(report_file))

    def test_scgft_run_all_signature(self):
        call_kwargs = {}

        def capture_run_all(a1, a2, genes_top, col_grupo, grupo_a, grupo_b, **kwargs):
            call_kwargs.update({
                "genes_top": genes_top,
                "col_grupo": col_grupo,
                "grupo_a": grupo_a,
                "grupo_b": grupo_b,
            })
            return pd.DataFrame([{"ARI_TypeA": 0.5}])

        mock_ad, mock_sc, mock_scgft = _mock_scgft_modules(capture_run_all)

        with patch.dict(sys.modules, {
            "anndata": mock_ad,
            "scanpy": mock_sc,
            "scgft_evaluator": mock_scgft,
        }):
            reporter = QualityReporter(verbose=False)
            reporter._run_scgft_evaluation(
                real_df=self.real_df,
                synthetic_df=self.synth_df,
                output_dir=self.output_dir,
                target_col="cell_type",
            )

        self.assertEqual(call_kwargs["col_grupo"], "cell_type")
        self.assertIn(call_kwargs["grupo_a"], ["TypeA", "TypeB"])
        self.assertIn(call_kwargs["grupo_b"], ["TypeA", "TypeB"])
        self.assertIsInstance(call_kwargs["genes_top"], list)


if __name__ == "__main__":
    unittest.main()
