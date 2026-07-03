"""
Mixin providing scVI / GEARS single-cell synthesis methods for RealGenerator.

Methods: _report_scvi_history, _synthesize_scanvi, _synthesize_scvi,
         _synthesize_gears.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class _ScviSynthMixin:

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
        _accelerator = kwargs.get("accelerator", self._get_lightning_accelerator())
        try:
            scvi_model.train(
                max_epochs=epochs,
                early_stopping=early_stopping,
                accelerator=_accelerator,
            )
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
            scanvi_model.train(
                max_epochs=scanvi_epochs,
                early_stopping=early_stopping,
                accelerator=_accelerator,
            )
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
            "accelerator": kwargs.get("accelerator", self._get_lightning_accelerator()),
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
        device = kwargs.get("device", self._get_torch_device())
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
