import logging

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    # Redefine properly for picklability
    class SimpleDenoiser(nn.Module):
        def __init__(self, dim, steps=1000):
            super().__init__()
            self.steps = steps
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, dim),
            )

        def forward(self, x, t):
            t_emb = t.unsqueeze(-1) / self.steps
            return self.net(torch.cat([x, t_emb], dim=-1))

except ImportError:

    class SimpleDenoiser:
        pass


try:
    import numpy as np
    import pandas as pd

    class FCSModel:
        """
        A persistent model container for Fully Conditional Specification (FCS) methods.
        Stores conditional models for each feature and marginal distributions for initialization.
        """

        def __init__(self, models, marginals, encoding_info, visit_order, random_state=None):
            self.models = models  # Dict[col, model]
            self.marginals = marginals  # Dict[col, values_to_sample]
            self.encoding_info = (
                encoding_info  # Dict[col, categories] for reconstruction
            )
            self.visit_order = visit_order  # List[col]
            self.rng = np.random.default_rng(random_state)

        def generate(self, n_samples):
            # 1. Initialize from marginals
            synth_data = {}
            for col, values in self.marginals.items():
                synth_data[col] = self.rng.choice(values, size=n_samples, replace=True)

            X_synth = pd.DataFrame(synth_data)

            # Ensure categories use correct dtype if present in encoding_info
            if self.encoding_info:
                for col, categories in self.encoding_info.items():
                    if col in X_synth.columns:
                        X_synth[col] = pd.Categorical(
                            X_synth[col], categories=categories
                        )

            # 2. Iterate once (Gibbs step) to apply conditional models
            for col in self.visit_order:
                if col not in self.models:
                    continue

                model = self.models[col]
                Xs = X_synth.drop(columns=col)

                # Apply encoding if model is not LGBM (heuristic)
                # We need to check if the model object expects encoded inputs.
                # In _synthesize_fcs_generic, we checked "LGBM" in class name.
                is_native_cat = any(name in model.__class__.__name__ for name in ("LGBM", "XGB"))

                if not is_native_cat:
                    cat_enc_cols = Xs.select_dtypes(include=["category"]).columns
                    updates = {}
                    for c in cat_enc_cols:
                        aligned = (
                            pd.Categorical(Xs[c], categories=self.encoding_info[c])
                            if c in self.encoding_info
                            else Xs[c].astype("category")
                        )
                        updates[c] = aligned.codes
                    Xs_encoded = Xs.assign(**updates) if updates else Xs
                else:
                    Xs_encoded = Xs

                # Predict
                new_vals = None

                # Check for probabilistic sampling
                if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                    try:
                        probs = model.predict_proba(Xs_encoded)
                        classes = model.classes_
                        # Probabilistic sampling
                        # Optimization: If binary, use vectorization
                        if len(classes) == 2:
                            p1 = probs[:, 1]
                            draws = self.rng.random(n_samples)
                            preds_idx = (draws < p1).astype(int)
                            new_vals = classes[preds_idx]
                        else:
                            # Vectorized multiclass sampling via cumsum + searchsorted
                            probs_norm = probs / probs.sum(axis=1, keepdims=True)
                            cum = np.cumsum(probs_norm, axis=1)
                            draws = self.rng.random(len(cum))
                            preds_idx = (draws[:, None] > cum).sum(axis=1).clip(0, len(classes) - 1)
                            new_vals = classes[preds_idx]
                    except Exception as e:
                        _logger.warning(f"predict_proba failed; falling back to predict(). Reason: {e}")
                        # Fallback to direct prediction if proba fails
                        new_vals = model.predict(Xs_encoded)
                else:
                    new_vals = model.predict(Xs_encoded)

                # Restore categorical type if needed
                if col in self.encoding_info:
                    new_vals = pd.Categorical(
                        new_vals, categories=self.encoding_info[col]
                    )

                X_synth[col] = new_vals

            return X_synth

except ImportError:

    class FCSModel:
        pass
