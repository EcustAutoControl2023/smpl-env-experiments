# collision-risk-patch

Utility package providing reusable collision risk estimation tooling used by the experiments.

## Usage in Experiments

The risk model is wired into the Hydra configuration that drives
`experiments/JPC/run_experiments.py`. Enable it by toggling the
`experiment.risk_model` options when launching a run. For example, to train
with a saved Gaussian-process checkpoint located at
`artifacts/collision_gp.joblib` while appending the predicted risk to the
offline dataset, run:

```bash
uv run python -m experiments.JPC.run_experiments \
  experiment.risk_model.enabled=true \
  experiment.risk_model.model_path=artifacts/collision_gp.joblib \
  experiment.risk_model.default_risk=0.05 \
  experiment.risk_model.clip_min=0.0 \
  experiment.risk_model.clip_max=1.0 \
  experiment.risk_model.augment_dataset=true
```

Key flags:

- `experiment.risk_model.enabled` – activates the observation wrapper that
  queries the predictor each environment step.
- `experiment.risk_model.model_path` – filesystem path to the serialized model
  created with the utilities in this patch.
- `experiment.risk_model.default_risk` – fallback value if the model fails to
  produce a prediction (e.g., due to missing history).
- `experiment.risk_model.clip_min` / `clip_max` – bounds applied to keep risk
  scores within a desired range before they are concatenated to the
  observation vector.
- `experiment.risk_model.augment_dataset` – when set, the loader enriches the
  offline replay buffer with risk predictions prior to training.

Hydra allows additional overrides (such as `experiment.environment.name` or
`experiment.exp_name`) to be appended on the same command line if you need to
customize other aspects of the run.
