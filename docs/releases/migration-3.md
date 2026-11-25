# Migrate from 2.x to 3.0

Most core imports remain the same. This major release deliberately changes some numerical edge cases and native compilation defaults. Re-run your experiment's correctness checks before comparing results across versions.

| Area | 2.x behavior | 3.0 behavior / action |
|---|---|---|
| Dense softmax + categorical cross-entropy | Clipped probability loss could report about 27.63 for a wrong-class logit gap of 2000 | Stable logits loss reports 2000; gradients follow the actual objective. Expect different loss curves at saturation |
| Sparse labels | Fractional values could be truncated to integers | Invalid/fractional labels raise `ValueError`; provide integer class IDs |
| Probability targets | Invalid distributions could be accepted | Provide finite nonnegative rows summing to one |
| PINN residual gradients | Larger weight matrices could crash; gradients were silently sampled | Every parameter is differentiated. Larger models may take longer but no weights are silently omitted |
| Multi-output PINNs | Data gradients were over-scaled by output width | Mean-squared-error gradients normalize over every output element |
| Residual callbacks | An internal `TypeError` could trigger a second call with a different signature | Internal exceptions propagate, and perturbed parameters are restored |
| Model checkpoints | Architecture/weights loaded with fresh optimizer buffers | New checkpoints restore built-in optimizer configuration, step and buffers |
| Old checkpoints | No saved optimizer snapshot | Valid older checkpoints load with fresh optimizer state; files that never saved complete layer weights cannot reconstruct the missing parameters |
| Custom optimizers | State could be silently lost when saving | `save()` rejects unsupported optimizer types explicitly |
| Metric selection | `metrics=[]` became the default accuracy list | Empty selection remains empty |
| Native C target | `-mavx2` was forced on many x86 systems | Portable compiler target; rebenchmark native kernels on your machine |
| C options and variables | Arbitrary optimization string / incomplete variable list | Supported `-O*` values and complete unique variables are validated |

## Install from this checkout

```bash
python -m pip install -e .
python -c "import neurogebra; print(neurogebra.__version__)"
```

After publication, pin a deployment with `python -m pip install "neurogebra==3.0.0"`.

## Check a model checkpoint

```python
import numpy as np
from neurogebra import Model

# Load only your own or otherwise trusted pickle checkpoint.
restored = Model.load("model.pkl")
# Compare against predictions recorded before the upgrade.
```

Scheduler state, data order and RNG state are not serialized. Capture those separately for a reproducible experiment. Model bridges need their own prediction/gradient checks after conversion.

Read the [release notes](3.0.0.md) for the full scope and [capabilities](../project/capabilities.md) for the remaining boundaries.
