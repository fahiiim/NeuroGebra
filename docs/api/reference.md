# API reference

Choose a subsystem below. Reference pages are generated from the Python implementation with mkdocstrings; tutorial pages provide context and runnable examples.

| Subsystem | Main entry points | Reference |
|---|---|---|
| Symbolic mathematics | `Expression`, `MathForge`, `NeuroCraft`, `Trainer` | [Core](core.md) |
| Eager differentiation | `Value`, `Tensor` | [Core](core.md) |
| Neural models | `ModelBuilder`, `Model`, `Layer` | [Builders](builders.md) |
| Native expression kernels | `compile_c`, `compile_cuda`, `get_acceleration_info` | [Acceleration](acceleration.md) |
| Scientific experiments | `PINN` | [Science](science.md) |
| Parameter updates and schedules | `SGD`, `Adam`, `RMSprop`, `StepLR`, `CosineAnnealingLR`, `ExponentialLR`, `ReduceLROnPlateau` | [Optimizers](optim.md) |
| Evaluation | Classification and regression functions | [Metrics](metrics.md) |
| Training inspection | `TrainingLogger`, `LogConfig`, adaptive logging, exports, storage, fingerprints | [Logging and training](logging.md) |
| Interoperability and data | Framework adapters, `Datasets`, `ExpandedDatasets`, `CombinedDatasets` | [Bridges and datasets](bridges-datasets.md) |
| Formula definitions and plotting | Repository, search, static/interactive visualization | [Repository and visualization](repository.md) |
| Learning utilities | `TutorialSystem`, explanation and helper functions | [Learning utilities](utilities.md) |

## Quick imports

```python
from neurogebra import Expression, MathForge, Layer, Model, ModelBuilder, PINN
from neurogebra.core.autograd import Value, Tensor
from neurogebra.core.trainer import Trainer
from neurogebra.datasets import Datasets, CombinedDatasets
from neurogebra.optim import Adam, SGD, RMSprop, StepLR
from neurogebra import metrics
```

Browse all [285 formulas](../reference/formulas.md), the [48 dataset loaders](../reference/datasets.md), and the [capability boundaries](../project/capabilities.md).
