# NeuroGebra 3.0.0

Make the mathematics visible: a complete documentation refresh, interactive visual formula exploration, and focused fixes for reliable training and scientific experiments.

## Highlights & Interactive Web Experience

- **Interactive Documentation**: Interactive formula explorer, activation visualization, and loss progression demo right on the home page with full accessibility and keyboard navigation support.
- **Granular API Documentation**: Dedicated reference guides for `Expression`, `MathForge`, `Value`, `Tensor`, `Trainer`, and `NeuroCraft` with complete signatures, parameters, and executable examples.
- **Classic Light/Dark Design**: Tailored theme with teal accents, warm paper backgrounds, serif headings, responsive navigation, and local system fonts.
- **Full Reference Catalogs**: Generated catalogs covering all 285 symbolic formulas and 48 dataset loaders, including mathematical formulation, provenance, and return conventions.
- **Frontend Quality Assurance**: Automated browser-less component test suite (`node --test tests/web/home-demo.test.cjs`) and site data generator validating graph bounds, ARIA states, and UI/Python agreement.
- **Updated Guides**: Comprehensive onboarding, architecture, capability matrix, troubleshooting, migration guide, and contributor documentation.

## Runtime & Engine Fixes

- **Optimizer State Persistence**: Preserve built-in Adam, SGD, and RMSprop configuration, step counters, and momentum/velocity buffers in model checkpoints.
- **Numerically Stable Softmax / Cross-Entropy**: Compute Dense softmax cross-entropy loss directly from logits, with exact fused gradients for class indices and probability targets.
- **Categorical Target Validation**: Validate integer categorical labels and probability distributions; preserve true general softmax vector-Jacobian product.
- **PINN Residual Derivatives**: Differentiate every residual weight and bias (supporting matrices with >256 entries), normalize multi-output gradients across all elements, and safely restore parameters upon callback exceptions.
- **Analytical Derivative Caching**: Cache and reuse analytical derivative expressions and compiled numerical functions across training epochs.
- **Portable Native Acceleration**: Safe C compiler target detection, compiler-aware cache hashing, and configurable cache root via `NEUROGEBRA_CACHE_DIR`.
- **Checkpointing Hygiene**: Respect `metrics=[]` and explicitly reject unsupported custom optimizer checkpoint serialization.

## Release & Verification Process

- Version consistency verified across Git tag, runtime `__version__`, and package metadata.
- 509 test cases passing, 8 optional/environment checks skipped cleanly.
- Strict MkDocs build passing with zero warnings across 75+ documentation pages.
- Over 16,500 local links, anchors, and 358 Python code snippets verified.
- Python 3.9+ remains fully supported.

See the [3.0 migration guide](https://neurogebra.readthedocs.io/en/latest/releases/migration-3/) for numerical changes and checkpoint compatibility.
