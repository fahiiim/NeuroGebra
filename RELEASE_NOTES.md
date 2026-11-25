# NeuroGebra 3.0.0

Make the mathematics visible: a complete documentation refresh and focused fixes for reliable training and scientific experiments.

## Documentation

- Classic light/dark design with teal accents, warm paper backgrounds, serif headings and responsive navigation.
- Complete generated catalogs of 285 symbolic formulas and 48 dataset loaders, including provenance and return conventions.
- Updated onboarding, architecture, native acceleration and performance guides.
- New capability matrix, troubleshooting, contributor reference, migration guide and technical review.
- Corrected dataset unpacking, backend installation claims and rendered API links; configured MathJax for the documentation's equation wrappers.

## Runtime fixes

- Preserve built-in Adam, SGD and RMSprop configuration, step counters and buffers in model checkpoints.
- Compute final Dense softmax/cross-entropy loss from logits, with exact fused gradients for class IDs and probability targets.
- Validate categorical labels and preserve the general softmax vector-Jacobian product.
- Fix PINN crashes on weight matrices larger than 256 entries, differentiate every residual parameter and normalize multi-output data gradients correctly.
- Restore finite-difference perturbations after callback errors; support both residual callback signatures without masking internal exceptions.
- Reuse analytical derivative expressions and compiled callables during expression training.
- Use portable C compiler targets, compiler-aware cache keys and an optional writable cache root.
- Respect `metrics=[]` and explicitly reject unsupported custom optimizer checkpointing.

## Release process

- Verify tag, package and runtime versions before publication.
- Gate uploads on tests, strict documentation/link checks, distribution metadata and an installed-wheel smoke test.
- Include MkDocs configuration, theme overrides and validation scripts in the source archive.

Python 3.9+ remains supported. See the [3.0 migration guide](https://neurogebra.readthedocs.io/en/latest/releases/migration-3/) for numerical changes and checkpoint compatibility. Neural model and PINN training remain NumPy based; native expression acceleration does not provide a universal speed advantage over other frameworks.

These notes are prepared for the maintainer to publish as GitHub release **v3.0.0**. See `PUBLISHING.md` for the release sequence.
