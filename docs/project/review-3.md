# 3.0 technical review and roadmap

This review covers the local source tree, public modules, tests, packaging, workflows, documentation and notebook inventory. It distinguishes fixes made for 3.0 from work that still needs validation. Passing tests do not establish that every numerical domain or optional platform is correct.

## What is already working well

The package has a clear differentiator: executable mathematical formulas with explanations and training inspection. The source is organized into focused modules rather than one monolithic runtime. The existing suite covers symbolic operations, autograd, model building, repositories, datasets, training events and optional bridges. Mkdocstrings keeps API reference signatures connected to the implementation.

The pre-change baseline on Python 3.12 was **489 passed, 6 skipped**. A separate screening of all 285 repository formulas on ordinary array inputs completed without exceptions. Previously reported fixes for optimizer dispatch, scheduler stepping, LayerNorm gradients, macro F1, transpose axes, residual checkpoint predictions and even-kernel padding were reproduced successfully.

## Findings fixed for 3.0

| Priority | Finding | Result |
|---|---|---|
| High | Large PINNs accessed nonexistent `_rng` during residual-gradient sampling | All parameter residual gradients are now evaluated, with no silent subsampling |
| High | Multi-output PINN gradients did not match mean reduction | Finite-difference regression checks verify the normalization |
| High | Failed residual evaluation could leave a parameter perturbed | Restoration uses `finally`; callback exceptions retain their meaning |
| High | Checkpoints discarded built-in optimizer state | Adam, SGD and RMSprop continuation is checked against uninterrupted updates |
| High | Saturated categorical loss disagreed with the intended objective; softmax used an underflow heuristic | Stable logits loss and explicit fused gradients; general softmax VJP remains mathematically correct |
| Medium | Fractional labels and invalid probability targets were accepted | Explicit target validation |
| Medium | Analytical expression derivatives were rebuilt each training step | Cached derivative expressions use current parameter values |
| Medium | C compilation forced AVX2 without verifying CPU support | Portable compilation default and compiler-aware cache key |
| Medium | Homepage dataset unpacking and setup claims were misleading | Executable onboarding, source-derived catalogs and explicit backend boundaries |
| Medium | API cards linked directly to nonexistent rendered `.md` URLs | Markdown links and built-site link/anchor checks |
| Medium | Upload workflow had no tests or documentation gate | Version, tests, documentation, package metadata and wheel smoke checks precede publication |

## Remaining engineering priorities

| Priority | Evidence / constraint | Recommended next change |
|---|---|---|
| High | Tensor backward closures can observe an input mutated after the forward pass | Add storage version counters or immutable saved operands; test every differentiable operation |
| High | Native CUDA lifecycle and optional bridge paths are not fully exercised by local CPU tests | Dedicated GPU/driver CI and framework numerical parity tests; validate NHWC/NCHW flatten order and unsupported layer handling |
| Medium | Several expanded data generators reseed the process-wide NumPy RNG | Convert each generator to a local `default_rng` and add provenance metadata to returned data |
| Medium | C cache publication is not coordinated across simultaneous processes | Atomic temporary builds, locking and cache lifecycle management |
| Medium | Checkpoints use pickle and omit scheduler/data-order/RNG state | Design a versioned array/JSON format and an explicit full experiment checkpoint |
| Medium | PINN derivatives require many residual evaluations and are sensitive to finite-difference step size | Add analytic/autograd residual differentiation, error estimates and convergence benchmarks |
| Medium | Framework extras are large and platform compatibility varies | Add separate `torch`, `tensorflow` and `jax` extras with integration-specific compatibility checks |
| Medium | Observability can scan arrays and collect many events | Benchmark retention/sampling policies and expose predictable memory/overhead budgets |

## Performance strategy

Moving Python syntax into C++ alone will not make the framework broadly faster than established systems. Prioritize the bottlenecks shown by profiles: Python dispatch, repeated symbolic setup, convolution loops, memory movement, optimizer work and telemetry. Derivative reuse is implemented in 3.0; a device-resident tensor backend is a larger project.

Establish a benchmark suite for expression evaluation, small dense training, convolution, checkpointing and Observatory overhead. Validate numerical parity, report warm/cold timings and peak memory, and run on documented hardware. Then target a defensible niche such as transparent, fused symbolic workloads. The [performance guide](../advanced/performance.md) outlines the sequence.

## Features that fit the project

1. A gradient-check utility that compares layer/autograd gradients with finite differences and explains discrepancies.
2. Dataset provenance objects with original-versus-generated status and reproducible split metadata.
3. A reproducible experiment bundle containing model, optimizer, scheduler, random state, dataset hashes and Observatory summary.
4. A documentation playground for formula exploration, with bounded examples and explicit supported domains.
5. A backend capability query that reports supported operations and explains fallback decisions.

Prioritize correctness and trustworthy examples before expanding layer count. This preserves NeuroGebra's strongest reason for adoption: understanding the mathematics while running the experiment.

## Local validation results

Validation performed on November 25, 2026 using Python 3.12.7:

| Check | Result |
|---|---|
| Full runtime suite | 509 passed, 8 skipped; one existing complex-to-real cast warning in the statistics smoke test |
| Native and optional skips | Two native C checks, two CUDA checks and four framework runtime checks; their required execution environments were unavailable |
| Python compatibility syntax | All 65 runtime modules parse with Python 3.9 grammar; this does not substitute for running the cross-platform CI matrix |
| Documentation | Strict MkDocs build succeeds across 69 Markdown pages |
| Python documentation syntax | 357 code blocks parse successfully |
| Main example execution | 20 code blocks across seven onboarding/science pages execute successfully |
| Built-site links | Over 38,000 local links/assets/anchors checked without missing targets |
| Catalog generation | Regeneration reproduces the checked-in 285 formulas and 48 dataset loaders |
| Focused lint | Undefined names, import ordering and syntax checks pass for changed training/backend files and validation scripts |
| Distribution metadata | Wheel and source archive build successfully and pass `twine check` |
| Source archive contents | Includes MkDocs configuration, theme override, documentation and validation scripts |

The installed wheel is also checked outside the source checkout. Hardware performance, optional framework parity and the complete OS/Python matrix require their configured environments. A browser screenshot could not be captured in this execution environment; preview the site with `python -m mkdocs serve` to review desktop/mobile spacing and interactive controls before publication.

Changes remain local. No push, tag creation or hosted publication was performed.
