# 20 Executed GPU Portfolio Notebooks

Explore 20 end-to-end, fully executed Jupyter notebooks under `examples/projects/` showcasing every subsystem of NeuroGebra—from deep vision, GANs, Diffusion, and Language Modeling to bare-metal CUDA JIT, Physics-Informed Neural Networks, and Rich LaTeX Symbolic Autograd.

Every notebook uses self-contained NeuroGebra data loaders (`neurogebra.datasets`) or analytical domain generators and preserves stored stdout metrics and embedded Matplotlib visualizations.

---

## Browse by Track

=== "Vision, Generative AI & NLP (01&ndash;10)"

    | # | Project Notebook | Domain | Key NeuroGebra APIs | Dataset / Problem | Links |
    | :-: | :--- | :--- | :--- | :--- | :--- |
    | **01** | Fashion-MNIST Image Classification | Vision | `ModelBuilder`, `Dense`, `Dropout`, `categorical_crossentropy` | Fashion-MNIST | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/01-image-classification-fmnist-cnn.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/01-image-classification-fmnist-cnn.ipynb) |
    | **02** | Fashion-MNIST with Observatory Pro | Telemetry | `AdaptiveLogger`, `AutoHealthWarnings`, `TieredStorage`, `DashboardExporter` | Fashion-MNIST | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/02-image-classification-observatory-pro.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/02-image-classification-observatory-pro.ipynb) |
    | **03** | MNIST GAN Baseline | Generative AI | Dual `ModelBuilder` ($G$ &amp; $D$), min-max loop | MNIST Digits | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/03-gan-mnist-baseline.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/03-gan-mnist-baseline.ipynb) |
    | **04** | GAN Instability & Mode Collapse Diagnostics | Diagnostics | Feature variance monitoring, diversity index, collapse alerts | MNIST Digits | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/04-gan-mode-collapse-diagnostics.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/04-gan-mode-collapse-diagnostics.ipynb) |
    | **05** | Diffusion Image Denoising Baseline | Diffusion | Forward Gaussian schedule, reverse denoising network | MNIST Digits | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/05-diffusion-image-denoising-baseline.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/05-diffusion-image-denoising-baseline.ipynb) |
    | **06** | Diffusion Math & Scheduler Deep-Dive | Mathematical Physics | `MathForge`, Linear vs Cosine $\bar{\alpha}_t$ SNR schedules | Analytical Diffusion | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/06-diffusion-math-scheduler-deep-dive.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/06-diffusion-math-scheduler-deep-dive.ipynb) |
    | **07** | Core NLP Spam Classification | Text NLP | `Datasets.load_spam`, dense classifier, `confusion_matrix`, `f1_score` | Spambase Benchmark | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/07-core-nlp-spam-classifier.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/07-core-nlp-spam-classifier.ipynb) |
    | **08** | Core NLP Sentiment Classifier | Text Embeddings | Token embedding projection, `BatchNorm`, multi-class softmax | Sentiment Tokens | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/08-core-nlp-sentiment-classifier.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/08-core-nlp-sentiment-classifier.ipynb) |
    | **09** | Small Language Model (CharLM) from Scratch | Language Modeling | Character tokenizer, autoregressive sequence model | Shakespeare Corpus | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/09-small-language-model-charlm-from-scratch.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/09-small-language-model-charlm-from-scratch.ipynb) |
    | **10** | Character LM Sampling & Tuning | Generation Algorithms | Temperature scaling ($T=0.3, 1.0, 2.5$), Top-$k$, Top-$p$ decoding | Language Vocab | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/10-small-language-model-charlm-tuning.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/10-small-language-model-charlm-tuning.ipynb) |

=== "Acceleration, SciML, Modern Architectures & Enterprise (11&ndash;20)"

    | # | Project Notebook | Domain | Key NeuroGebra APIs | Dataset / Problem | Links |
    | :-: | :--- | :--- | :--- | :--- | :--- |
    | **11** | Hardware Acceleration (C-JIT AVX2 &amp; CUDA) | HPC Acceleration | `compile_c`, `compile_cuda`, `get_acceleration_info` | Vectorized Benchmarks | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/11-hardware-acceleration-c-and-cuda-jit.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/11-hardware-acceleration-c-and-cuda-jit.ipynb) |
    | **12** | Physics-Informed Neural Networks (PINN) | Scientific ML &amp; PDEs | `PINN`, `harmonic_oscillator_residual`, `burgers_residual` | ODE &amp; 1D Burgers' PDE | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/12-physics-informed-neural-networks-pinn.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/12-physics-informed-neural-networks-pinn.ipynb) |
    | **13** | Symbolic Autograd &amp; Higher-Order Derivatives | Symbolic Math &amp; LaTeX | `Expression._repr_latex_`, `gradient()`, Hessian matrix | Himmelblau Surface | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/13-symbolic-autograd-and-higher-order-derivatives.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/13-symbolic-autograd-and-higher-order-derivatives.ipynb) |
    | **14** | Cross-Framework Interoperability &amp; Bridges | Production Export | `to_pytorch`, `model_to_pytorch`, `to_tensorflow`, `to_jax` | Weight Bridge | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/14-framework-interoperability-pytorch-tensorflow-jax.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/14-framework-interoperability-pytorch-tensorflow-jax.ipynb) |
    | **15** | Learning Rate Scheduling &amp; Optimization | Optimization Dynamics | `StepLR`, `CosineAnnealingLR`, `ExponentialLR`, `Adam`, `RMSprop` | Anisotropic Bowl | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/15-learning-rate-scheduling-and-optimization.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/15-learning-rate-scheduling-and-optimization.ipynb) |
    | **16** | Advanced CNNs with Residual Blocks | Deep Vision | `Conv2D`, `LayerNorm`, `ResidualDense`, `GlobalAveragePooling2D` | Handwritten Digits | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/16-advanced-cnn-architectures-with-residual-blocks.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/16-advanced-cnn-architectures-with-residual-blocks.ipynb) |
    | **17** | Tabular Deep Learning on California Housing | Tabular ML | `Datasets.load_california_housing`, `r2_score`, `mean_squared_error` | California Housing | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/17-tabular-deep-learning-california-housing.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/17-tabular-deep-learning-california-housing.ipynb) |
    | **18** | Time-Series Forecasting &amp; Spectral Analysis | Signal Processing | FFT spectral analysis, autoregressive sliding window | Harmonic Time Series | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/18-time-series-forecasting-and-transforms.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/18-time-series-forecasting-and-transforms.ipynb) |
    | **19** | Unsupervised Anomaly Detection Autoencoder | Representation Learning | Bottleneck Autoencoder ($784 \to 64 \to 16 \to 64 \to 784$) | MNIST Normal vs Anomaly | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/19-autoencoder-anomaly-detection.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/19-autoencoder-anomaly-detection.ipynb) |
    | **20** | Observatory Enterprise Diagnostics | Enterprise Telemetry | `TrainingLogger`, `LogLevel.EXPERT`, `AutoHealthWarnings`, HTML reports | Digits Telemetry | [Colab](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/20-training-observatory-enterprise-diagnostics.ipynb) &bull; [GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/20-training-observatory-enterprise-diagnostics.ipynb) |

---

## Running Notebooks Locally

```bash
git clone https://github.com/fahiiim/NeuroGebra.git
cd NeuroGebra
pip install -e ".[all]"
jupyter lab examples/projects/
```
