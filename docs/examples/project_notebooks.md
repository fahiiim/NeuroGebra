# Real Project Notebooks (Google Colab)

These are intermediate to advanced, real dataset projects designed for copy-paste and run in Google Colab.

## Real Dataset Policy

- Every notebook uses a real public dataset.
- Every notebook includes a `Dataset Source and Download Instructions` block.
- Each block includes: official source URL, auto-download command, and manual fallback instructions.

## Notebook Portfolio

| Notebook | Focus | Dataset | Open in Colab | GitHub |
|:--|:--|:--|:--|:--|
| 01-image-classification-fmnist-cnn.ipynb | Image classification baseline with NeuroGebra ModelBuilder | Fashion-MNIST (Zalando) | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/01-image-classification-fmnist-cnn.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/01-image-classification-fmnist-cnn.ipynb) |
| 02-image-classification-observatory-pro.ipynb | Image classifier diagnostics with AdaptiveLogger, health warnings, tiered storage, dashboard | Fashion-MNIST (Zalando) | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/02-image-classification-observatory-pro.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/02-image-classification-observatory-pro.ipynb) |
| 03-gan-mnist-baseline.ipynb | GAN baseline with NeuroGebra activation bridge into PyTorch | MNIST | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/03-gan-mnist-baseline.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/03-gan-mnist-baseline.ipynb) |
| 04-gan-mode-collapse-diagnostics.ipynb | GAN instability and mode-collapse diagnostics with Observatory Pro | MNIST | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/04-gan-mode-collapse-diagnostics.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/04-gan-mode-collapse-diagnostics.ipynb) |
| 05-diffusion-image-denoising-baseline.ipynb | Diffusion-inspired noise prediction baseline | CIFAR-10 | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/05-diffusion-image-denoising-baseline.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/05-diffusion-image-denoising-baseline.ipynb) |
| 06-diffusion-math-scheduler-deep-dive.ipynb | Diffusion schedule math, forward-process visualization, reproducibility fingerprint | CelebA (primary), CIFAR-10 fallback | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/06-diffusion-math-scheduler-deep-dive.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/06-diffusion-math-scheduler-deep-dive.ipynb) |
| 07-core-nlp-spam-classifier.ipynb | Core NLP spam detection with NeuroGebra ModelBuilder | SMS Spam Collection (UCI) | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/07-core-nlp-spam-classifier.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/07-core-nlp-spam-classifier.ipynb) |
| 08-core-nlp-sentiment-classifier.ipynb | Core NLP sentiment classification with bridge-assisted training and adaptive warnings | IMDb Large Movie Review | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/08-core-nlp-sentiment-classifier.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/08-core-nlp-sentiment-classifier.ipynb) |
| 09-small-language-model-charlm-from-scratch.ipynb | Small language model from scratch (character-level) | Tiny Shakespeare | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/09-small-language-model-charlm-from-scratch.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/09-small-language-model-charlm-from-scratch.ipynb) |
| 10-small-language-model-charlm-tuning.ipynb | Character LM tuning and fingerprint-based comparison | WikiText-2 | [Open](https://colab.research.google.com/github/fahiiim/NeuroGebra/blob/main/examples/projects/10-small-language-model-charlm-tuning.ipynb) | [View](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/10-small-language-model-charlm-tuning.ipynb) |

## Dataset Sources

- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- MNIST: http://yann.lecun.com/exdb/mnist/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- IMDb Reviews: https://ai.stanford.edu/~amaas/data/sentiment/
- Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- WikiText-2: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
