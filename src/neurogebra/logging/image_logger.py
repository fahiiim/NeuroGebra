"""
ImageLogger — Pixel-level image and activation-map visualisation.

Renders input images as coloured ASCII art, visualises convolutional
activation maps, filter weights, and saliency maps directly in the
terminal or exports them as PNG/GIF files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Unicode block characters for grayscale rendering (darkest → lightest)
_GRAY_CHARS = " ░▒▓█"
_BLOCK = "█"


class ImageLogger:
    """
    Render images and activation maps in the terminal.

    Features:
    - Input image preview (grayscale & RGB) as ASCII art
    - Conv2D activation map rendering
    - Filter weight visualization
    - Saliency / gradient-weighted map display
    - Export to PNG (requires matplotlib)
    """

    def __init__(self, max_width: int = 60, max_height: int = 30):
        self.max_width = max_width
        self.max_height = max_height
        self._console = Console(width=max_width + 20) if HAS_RICH else None

    # ------------------------------------------------------------------
    # Input image rendering
    # ------------------------------------------------------------------

    def render_image(self, image: np.ndarray, title: str = "Input Image",
                     grayscale: bool = False) -> str:
        """
        Render an image as ASCII art in the terminal.

        Args:
            image: numpy array of shape (H, W), (H, W, 1), or (H, W, 3).
                   Values should be in [0, 1] or [0, 255].
            title: Title for the panel.
            grayscale: Force grayscale rendering even for RGB.

        Returns:
            The ASCII art string.
        """
        img = self._normalise(image)

        # Determine if grayscale
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            return self._render_grayscale(img.squeeze(), title)
        elif img.ndim == 3 and img.shape[2] == 3 and not grayscale:
            return self._render_rgb(img, title)
        else:
            return self._render_grayscale(img.mean(axis=-1) if img.ndim == 3 else img, title)

    def _normalise(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float64)
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0, 1)

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        """Downsample image to fit terminal dimensions."""
        h, w = img.shape[:2]
        scale_w = self.max_width / w if w > self.max_width else 1.0
        scale_h = self.max_height / h if h > self.max_height else 1.0
        scale = min(scale_w, scale_h)
        if scale < 1.0:
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            # Simple nearest-neighbour downsampling
            row_idx = np.linspace(0, h - 1, new_h).astype(int)
            col_idx = np.linspace(0, w - 1, new_w).astype(int)
            if img.ndim == 3:
                return img[np.ix_(row_idx, col_idx)]
            return img[np.ix_(row_idx, col_idx)]
        return img

    def _render_grayscale(self, img: np.ndarray, title: str) -> str:
        img = self._downsample(img)
        h, w = img.shape
        lines = []
        for row in range(h):
            line_chars = []
            for col in range(w):
                v = img[row, col]
                idx = int(v * (len(_GRAY_CHARS) - 1))
                line_chars.append(_GRAY_CHARS[idx] * 2)  # double-width for aspect ratio
            lines.append("".join(line_chars))

        art = "\n".join(lines)
        if self._console:
            self._console.print(Panel(art, title=f"[bold cyan]{title}[/]",
                                      subtitle=f"{img.shape[1]}×{img.shape[0]} px",
                                      border_style="cyan"))
        else:
            print(f"\n{title} ({img.shape[1]}×{img.shape[0]}):\n{art}\n")
        return art

    def _render_rgb(self, img: np.ndarray, title: str) -> str:
        img = self._downsample(img)
        h, w, _ = img.shape

        if not self._console:
            return self._render_grayscale(img.mean(axis=-1), title)

        text = Text()
        for row in range(h):
            for col in range(w):
                r, g, b = (int(img[row, col, c] * 255) for c in range(3))
                text.append(_BLOCK * 2, style=f"rgb({r},{g},{b})")
            text.append("\n")

        self._console.print(Panel(text, title=f"[bold cyan]{title}[/]",
                                  subtitle=f"{w}×{h} px (RGB)",
                                  border_style="cyan"))
        return str(text)

    # ------------------------------------------------------------------
    # Activation map rendering
    # ------------------------------------------------------------------

    def render_activation_maps(self, activations: np.ndarray,
                               title: str = "Activation Maps",
                               max_maps: int = 8) -> None:
        """
        Render Conv2D activation maps.

        Args:
            activations: shape (H, W, C) or (C, H, W) — multiple feature maps.
            max_maps: Maximum number of maps to display.
        """
        a = np.asarray(activations, dtype=np.float64)
        if a.ndim == 4:
            a = a[0]  # Take first sample from batch

        # Ensure (H, W, C)
        if a.ndim == 3 and a.shape[0] < a.shape[2]:
            a = np.transpose(a, (1, 2, 0))

        n_maps = min(a.shape[2] if a.ndim == 3 else 1, max_maps)

        if self._console:
            self._console.print(f"\n[bold cyan]{title}[/] — showing {n_maps} feature maps")

        for i in range(n_maps):
            fmap = a[:, :, i] if a.ndim == 3 else a
            # Normalise per-map
            mn, mx = fmap.min(), fmap.max()
            if mx - mn > 0:
                fmap = (fmap - mn) / (mx - mn)
            else:
                fmap = np.zeros_like(fmap)
            self._render_grayscale(fmap, title=f"Feature Map {i}")

    # ------------------------------------------------------------------
    # Filter visualisation
    # ------------------------------------------------------------------

    def render_filters(self, weights: np.ndarray,
                       title: str = "Conv Filter Weights",
                       max_filters: int = 8) -> None:
        """
        Render convolutional filter weights.

        Args:
            weights: shape (kH, kW, C_in, C_out) or similar.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 4:
            n_filters = min(w.shape[3], max_filters)
            for f in range(n_filters):
                filt = w[:, :, 0, f]  # first input channel
                mn, mx = filt.min(), filt.max()
                if mx - mn > 0:
                    filt = (filt - mn) / (mx - mn)
                else:
                    filt = np.zeros_like(filt)
                self._render_grayscale(filt, title=f"Filter {f}")

    # ------------------------------------------------------------------
    # Saliency map
    # ------------------------------------------------------------------

    def render_saliency(self, input_image: np.ndarray,
                        gradients: np.ndarray,
                        title: str = "Saliency Map") -> None:
        """
        Render a saliency map highlighting which pixels matter most.

        Args:
            input_image: Original input (H, W) or (H, W, C).
            gradients: Gradient of loss w.r.t. input (same shape).
        """
        saliency = np.abs(np.asarray(gradients, dtype=np.float64))
        if saliency.ndim == 3:
            saliency = saliency.max(axis=-1)
        # Normalise
        mn, mx = saliency.min(), saliency.max()
        if mx - mn > 0:
            saliency = (saliency - mn) / (mx - mn)
        else:
            saliency = np.zeros_like(saliency)

        self._render_grayscale(saliency, title=title)

    # ------------------------------------------------------------------
    # Detect image data
    # ------------------------------------------------------------------

    @staticmethod
    def is_image_data(X: np.ndarray) -> bool:
        """Heuristic: does X look like image data?"""
        if X.ndim == 4 and X.shape[1] >= 4 and X.shape[2] >= 4:
            return True  # (N, H, W, C) or (N, C, H, W)
        if X.ndim == 3 and X.shape[1] >= 4:
            return True  # (N, H, W) grayscale
        return False

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_image(self, image: np.ndarray, path: str,
                   title: str = "Image") -> None:
        """Save image to file using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            img = self._normalise(image)
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax.imshow(img.squeeze(), cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=100)
            plt.close(fig)
        except ImportError:
            pass
