"""Build the website illustrations from real NeuroGebra calculations.

Run from the checkout: python scripts/generate_site_data.py
No timings, timestamps, external datasets or network services are used.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from neurogebra import Layer, MathForge, Model, __version__

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/assets/data/explorer.json"


def build_data():
    forge = MathForge()
    x = np.linspace(-4.0, 4.0, 161)
    names = {"sigmoid": "Sigmoid", "tanh": "Tanh", "relu": "ReLU", "swish": "Swish"}
    curves = {}
    for name, label in names.items():
        for beta in ([0.5, 1.0, 2.0] if name == "swish" else [None]):
            expression = forge.get(
                name, params={"beta": beta} if beta is not None else None
            )
            derivative = expression.gradient("x")
            values = np.asarray(expression.eval(x=x), dtype=float)
            slopes = np.asarray(derivative.eval(x=x), dtype=float)
            assert values.shape == slopes.shape == x.shape
            assert np.isfinite(values).all() and np.isfinite(slopes).all()
            key = f"swish-{beta:g}" if beta is not None else name
            curves[key] = {
                "name": name,
                "label": label,
                "beta": beta,
                "formula": str(expression.symbolic_expr),
                "latex": expression.latex(),
                "values": values.round(10).tolist(),
                "derivatives": slopes.round(10).tolist(),
                "note": (
                    "ReLU is not differentiable at zero. The symbolic Heaviside convention gives 0.5 there."
                    if name == "relu"
                    else "Change x to inspect the function and its symbolic derivative at the same input."
                ),
            }

    # Restore the caller's global RNG; Model's initialization and shuffle use it.
    random_state = np.random.get_state()
    try:
        np.random.seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            model = Model([Layer("dense", units=1)], name="website_line")
            model.compile(
                loss="mse",
                optimizer="sgd",
                learning_rate=0.08,
                log_level="basic",
                metrics=[],
            )
            features = np.linspace(-1.0, 1.0, 64).reshape(-1, 1)
            targets = 2.0 * features + 1.0
            history = model.fit(
                features,
                targets,
                epochs=80,
                batch_size=64,
                validation_split=0,
                verbose=False,
            )
        losses = np.asarray(history["loss"])
        assert np.isfinite(losses).all() and losses[-1] < losses[0]
        training = {
            "seed": 7,
            "epochs": len(losses),
            "samples": len(features),
            "backend": "NumPy / CPU",
            "optimizer": "SGD",
            "learning_rate": 0.08,
            "target": "y = 2x + 1",
            "loss": "mean squared error",
            "losses": losses.round(10).tolist(),
            "weight": round(float(model.layers[0].weights[0, 0]), 8),
            "bias": round(float(model.layers[0].bias[0]), 8),
        }
    finally:
        np.random.set_state(random_state)
    return {
        "schema": 1,
        "version": __version__,
        "x": x.round(2).tolist(),
        "curves": curves,
        "training": training,
    }


def render_preview(data):
    """Write a compact, accessible SVG for visitors without JavaScript."""
    curve = data["curves"]["sigmoid"]

    def path(values):
        return " ".join(
            f"{'M' if i == 0 else 'L'}{42 + 556 * i / 160:.2f},{195 - float(y) * 154:.2f}"
            for i, y in enumerate(values)
        )

    # Exact coordinates from the same library-generated arrays used by the explorer.
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 240" role="img" aria-labelledby="title desc">'
        '<title id="title">Sigmoid and its derivative</title>'
        '<desc id="desc">NeuroGebra-generated values from x=-4 to 4. At zero the sigmoid is 0.5 and its derivative is 0.25.</desc>'
        '<g stroke="#c5d4ce" stroke-width="1" fill="none">'
        '<path d="M42 41H598M42 118H598M42 195H598M42 25V210M320 25V210M598 25V210"/></g>'
        f'<path d="{path(curve["values"])}" fill="none" stroke="#007d76" stroke-width="3"/>'
        f'<path d="{path(curve["derivatives"])}" fill="none" stroke="#a57028" stroke-width="2.5" stroke-dasharray="6 5"/>'
        '<g font-family="system-ui,sans-serif" font-size="12" fill="#637575">'
        '<text x="35" y="231">-4</text><text x="316" y="231">0</text><text x="589" y="231">4</text>'
        '<text x="12" y="46">1</text><text x="12" y="199">0</text></g></svg>\n'
    )


def main():
    data = build_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    preview = ROOT / "docs/assets/images/sigmoid-preview.svg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    with preview.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(render_preview(data))
    print(
        f"Generated {len(data['curves'])} activation presets and {data['training']['epochs']} recorded training epochs"
    )


if __name__ == "__main__":
    main()
