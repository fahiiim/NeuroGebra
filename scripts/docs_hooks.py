"""Small build-time additions shared by the documentation templates."""

import json
from pathlib import Path


def on_config(config):
    path = Path(__file__).resolve().parents[1] / "docs/assets/data/explorer.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Run python scripts/generate_site_data.py before building the site"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError("Unsupported website demonstration data schema")
    config.extra["site_demo"] = data
    return config
