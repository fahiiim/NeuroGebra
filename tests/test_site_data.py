"""Verify that the website's mathematical illustrations match the real library."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neurogebra import MathForge

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs/assets/data/explorer.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", sorted(DATA["curves"]))
def test_activation_samples_match_library(key):
    curve = DATA["curves"][key]
    params = {"beta": curve["beta"]} if curve["beta"] is not None else None
    expression = MathForge().get(curve["name"], params=params)
    x = np.asarray(DATA["x"])
    np.testing.assert_allclose(curve["values"], expression.eval(x=x), atol=5e-10)
    np.testing.assert_allclose(
        curve["derivatives"], expression.gradient("x").eval(x=x), atol=5e-10
    )
    assert curve["formula"] == str(expression.symbolic_expr)


def test_relu_zero_convention_is_explained():
    index = DATA["x"].index(0.0)
    assert DATA["curves"]["relu"]["values"][index] == 0
    assert DATA["curves"]["relu"]["derivatives"][index] == 0.5
    assert "not differentiable at zero" in DATA["curves"]["relu"]["note"]


def test_training_trace_is_reproducible_and_does_not_change_rng():
    spec = importlib.util.spec_from_file_location(
        "site_data_generator", ROOT / "scripts/generate_site_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = np.random.get_state()
    fresh = module.build_data()
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    np.testing.assert_allclose(
        fresh["training"]["losses"], DATA["training"]["losses"], atol=2e-9
    )
    assert fresh["training"]["epochs"] == 80
    assert fresh["training"]["losses"][-1] < fresh["training"]["losses"][0]
    assert fresh["training"]["weight"] == pytest.approx(2, abs=0.03)
    assert fresh["training"]["bias"] == pytest.approx(1, abs=0.01)


def test_core_reference_retains_legacy_member_anchors():
    content = (ROOT / "docs/api/core.md").read_text(encoding="utf-8")
    for name in [
        "Expression.eval",
        "Expression.gradient",
        "MathForge.get",
        "Trainer.fit",
    ]:
        module = (
            "forge"
            if name.startswith("MathForge")
            else "trainer" if name.startswith("Trainer") else "expression"
        )
        assert f'id="neurogebra.core.{module}.{name}"' in content
    for qualified in [
        "expression.Expression-functions",
        "expression.Expression-attributes",
        "autograd.Value-functions",
        "autograd.Tensor-functions",
        "autograd.Tensor-attributes",
        "forge.MathForge-functions",
        "trainer.Trainer-functions",
        "neurocraft.NeuroCraft-functions",
    ]:
        assert f'id="neurogebra.core.{qualified}"' in content


def test_preview_svg_contains_real_function_paths():
    from xml.etree import ElementTree

    svg = ElementTree.parse(ROOT / "docs/assets/images/sigmoid-preview.svg")
    assert svg.getroot().attrib["viewBox"] == "0 0 640 240"
    paths = svg.findall("{http://www.w3.org/2000/svg}path")
    assert len(paths) == 2
    assert all(path.attrib["d"].count("L") == len(DATA["x"]) - 1 for path in paths)
