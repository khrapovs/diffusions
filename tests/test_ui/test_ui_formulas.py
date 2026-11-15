"""Tests for model formulas rendered in the UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from streamlit.testing.v1 import AppTest

from affidiff.ui.formulas import extract_model_formula

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("model_name", "equation"),
    [
        ("GBM", r"\frac{dS_{t}}{S_{t}}"),
        ("Vasicek", r"dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta dW_{t}"),
        ("CIR", r"\eta\sqrt{r_{t}}dW_{t}"),
        ("Heston", r"d\sigma_{t}^{2}&=\kappa\left(\mu-\sigma_{t}^{2}\right)dt"),
        ("CentTend", r"dv_{t}^{2}&=\kappa_{v}"),
    ],
)
def test_extract_model_formula_reads_docs_page(*, model_name: str, equation: str) -> None:
    formula = extract_model_formula(model_name)

    assert formula is not None
    assert equation in formula
    assert "../reference" not in formula


def test_extract_model_formula_returns_none_for_missing_page(tmp_path: Path) -> None:
    assert extract_model_formula("Unknown", docs_dir=tmp_path) is None


def _run_formula_app(model_name: str) -> None:
    from affidiff.ui.formulas import render_model_formula

    render_model_formula(model_name)


@pytest.mark.parametrize(("model_name", "n_expanders"), [("GBM", 1), ("Unknown", 0)])
def test_render_model_formula_shows_collapsed_card(*, model_name: str, n_expanders: int) -> None:
    app = AppTest.from_function(_run_formula_app, args=(model_name,)).run()

    assert not app.exception
    assert len(app.expander) == n_expanders
    if n_expanders:
        assert app.expander[0].label == f"Model equations: {model_name}"
        assert r"\sigma dW_{t}" in app.expander[0].markdown[0].value
