"""Model formulas sourced from the MkDocs pages in ``docs_src/models``."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import streamlit as st

MODELS_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs_src" / "models"


@cache
def extract_model_formula(model_name: str, *, docs_dir: Path = MODELS_DOCS_DIR) -> str | None:
    """Extract the math description of a model from its documentation page.

    Paragraphs with relative links (API reference pointers) are dropped because they only resolve inside MkDocs.

    Parameters
    ----------
    model_name : str
        Model name as used in the UI, e.g. ``"GBM"``; the page is ``<model_name.lower()>.md``.
    docs_dir : Path
        Directory containing the model documentation pages.

    Returns
    -------
    str | None
        Markdown body, or ``None`` if the page does not exist.
    """
    page = docs_dir / f"{model_name.lower()}.md"
    if not page.is_file():
        return None
    _, _, body = page.read_text(encoding="utf-8").partition("\n")
    paragraphs = [p for p in body.strip().split("\n\n") if "](../" not in p]
    return "\n\n".join(paragraphs)


def render_model_formula(model_name: str) -> None:
    """Render the model equations in a collapsed card.

    Parameters
    ----------
    model_name : str
        Model name as used in the UI.
    """
    formula = extract_model_formula(model_name)
    if formula is None:
        return
    with st.expander(f"Model equations: {model_name}", icon=":material/function:", expanded=False):
        st.markdown(formula)
