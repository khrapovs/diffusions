# Purpose

`docs_src` contains the MkDocs source files (built via `mkdocs.yaml` at the repo root) that generate the documentation
website and API reference for `affidiff`. `source/` holds the legacy Sphinx sources kept for historical reference; it
is not part of the live MkDocs build.

# Ownership

- `index.md` — MkDocs home page (includes `README.md`).
- `models/` — Model description pages (math + links to generated API reference) rendered by MkDocs, replacing the
  descriptions previously in `source/*.rst`.
- `gen_ref_pages.py` — Auto-generates the full `reference/` API section from `src/` docstrings via `mkdocstrings`.
- `source/` — Legacy Sphinx configuration and `.rst` sources; not built or maintained going forward.

# Local Contracts

- Docstrings across production modules must adhere to standard NumPy format (as configured in `pyproject.toml` under
  `tool.ruff.lint.pydocstyle`).
- `models/*.md` pages must not re-embed `mkdocstrings` (`:::`) blocks for classes already auto-documented under
  `reference/` — this causes duplicate-anchor build failures under `mkdocs build -s`. Link to the reference page
  instead (e.g. `../reference/affidiff/param_vasicek.md`).
- New model pages must be added to the `nav.Models` section in `mkdocs.yaml`.

# Work Guidance

- Ensure new public functions, classes, and parameter types added to `affidiff` are documented and visible in the
  generated `reference/` section (automatic via `gen_ref_pages.py`, no manual step needed).

# Verification

- Build documentation via `uv run mkdocs build -s -c` from the repo root; must complete without warnings/errors.

# Child DOX Index

None (leaf boundary).
