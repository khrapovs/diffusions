# AGENTS Instructions

## Commands

- Run a single test: `uv run pytest tests/path/to/test.py::TestClass::test_method -xvs`.
- Run all tests in package: `uv run pytest tests -xvs`.
- Run pre-commit hooks on all files before finishing off with a change: `uv run prek run -v --show-diff-on-failure --all-files`.
- Run Python scripts with `uv run path/to/script.py`.

## Coding Standards

- Each change in `.py` files should be formatted and linted with `ruff`, and type checked with `mypy`. These checks can be done in one command using pre-commit hooks (`prek`).
- No `assert` in production code.
- Comment sparingly — code says what, comments say why. Add a comment only when the reasoning is non-obvious and cannot be carried by a clear name or the code itself. Do not write narrating comments that restate the next line, do not pad logic with multi-line prose, and do not repeat the same rationale at several sites — put one concise note at the source of truth and let the others stand on their own. Tests whose names already describe intent need no explanatory comment. Reserve longer explanation for genuinely complex or non-obvious logic, and keep even that as tight as it can be. Over-commenting is noise that ages badly and obscures the code it wraps.
- Name functions and methods with action verbs: `get_`, `extract_`, `find_`, `compute_`, `build_`, etc. Avoid noun-only names like `_serialize_keys` or `_base_names` — they read as attributes, not callables. Predicates (`is_`, `has_`) are the one exception.

## Testing Standards

- Target exactly 100% coverage of what the PR changes — no more, no less. Every changed or added behavior must have a test; every test must fail without the PR's change.
- Do not add tests for pre-existing logic that was already present before the PR, and do not test standard-library or third-party functions. The exception is deliberate behavior or integration tests, which may cross those boundaries by design.
- Use `pytest` patterns, not `unittest.TestCase`.
- Use `@pytest.mark.parametrize` for multiple similar inputs — consolidate tests that only differ in input/expected values into a single parametrized test.
- Unit tests are not allowed to access private attribute/methods of classes.

## Scope Discipline

- Do not commit anything. Only add to the staging area. Committing is the responsibility of the user.
- Only read files I explicitly name or point to.
- Do not read additional files to "get context," "understand the project," or "see how things connect" unless I ask you to.
- If you think reading more files would help, ask first. One sentence: "Want me to also read X?" Wait for my answer.
- This applies to every task in this project. No exceptions for "just checking" or "quick look."
- Each class should expose only those methods and attributes that are used in the other classes/functions. All other attributes and methods should be private (_method). Example:

    ```python
    class A:
        def __init__(self):
            self._x = 1 # should be private
            self.y = 2 # should be public

        def _private_method(self):
            pass # should be private

        def public_method(self):
            pass # should be public
    ```

## Build System

- **Modern build backend**: Uses `scikit-build-core` with PEP 517 instead of legacy `setup.py`.
- **Cython compilation**: The `simulate.pyx` module is pre-compiled to `simulate.c` and built via CMakeLists.txt.
- **Workflow for Cython changes**:
  1. Edit `src/affidiff/simulate.pyx` as needed
  2. Pre-compile to C: `uv run cython src/affidiff/simulate.pyx`
  3. Build with `uv build`
  4. Test with `uv run pytest`

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

- `src/AGENTS.md` - Production source code root containing affidiff package
  - `src/affidiff/AGENTS.md` - Core affine diffusion models, parameter classes, moments, and Cython simulation helpers
- `tests/AGENTS.md` - Test suite covering affine diffusion models, parameter classes, moments estimation, and simulation utilities
- `examples/AGENTS.md` - Runnable usage scripts for affine diffusions models and simulation/estimation workflows
- `docs/AGENTS.md` - Sphinx documentation source files and configuration

**Root-owned files** (no child DOX needed):
- `.github/` - CI/CD workflows
- Configuration files: `pyproject.toml`, `.pre-commit-config.yaml`, `.travis.yml`, `.coveragerc`, `.gitignore`, `setup.py`
- Documentation & math assets: `README.md`, `CHANGELOG.md`, `LICENSE.md`, `models.lyx`, `models.pdf`
