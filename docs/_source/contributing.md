# Contributing

We welcome contributions from the wider scientific community and appreciate bug reports/fixes, documentation improvements, new adapters/models, and larger feature proposals.

For coding contributions, fork the repository, create a topic branch, and open a pull request (PR) against `main`. Please follow the project's linters/formatters (Black/flake8/isort), add type hints and clear docstrings for public APIs, include pytest tests (prefer existing fixtures), and avoid adding large dependencies without prior discussion.

If reporting a bug or requesting a feature, please open an issue with a minimal reproduction (code snippet or dataset), environment details, and the expected vs actual behaviour; users may also raise questions and contribute to discussions via issues. 

**Adding property-prediction models via calculator adapters:**
Extending crystALL with new property predictors is welcome. We integrate models as *calculator adapters* so the rest of the codebase can call different predictor objectives via a uniform interface (see `alignn.py` and `chgnet.py`). Key requirements for adapters:

- Adapter location: Create a module under `crystall/calculators/` (one file or a small package).
- I/O contract: accept pymatgen `Structure`; document outputs in the module docstring.
- API methods: adapters must expose two public methods:
	- `run_per_structure` — run the model on a single structure and return a value or structured dict.
	- `run_on_directory` — run the adapter over many structures and support both serial and parallel execution.
- Tests & examples: include unit tests with tiny fixtures.
- Lightweight runtime: Avoid forcing heavy runtime dependencies for users who don't need them. If an adapter requires optional packages, guard imports with try/except and show a clear error message describing how to install the extras.