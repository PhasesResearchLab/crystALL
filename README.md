# crystALL

[![GitHub top language](https://img.shields.io/github/languages/top/PhasesResearchLab/crystALL)](https://github.com/PhasesResearchLab/crystALL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Linux](https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_LinuxUbuntu.yaml/badge.svg)](https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_LinuxUbuntu.yaml)

<!---
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/crystall)](https://pypi.org/project/crystall)
[![PyPI - Version](https://img.shields.io/pypi/v/crystall?label=PyPI&color=green)](https://pypi.org/project/crystall)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/crystall)](https://pypi.org/project/crystall)

[![codecov](https://codecov.io/gh/PhasesResearchLab/pySIPFENN/branch/main/graph/badge.svg?token=S2J0KR0WKQ)](https://codecov.io/gh/PhasesResearchLab/pySIPFENN)
[![stable](https://img.shields.io/badge/Read%20The%20Docs-Stable-green)](https://pysipfenn.readthedocs.io/en/stable/) 
[![latest](https://img.shields.io/badge/Read%20The%20Docs-Latest-green)](https://pysipfenn.readthedocs.io/en/latest/)
-->

## Summary

**crystALL** is an open-source data-driven framework for atomic structure prediction and materials discovery, designed to address the critical gap between computational predictions and experimental observations. Unlike methods focused solely on stable ground states, crystALL systematically embraces structural diversity and compositional uncertainty, enabling the discovery, modeling, and validation of both stable and metastable crystal configurations.

By leveraging millions of relaxed and experimental structures from the MPDD database and OPTIMADE API, crystALL rapidly generates wide ensembles of candidate configurations for target compositions or compositional ranges. These candidates undergo atomic substitution, advanced machine learning-based featurization, clustering, and multi-model property prediction, supporting robust selection and direct ab initio or experimental validation.

crystALL uniquely supports searches accommodating experimental uncertainty and complex materials systems, generating unbiased, ensemble-focused predictions suitable for high-throughput candidate selection, statistical modeling, and validation against experiment or DFT.

## Features

**crystALL** provides a practical, science‑driven pipeline to generate, featurize, and prioritize candidate crystal structures for a target composition. At a high level the workflow gathers prototype structures (from local files or MPDD), computes consistent ML descriptors, applies one or more property predictors, and selects compact, experimentally actionable candidate sets.

Prototype generation and featurization are designed to be reproducible and lightweight: the code performs anonymized elemental substitutions and deduplication to build a controlled prototype ensemble (the default generation is intentionally constrained). Each prototype is then embedded into KS2022 descriptor space via pySIPFENN, yielding a uniform representation suitable for clustering and downstream models.

To illustrate the scoring step, the bar plot below shows formation energies sorted from lowest to highest for a focused ensemble of 100 NaMg(BH4)3 prototypes. This view helps rapidly identify low‑energy regions in the ensemble and informs which clusters deserve further attention.

<div align="center">
  <img src="docs/_static/barplot.png" alt="Sorted bar chart of formation energies" width="750"/>
</div>

Property prediction in crystALL is modular. pySIPFENN models are available natively and external calculators plug in through adapter classes. ALIGNN and CHGNet are provided as examples of such adapters: ALIGNN exposes multi‑head formation energy outputs (MP and JARVIS) which we filter according to user request, and CHGNet reports static and relaxed energies computed independently so users can skip relaxation when it is not needed. The adapters accept an explicit list of objectives, enabling selective computation and reduced runtime for expensive predictors.

After featurization and property assignment, crystALL clusters the prototype ensemble (t‑SNE for embedding + KMeans/DBSCAN for grouping) and annotates clusters with objective statistics. The figure below shows the t‑SNE embedding for the same NaMg(BH4)3 set: seven clusters are recovered and correspond to distinct structural families (for example, clusters 1 and 2 are enriched in hexagonal motifs, cluster 3 in triclinic arrangements, clusters 4, 5 and 7 in trigonal lattices, and cluster 6 contains perovskite‑type geometries). This separation illustrates how descriptor‑based clustering can disentangle fundamentally different local chemistries and geometries.

<div align="center">
  <img src="docs/_static/2dcluster.png" alt="2D t-SNE of prototypes colored by clusters" width="750"/>
</div>

The final selection step produces Top Candidates by selecting the lowest‑energy representative from each cluster for every available objective; these candidates provide a compact, diverse set for higher‑accuracy DFT relaxation or experimental validation. Outputs are exportable as CSV (`all-prototypes` and `top-candidates`), and an optional MPDD convex‑hull check (energy‑above‑hull) can be performed to further prioritize thermodynamically plausible entries.


## Installation

Installing `crystALL` on your machine can be done via the `PyPI` package repository, `conda-forge`  package repository, or by cloning from GitHub directly, following the instructions below. You can also effortlessly run it on the cloud [by clicking here to launch a pre-configured GitHub Codespaces environment](https://codespaces.new/PhasesResearchLab/crystALL?quickstart=1) with [tutorials](#tutorials) ready to go.

While not required, it is recommended to first set up a virtual environment using `micromamba`, `miniconda`, or `venv`. This ensures that (a) one of the required versions of Python (3.11+) is used and (b) there are no dependency conflicts. If you have Conda installed on your system (see [`micromamba` install instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)), you can create a new environment using:

``` shell
micromamba create -n crystall python=3.12
micromamba activate crystall
```

Then install `crystALL` (regular)
``` shell
pip install crystall
```

or if you work with the source code directly, you can clone the GitHub repository and install it in editable mode:
``` shell
pip install -e .
```

**Note:** To run _certain functions_ within the `crystALL` framework, you may need to connect to (a) access-restricted MPDD endpoint ([Material-Property-Descriptor Database](https://mpdd.org)) or your local MongoDB instance with the data loaded and (b) OPTIMADE-compliant endpoints, which are all open and include MPDD data with minor delay, for remote crystal structure data.

##### Additional Install Notes

The `dgl` library is infrequently updated as of 2026 but still widely used for deep learning on graphs and required by `ALIGNN` models within the `crystALL` framework, which are currently tightly integrated (until `v0.6.0`, which is fully modularized, becomes live). If you are having trouble installing `torch` / `dgl` / `mpdd-alignn` on your system, the recommended approach is to:

1. Consider utilizing the [provided pre-configured GitHub Codespace](https://codespaces.new/PhasesResearchLab/crystALL?quickstart=1), which includes all `crystALL` dependencies and a local MongoDB instance you can populate with MPDD or any other database of your choice, to avoid installation issues on your local system

2. If you are using **Linux**, the recommended approach is to first install the required system dependencies via wheels directly from `pytorch.org` and then from `dgl.ai` with the following commands:

    ```shell
    pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.7.0+cpu" "torchvision==0.22.0+cpu" "torchdata==0.9.0"
    pip install --no-deps 'dgl==2.5.0' -f https://data.dgl.ai/wheels/torch-2.6/repo.html
    ```

3. If you are using **Windows** or **macOS**, the versions between `v0.3.x` and `v0.5.0` should install without major issues using standard `pip` commands, possibly with manual `pip install "dgl==2.2.0" -f https://data.dgl.ai/wheels/torch-2.3/repo.html` first; however, versions `v0.5.x` had to top with the latest releases of `torch`. 

4. If you need to install the latest version on **Windows** or **macOS**, consider doing so in environments like Windows Subsystem for Linux [`WSL`](https://learn.microsoft.com/en-us/windows/wsl/install), Linux VM, or [`Docker`](https://www.docker.com/get-started) to avoid compatibility issues.

## Tutorials

If you would like to try **crystALL**'s functionalities in a reproducible, hands-on environment, we provide a pre-configured GitHub Codespace with all required dependencies and local-access to MongoDB database for running our tutorials:

   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PhasesResearchLab/crystALL?quickstart=1)

1. Click the button above to launch a ready-to-use, cloud-based workspace.
2. Once your Codespace loads, navigate to the `tutorial` directory to find curated Jupyter notebooks illustrating the main features and scientific workflows.
3. Open any notebook and execute cells interactively to explore candidate structure generation, ML-powered predictions, and data-driven analysis.

The tutorials currently consist of 3 Jupyter notebooks covering:
- [01. QuickStart.ipynb](tutorial/01.QuickStart.ipynb) - Core API, BaSO4 walk-through, clustering and visualization
- [02. HydridesExample.ipynb](tutorial/02.HydridesExample.ipynb) - Real-world discovery in complex metal hydrides.
- [03. RefiningMPDDSearch.ipynb](tutorial/03.RefiningMPDDSearch.ipynb) - Advanced queries (dimensionality filtering, crystal system constraints, compositional limits).

## Contributing

We welcome contributions from the wider scientific community and appreciate bug reports/fixes, documentation improvements, new adapters/models, and larger feature proposals.

For coding contributions, fork the repository, create a topic branch, and open a pull request (PR) against `main`. Please follow the project's linters/formatters (Black/flake8/isort), add type hints and clear docstrings for public APIs, include pytest tests (prefer existing fixtures), and avoid adding large dependencies without prior discussion.

If reporting a bug or requesting a feature, please open an issue with a minimal reproduction (code snippet or dataset), environment details, and the expected vs actual behaviour; users may also raise questions and contribute to discussions via issues. 

**Adding property-prediction models via calculator adapters:**
Extending crystALL with new property predictors is welcome. We integrate models as *calculator adapters* so the rest of the codebase can call different predictor objectives via a uniform interface (see `alignn.py` and `chgnet.py`). Key requirements for adapters:

- Adapter location: Create a module under `crystall/calculators/` (one file or a small package).
- I/O contract: accept pymatgen `Structure`; document outputs in the module docstring.
- API methods: adapters must expose two public methods:
	- `run_on_structure` — run the model on a single structure and return a value or structured dict.
	- `run_on_directory` — run the adapter over many structures and support both serial and parallel execution.
- Tests & examples: include unit tests with tiny fixtures.
- Lightweight runtime: Avoid forcing heavy runtime dependencies for users who don't need them. If an adapter requires optional packages, guard imports with try/except and show a clear error message describing how to install the extras.


## Citing

**crystALL** is an open-source Python package developed by the [Phases Research Lab](https://phaseslab.com/) at The Pennsylvania State University, under the supervision of Dr. Zi-Kui Liu, and distributed under the MIT License.

If you use **crystALL** in scientific publications, please cite:

- Adam Krajewski, Ricardo Amaral, Zi-Kui Liu, et al. "Data-driven candidate structure ensemble generation beyond the convex hull with crystALL". (2025).  
  [DOI:10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx), [arXiv:YYMM.xxxxx](https://arxiv.org/abs/YYMM.xxxxx)
