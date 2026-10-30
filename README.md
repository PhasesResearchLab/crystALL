# crystALL

<div align="center">
  <img src="https://github.com/PhasesResearchLab/crystALL/blob/ce4bfd1bb0b16eabdcc2cd28dcf9333f657b122c/docs/_static/crystALL.png" alt="crystALL Logo" width="180"/>
  <br>
</div>

[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/crystall-search)](https://pypi.org/project/crystall-search)
[![PyPI - Version](https://img.shields.io/pypi/v/crystall-search?label=PyPI&color=green)](https://pypi.org/project/crystall-search)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/crystall-search)](https://pypi.org/project/crystall-search)
[![LinuxTests](https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_LinuxUbuntu.yaml/badge.svg)](https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_LinuxUbuntu.yaml)


Example Publications using `crystALL`:
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.actamat.2021.117448-blue)](https://doi.org/10.1016/j.actamat.2021.117448)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--021--03578--0-blue)](https://doi.org/10.1038/s41598-021-03578-0)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.molliq.2025.127811-blue)](https://doi.org/10.1016/j.molliq.2025.127811)

## Summary

**crystALL** is an open-source data-driven framework for atomic structure prediction and materials discovery, designed to address the critical gap between computational predictions and experimental observations. Unlike methods focused solely on stable ground states, crystALL systematically embraces structural diversity and compositional uncertainty, enabling the discovery, modeling, and validation of both stable and metastable crystal configurations, going beyond more common data-driven discovery approaches that prioritize only the lowest-energy structures often failing to explain experimental observations or propose structures stable under conditions of interest.

By leveraging millions of relaxed and experimental structures from the MPDD Database and OPTIMADE API, `crystALL` rapidly generates wide ensembles of candidate configurations for either target compositions or compositional ranges, with the latter being a unique strength of `crystALL`+`MPDD` combination. These candidates undergo all permutations of atomic substitution, advanced machine learning-based featurization, clustering, and multi-model property prediction, supporting robust selection and direct ab initio or experimental validation (in a typical use case).

## Features

_This is just a quick introduction to the main features and workflow of crystALL. Please refer to the `crystALL` journal article for a complete, yet still concise, description of the framework, motivation behind it, and the underlying methodology._

### Overview

**On the high-level, `crystALL` allows you to take a chemical composition/s of interest, whether you measured it experimentally or hypothesized it, and then systematically explore them with minimal bias to form an approximately ergodic ensemble of candidates by feature space analysis and evaluation over any metric of interest to fulfill your research objectives.** This description is not vague, but rather emphasizes the framework's ability to fit different needs. These can range from [discovering crystal structure of previously unexplained experimental observations](https://doi.org/10.1016/j.actamat.2021.117448) to high-throughput screening for novel materials exhibiting certain topological characteristics (ongoing work). Overall, `crystALL` is a great fit for both experimentalists and computational researchers aiming to accelerate discovery of novel materials in underexplored regions of chemical space or cases escaping the generative learning or conventional computational approaches focusing only on the most probable answers.

In a _typical_ workflow, depicted in the figure below for a single target composition ("compound"), `crystALL` systematically generates, featurizes, and evaluates candidate crystal structures to identify the most promising configurations. It starts by gathering prototype structures (provided via MPDD / OPTIMADE API or local files) and performing all permutations of substitution into the target composition (orange). Then, it featurizes the candidates with `pySIPFENN`'s (python toolset for Structure-Informed Property and Feature Engineering with Neural Networks) `KS2022` (by default) to then analyze the embeddings (blue) while predicting properties of interest with ML models from pySIPFENN and community (purple). Filtering and clustering in the feature space are used to select top candidates from unique structures/energy basins (green), and final DFT relaxations leading to match with experimental observations (red). 

The example depicts a dramatic filtering cascade - 5M structures from MPDD are harnessed to generate around 50k ABCD3 candidates, then ML-filtered to 5k in 500 clusters, culminating in DFT calculations for the top 50 cluster representatives to construct an approximately ergodic ensemble and match experimental observations; while feedback loop to MPDD creates a self-improving ecosystem for materials discovery that grows with `crystALL` use.

<div align="center">
  <img src="https://raw.githubusercontent.com/PhasesResearchLab/crystALL/2cc27a91b6396b8cc776075b3696411f061b268c/docs/_static/crystALL_Compounds_V7.png" alt="Main figure for compound search" width="750"/>
</div>

**One of the unique features within `crystALL`, mentioned in the summary and explored in detail in the paper, is the ability to perform searches over compositional ranges, positioning it as an invaluable tool for interpreting real measurements of compositions, without assuming that molar ratios are low integers** - a very common pitfall researchers fall into. As illustrated below, **when combined with `MPDD`, `crystALL` accepts an uncertainty range for each element individually and systematically explores all prototypical structures within the specified compositional subspace**, covering both low- and high-index anonymous formulas.

<div align="center">
  <img src="https://raw.githubusercontent.com/PhasesResearchLab/crystALL/2cc27a91b6396b8cc776075b3696411f061b268c/docs/_static/crystALL_Compositional_V5.png" alt="Main figure for compositional search" width="750"/>
</div>

### Scoring and Selection - NaMg(BH4)3 Example

To illustrate the scoring step, the bar plot below shows formation energies sorted from lowest to highest for a focused ensemble of 100 `NaMg(BH4)3` prototypes. This view helps rapidly identify low‑energy regions in the ensemble and informs which clusters deserve further attention.

Property prediction in `crystALL` is modular. `pySIPFENN` models are available natively and external calculators plug in through adapter classes. `ALIGNN` and `CHGNet` are provided as examples of such adapters: `ALIGNN` exposes multi‑head formation energy outputs (`MP` and `DFT-JARVIS`) which can be filtered according to user request, and `CHGNet` reports static and relaxed energies computed independently so users can skip relaxation when it is not needed. The adapters accept an explicit list of objectives, enabling selective computation and reduced runtime for expensive predictors.

<div align="center">
  <img src="https://raw.githubusercontent.com/PhasesResearchLab/crystALL/2cc27a91b6396b8cc776075b3696411f061b268c/docs/_static/barplot.png" alt="Sorted bar chart of formation energies" width="750"/>
</div>

After featurization and property assignment, `crystALL` clusters the prototype ensemble (t‑SNE for embedding + KMeans/DBSCAN for grouping) and annotates clusters with objective statistics. The figure below shows the t‑SNE embedding for the same `NaMg(BH4)3` set: seven clusters are recovered and correspond to distinct structural families (for example, clusters 1 and 2 are enriched in hexagonal motifs, cluster 3 in triclinic arrangements, clusters 4, 5 and 7 in trigonal lattices, and cluster 6 contains perovskite‑type geometries). This separation illustrates how descriptor‑based clustering can disentangle fundamentally different local chemistries and geometries.

<div align="center">
  <img src="https://raw.githubusercontent.com/PhasesResearchLab/crystALL/2cc27a91b6396b8cc776075b3696411f061b268c/docs/_static/2dcluster.png" alt="2D t-SNE of prototypes colored by clusters" width="750"/>
</div>

The final selection step produces Top Candidates by selecting the lowest‑energy representative from each cluster for every available objective; these candidates provide a compact, diverse set for higher‑accuracy DFT relaxation or experimental validation. Outputs are exportable as CSV (`all-prototypes` and `top-candidates`), and an optional MPDD convex‑hull check (energy‑above‑hull) can be performed to further prioritize thermodynamically plausible entries.

## News

_For detailed account of changes, please consult the GitHub [changelog](https://github.com/PhasesResearchLab/crystALL/releases)._

- `v0.5.2` - Updated compatibility with the latest versions of Python and key dependencies.
- `v0.5.1` - Improved MPDD querying capabilities and documentation.
- `v0.5.0` - Major re-write of all code except for the core `crystALL` logic, turning it into a user-ready tool with additional ML models, polished architecture, user-friendly tutorials, and improved documentation.
- `v0.4.0` - Initial private GitHub release of crystALL, separated from MPDD repository; as used in [10.1016/j.actamat.2021.117448](https://doi.org/10.1016/j.actamat.2021.117448) or [10.1038/s41598-021-03578-0](https://doi.org/10.1038/s41598-021-03578-0).


## Installation

Installing `crystALL` on your machine can be done via the `PyPI` package repository, or by cloning from GitHub directly, following the instructions below. You can also effortlessly run it on the cloud [by clicking here to launch a pre-configured GitHub Codespaces environment](https://codespaces.new/PhasesResearchLab/crystALL?quickstart=1) with [tutorials](#tutorials) ready to go.

While not required, it is recommended to first set up a virtual environment using `micromamba`, `miniconda`, or `venv`. This ensures that (a) one of the required versions of Python (3.11+) is used and (b) there are no dependency conflicts. If you have Conda installed on your system (see [`micromamba` install instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)), you can create a new environment using:

``` shell
micromamba create -n crystall python=3.12
micromamba activate crystall
```

Then install `crystALL` (regular)
``` shell
pip install crystall-search
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

## New in v0.6.0

The upcoming `v0.6.0` release introduces a fully modular architecture driven by the `MLIPAdapter` registry, enabling automated structure relaxations and multi-property predictions across diverse machine-learned interatomic potentials. Each potential manages its own execution micro-environment with support for CPU and CUDA acceleration, full cell/volume optimization, and optional piping into downstream static property models. Below is a summary of the machine-learned interatomic potentials integrated into `crystALL` v0.6.0 via the `MLIPAdapter` registry:

| Model Family | Checkpoint Pinned | Hardware | Distribution | Citation |
| :--- | :--- | :--- | :---: | :--- |
| **DeePMD** | DPA-4.0.1-Pro-MPtrj | CPU, CUDA | From source | [Wang et al. (2018)](https://doi.org/10.1016/j.cpc.2018.03.016) |
| **CHGNet** | 0.3.0 | CPU, CUDA | PyPI | [Deng et al. (2023)](https://doi.org/10.1038/s42256-023-00716-3) |
| **TensorNet** | PES-MatPES-PBE-2025.2 | CPU, CUDA | PyPI | [Simeon & de Fabritiis (2023)](https://doi.org/10.48550/arXiv.2306.06482) |
| **ORB** | v3-omat:float32-high | CPU, CUDA | PyPI | [Rhodes et al. (2025)](https://doi.org/10.48550/arXiv.2504.06231) |
| **ORB-D3** | v3-omat:float32-high:PBE:BJ | CPU, CUDA | PyPI | [Rhodes et al. (2025)](https://doi.org/10.48550/arXiv.2504.06231) |
| **Allegro** | OAM-L:0.1 | CUDA | From source | [Musaelian et al. (2023)](https://doi.org/10.1038/s41467-023-36329-y) |
| **AlphaNet** | alex-0410:oma | CPU, CUDA | From source | [Yin et al. (2025)](https://doi.org/10.1038/s41524-025-01817-w) |
| **Eqnorm** | eqnorm-mptrj | CPU, CUDA | From source | [Chen et al. (2025)](https://github.com/yzchen08/eqnorm) |
| **eSEN** | eSEN-30M-OAM | CPU, CUDA | From source | [Fu et al. (2025)](https://doi.org/10.48550/arXiv.2502.12147) |
| **MACE** | mh-1:omat-pbe | CPU, CUDA | PyPI | [Batatia et al. (2022)](https://doi.org/10.48550/arXiv.2206.07697) |
| **MatterSim** | v1.0.0-5M | CPU, CUDA | PyPI | [Yang et al. (2024)](https://doi.org/10.48550/arXiv.2405.04967) |
| **NequIP** | OAM-XL:0.1 | CUDA | PyPI | [Batzner et al. (2022)](https://doi.org/10.1038/s41467-022-29939-5) |
| **Nequix** | oam-1 | CPU, CUDA | From source | [Koker et al. (2025)](https://doi.org/10.48550/arXiv.2508.16067) |
| **PolarMACE** | polar-1-l | CPU, CUDA | From source | [Batatia et al. (2026)](https://doi.org/10.48550/arXiv.2602.19411) |
| **QET** | PES-MatPES-PBE-2025.2 | CPU, CUDA | From source | [Ko et al. (2025)](https://doi.org/10.48550/arXiv.2511.07249) |
| **SevenNet** | omni-i12:omat24 | CPU, CUDA | PyPI | [Park et al. (2024)](https://doi.org/10.1021/acs.jctc.4c00190) |
| **UPET** | oam-xl | CPU, CUDA | PyPI | [Bigi et al. (2026)](https://doi.org/10.1038/s41467-025-65662-7) |
| **EquiformerV3** | omat24-mptrj-salex-gradient | CPU, CUDA | From source | [Liao et al. (2026)](https://doi.org/10.48550/arXiv.2604.09130) |
| **MatRIS** | 10m-oam:efsm | CPU, CUDA | From source | [Zhou et al. (2025)](https://doi.org/10.48550/arXiv.2603.02002) |
| **UMA** | s-1p2:omat | CPU, CUDA | PyPI | [Wood et al. (2025)](https://doi.org/10.48550/arXiv.2506.23971) |
| **GRACE** | 2L-SMAX-OMAT-large | CPU, CUDA | PyPI | [Bochkarev et al. (2024)](https://doi.org/10.1103/PhysRevX.14.021036) |
| **TACE** | OAM-L | CPU, CUDA | From source | [Xu et al. (2025)](https://doi.org/10.48550/arXiv.2509.14961) |
| **HIENet** | 0 | CPU, CUDA | From source | [Yan et al. (2025)](https://doi.org/10.48550/arXiv.2503.05771) |
| **OSAKA** | osaka26:small | CPU, CUDA | PyPI | [Kuroda et al. (2026)](https://doi.org/10.48550/arXiv.2603.03223) |
| **Total** | **24 Models** | | | |

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
