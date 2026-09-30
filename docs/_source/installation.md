# Installation

Installing `crystALL` on your machine can be done via the `PyPI` package repository, or by cloning from GitHub directly, following the instructions below. You can also effortlessly run it on the cloud [by clicking here to launch a pre-configured GitHub Codespaces environment](https://codespaces.new/PhasesResearchLab/crystALL?quickstart=1) with [tutorials](tutorials.md) ready to go.

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