# Installation

Installing **crystALL** is simple and easy by utilizing **PyPI** package repository, **conda-forge**  package repository, or by cloning from **GitHub** directly. 

In order to use **crystALL**, it is required to first set up a virtual environment using `venv`, `uv`, or `conda` (preferred). This ensures that (a) one of the required versions of Python (3.11+) is used and (b) there are no dependency conflicts. It is recommended that your virtual environment be located on a drive with sufficient disk space, which is especially important when working on remote HPC clusters where storage space is often limited. 

<blockquote style="color:#000000; background:#ffffff; border-left:4px solid #7b53d7; padding-left:12px; margin-bottom: 20;">
🚧 <strong>crystALL</strong>'s full implementation and features consist of a lightweight Python package that requires minimal disk space and dependencies. However, the optional use of external calculator adapters (e.g., ALIGNN, MACE, ORB, MatterSim, etc.) to expand its functionalities may require significant disk space to download and store models, as some individual adapters exceed 2 GB each. Also, note that while <strong>crystALL</strong> does not require GPU-enabled systems, some adapters may benefit from or require a compatible GPU depending on the specific adapter's implementation.
</blockquote>


If you have `conda` installed on your system (see [`miniconda` install instructions](https://docs.conda.io/en/latest/miniconda.html)), you can create a new environment using:

``` shell
conda create -n crystall python=3.11
conda activate crystall
```

Then install **crystALL** (regular)
``` shell
pip install crystall
```

Or for editable install (development)
``` shell
pip install -e .
```

<blockquote style="color:#000000; background:#ffffff; border-left:4px solid #3d7d40; padding-left:12px; margin-bottom: 20;">
💡 <strong>HPC cluster users:</strong> When submitting a python script using <strong>crystALL</strong> (e.g., <code>crystall_script.py</code>) in a non-interactive job on a cluster, make sure that the conda installation path is included and activated in your job script. For instance:

``` bash
CONDA_INSTALL_DIR="/absolute/path/of/your/conda/installation"
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
conda activate crystall
python crystall_script.py
```
</blockquote>
