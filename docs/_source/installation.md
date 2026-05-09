# Installation

Installing **crystALL** is simple and easy by utilizing **PyPI** package repository, **conda-forge**  package repository, or by cloning from **GitHub** directly. While not required, it is recommended to first set up a virtual environment using `venv`, `uv`, `micromamba` or `conda` (preferred). This ensures that (a) one of the required versions of Python (3.11+) is used and (b) there are no dependency conflicts. If you have Conda installed on your system (see [`miniconda` install instructions](https://docs.conda.io/en/latest/miniconda.html)), you can create a new environment using:

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

<blockquote style="color:#000000; background:#ffffff; border-left:4px solid #8e5e16; padding-left:12px; margin-bottom: 20;">
<strong>Linux users:</strong>  First install DGL for torch >=2.3

``` shell
python -m pip install "dgl==2.2.0" -f https://data.dgl.ai/wheels/torch-2.3/repo.html
pip install crystall
```
</blockquote>