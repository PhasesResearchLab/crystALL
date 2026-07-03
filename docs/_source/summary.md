<div align="center">
  <img src="_static/crystALL.png" alt="crystALL Logo" width="180"/>
  <br>

  <br>
  <!-- Badges line 1 -->
  <a href="https://doi.org/10.xxxx/xxxxx"><img src="https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue"></a>
  <a href="https://arxiv.org/abs/YYMM.xxxxx"><img src="https://img.shields.io/badge/arXiv-YYMM.xxxxx-darkred"/></a>
  <img src="https://img.shields.io/badge/Docs-readthedocs-informational"/>

  <br>
  <!-- Badges line 2 -->
  <img src="https://img.shields.io/badge/Windows-pass-brightgreen"/>
  <img src="https://img.shields.io/badge/Macos-pass-brightgreen"/>
  <img src="https://img.shields.io/badge/Linux-pass-brightgreen"/>
  <img src="https://img.shields.io/badge/Codecov-100%25-fe226d"/>

  <br>
  <!-- Badges line 3 -->
  <img src="https://img.shields.io/badge/python-3.11%2B-lightgray"/>
  <a href="https://pypi.org/project/crystALL"><img src="https://img.shields.io/badge/pypi-v0.5.0-purple"></a>
  <a href="https://anaconda.org/conda-forge/crystALL"><img src="https://img.shields.io/badge/condaforge-v0.5.0-yellow"></a>
</div>

# Summary

**crystALL** is an open-source data-driven framework for atomic structure prediction and materials discovery, designed to address the critical gap between computational predictions and experimental observations. Unlike methods focused solely on stable ground states, crystALL systematically embraces structural diversity and compositional uncertainty, enabling the discovery, modeling, and validation of both stable and metastable crystal configurations.

By leveraging millions of relaxed and experimental structures from the MPDD database and OPTIMADE API, crystALL rapidly generates wide ensembles of candidate configurations for target compositions or compositional ranges. These candidates undergo atomic substitution, advanced machine learning-based featurization, clustering, and multi-model property prediction, supporting robust selection and direct ab initio or experimental validation.

crystALL uniquely supports searches accommodating experimental uncertainty and complex materials systems, generating unbiased, ensemble-focused predictions suitable for high-throughput candidate selection, statistical modeling, and validation against experiment or DFT.