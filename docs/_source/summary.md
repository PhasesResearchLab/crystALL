<div align="center">
<table>
  <tr>
    <td width="160">
      <img src="_static/crystALL.png" alt="crystALL Logo" width="140"/>
    </td>
    <td style="vertical-align: middle;">
      <!-- Row 1: Publication, Docs & License -->
      <a href="https://doi.org/10.xxxx/xxxxx"><img src="https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue" alt="DOI"/></a>
      <a href="https://arxiv.org/abs/YYMM.xxxxx"><img src="https://img.shields.io/badge/arXiv-YYMM.xxxxx-darkred" alt="arXiv"/></a>
      <img src="https://img.shields.io/badge/Docs-readthedocs-informational" alt="Docs"/>
      <br>
      <!-- Row 2: Live CI / OS Tests -->
      <a href="https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_LinuxUbuntu.yaml"><img src="https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_LinuxUbuntu.yaml/badge.svg" alt="Linux"/></a>
      <a href="https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_Windows.yaml"><img src="https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_Windows.yaml/badge.svg" alt="Windows"/></a>
      <a href="https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_Mac.yaml"><img src="https://github.com/PhasesResearchLab/crystALL/actions/workflows/coreTests_Mac.yaml/badge.svg" alt="macOS"/></a>
      <img src="https://img.shields.io/badge/Codecov-100%25-fe226d" alt="Codecov"/>
      <br>
      <!-- Row 3: Environment & Release -->
      <a href="https://github.com/PhasesResearchLab/crystALL"><img src="https://img.shields.io/github/languages/top/PhasesResearchLab/crystALL" alt="GitHub top language"/></a>
      <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
      <a href="https://pypi.org/project/crystall-search"><img src="https://img.shields.io/badge/pypi-v0.5.2-purple" alt="PyPI"/></a>
    </td>
  </tr>
</table>
</div>
<br>

# Summary

**crystALL** is an open-source data-driven framework for atomic structure prediction and materials discovery, designed to address the critical gap between computational predictions and experimental observations. Unlike methods focused solely on stable ground states, crystALL systematically embraces structural diversity and compositional uncertainty, enabling the discovery, modeling, and validation of both stable and metastable crystal configurations, going beyond more common data-driven discovery approaches that prioritize only the lowest-energy structures often failing to explain experimental observations or propose structures stable under conditions of interest.

By leveraging millions of relaxed and experimental structures from the MPDD Database and OPTIMADE API, `crystALL` rapidly generates wide ensembles of candidate configurations for either target compositions or compositional ranges, with the latter being a unique strength of `crystALL`+`MPDD` combination. These candidates undergo all permutations of atomic substitution, advanced machine learning-based featurization, clustering, and multi-model property prediction, supporting robust selection and direct ab initio or experimental validation (in a typical use case).