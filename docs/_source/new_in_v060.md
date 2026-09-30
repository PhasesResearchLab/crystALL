# New in v0.6.0

The upcoming `v0.6.0` release introduces a fully modular architecture driven by the `MLIPAdapter` registry, enabling automated structure relaxations and multi-property predictions across diverse machine-learned interatomic potentials. Each potential manages its own execution micro-environment with support for CPU and CUDA acceleration, full cell/volume optimization, and optional piping into downstream static property models. Below is a summary of the machine-learned interatomic potentials integrated into `crystALL` v0.6.0 via the `MLIPAdapter` registry:

| Model | Architecture Backbone | Training Dataset | Hardware | Variants | Distribution | Citation |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **DeePMD** | Descriptor-based (Deep Potential) | MPtrj / DPA-4.0 | CPU, CUDA | 3 | From source | [Wang et al. (2018)](https://doi.org/10.1016/j.cpc.2018.03.016) |
| **CHGNet** | Invariant GNN (charge-informed) | MPtrj | CPU, CUDA | 3 | Built in | [Deng et al. (2023)](https://doi.org/10.1038/s42256-023-00716-3) |
| **TensorNet** | Invariant Cartesian Tensor GNN | MatPES / QM9 | CPU, CUDA | 2 | Built in | [Simeon & de Fabritiis (2023)](https://arxiv.org/abs/2306.06482) |
| **ORB** | Invariant Graph Transformer | OMat24 / MPtrj | CPU, CUDA | 3 | Built in | [Rhodes et al. (2025)](https://arxiv.org/abs/2504.06231) |
| **ORB-D3** | Invariant Graph Transformer (+ D3) | OMat24 / MPtrj | CPU, CUDA | 2 | Built in | [Rhodes et al. (2025)](https://arxiv.org/abs/2504.06231) |
| **Allegro** | Equivariant GNN (strictly local) | OMat24 / OAM | CUDA | 3 | From source | [Musaelian et al. (2023)](https://doi.org/10.1038/s41467-023-36329-y) |
| **AlphaNet** | Equivariant GNN (local-frame) | Alexandria / OMat24 | CPU, CUDA | 3 | From source | [Yin et al. (2025)](https://arxiv.org/abs/2501.12932) |
| **Eqnorm** | Equivariant GNN (normalized) | MPtrj / OMat24 | CPU, CUDA | 3 | From source | [Chen et al. (2025)](https://arxiv.org/abs/2502.04356) |
| **eSEN** | Equivariant GNN (spherical-channel) | OMat24 | CPU, CUDA | 2 | From source | [Fu et al. (2025)](https://arxiv.org/abs/2502.12147) |
| **MACE** | Higher-Order Equivariant GNN (Multi-ACE) | MPtrj / OMat24 | CPU, CUDA | 6 | Built in | [Batatia et al. (2022)](https://arxiv.org/abs/2206.00133) |
| **MatterSim** | Equivariant GNN (deep learning potential) | MatterSim-DB | CPU, CUDA | 2 | Built in | [Yang et al. (2024)](https://arxiv.org/abs/2405.04967) |
| **NequIP** | Equivariant GNN ($E(3)$-equivariant) | OMat24 | CUDA | 3 | Built in | [Batzner et al. (2022)](https://doi.org/10.1038/s41467-022-29939-5) |
| **Nequix** | Equivariant GNN (compact) | OMat24 | CPU, CUDA | 2 | From source | [Koker et al. (2025)](https://arxiv.org/abs/2502.13840) |
| **PolarMACE** | Polarisable Equivariant GNN | SPICE / QM9 | CPU, CUDA | 3 | From source | Batatia et al. (2026) |
| **QET** | Equivariant GNN (charge-equilibrated) | MatPES / MatQ | CPU, CUDA | 2 | From source | [Ko et al. (2025)](https://arxiv.org/abs/2511.07249) |
| **SevenNet** | Equivariant GNN (parallel) | OMat24 / MPtrj | CPU, CUDA | 3 | Built in | [Park et al. (2024)](https://doi.org/10.1021/acs.jctc.4c00190) |
| **UPET** | Equivariant GNN (point equivariant transformer) | OMat24 / PET-MAD | CPU, CUDA | 3 | Built in | [Bigi et al. (2026)](https://arxiv.org/abs/2601.16195) |
| **EquiformerV3** | Equivariant Graph Transformer | OMat24 / MPtrj | CPU, CUDA | 2 | From source | Liao et al. (2026) |
| **MatRIS** | Equivariant Graph Transformer | OMat24 | CPU, CUDA | 2 | From source | [Zhou et al. (2025)](https://arxiv.org/abs/2502.09506) |
| **UMA** | Equivariant Graph Transformer (universal model) | OMat24 | CPU, CUDA | 3 | Built in | [Wood et al. (2025)](https://arxiv.org/abs/2506.23971) |
| **GRACE** | Graph Atomic Cluster Expansion | OMat24 | CPU, CUDA | 3 | Built in | [Bochkarev et al. (2024)](https://doi.org/10.1103/PhysRevX.14.021036) |
| **TACE** | Tensor Atomic Cluster Expansion | OMat24 | CPU, CUDA | 3 | From source | [Xu et al. (2025)](https://arxiv.org/abs/2501.18967) |
| **HIENet** | Hybrid Invariant–Equivariant GNN | MPtrj / OMat24 | CPU, CUDA | 2 | From source | [Yan et al. (2025)](https://arxiv.org/abs/2502.01584) |
| **OSAKA** | Fine-tuned Equivariant GNN (actinide-extended) | Literature Heavy-Element | CPU, CUDA | 3 | Built in | Kuroda et al. (2026) |
| **Total** | **24 Architectures** | | | **66** | | |
