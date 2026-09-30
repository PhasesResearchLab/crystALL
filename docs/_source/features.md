# Features

_This is just a quick introduction to the main features and workflow of crystALL. Please refer to the `crystALL` journal article for a complete, yet still concise, description of the framework, motivation behind it, and the underlying methodology._

### Overview

**On the high-level, `crystALL` allows you to take a chemical composition/s of interest, whether you measured it experimentally or hypothesized it, and then systematically explore them with minimal bias to form an approximately ergodic ensemble of candidates by feature space analysis and evaluation over any metric of interest to fulfill your research objectives.** This description is not vague, but rather emphasizes the framework's ability to fit different needs. These can range from [discovering crystal structure of previously unexplained experimental observations](https://doi.org/10.1016/j.actamat.2021.117448) to high-throughput screening for novel materials exhibiting certain topological characteristics (ongoing work). Overall, `crystALL` is a great fit for both experimentalists and computational researchers aiming to accelerate discovery of novel materials in underexplored regions of chemical space or cases escaping the generative learning or conventional computational approaches focusing only on the most probable answers.

In a _typical_ workflow, depicted in the figure below for a single target composition ("compound"), `crystALL` systematically generates, featurizes, and evaluates candidate crystal structures to identify the most promising configurations. It starts by gathering prototype structures (provided via MPDD / OPTIMADE API or local files) and performing all permutations of substitution into the target composition (orange). Then, it featurizes the candidates with `pySIPFENN`'s (python toolset for Structure-Informed Property and Feature Engineering with Neural Networks) `KS2022` (by default) to then analyze the embeddings (blue) while predicting properties of interest with ML models from pySIPFENN and community (purple). Filtering and clustering in the feature space are used to select top candidates from unique structures/energy basins (green), and final DFT relaxations leading to match with experimental observations (red). 

The example depicts a dramatic filtering cascade - 5M structures from MPDD are harnessed to generate around 50k ABCD3 candidates, then ML-filtered to 5k in 500 clusters, culminating in DFT calculations for the top 50 cluster representatives to construct an approximately ergodic ensemble and match experimental observations; while feedback loop to MPDD creates a self-improving ecosystem for materials discovery that grows with `crystALL` use.

<div align="center">
  <img src="https://raw.githubusercontent.com/PhasesResearchLab/crystALL/2cc27a91b6396b8cc776075b3696411f061b268c/docs/_static/crystALL_Compounds_V7.png" alt="Main figure for compound search" width="750"/>
</div>

**One of the unique features within `crystALL`, mentioned in the summary and explored in detail in the paper, is the ability to perform searches over compositional ranges, positioning it as an invaluable tool for interpreting real measurements of compositions, without assuming that molar ratios are low integers** - a very common pitfall researchers fall into. As illustrated below, **when combined with `MPDD`, `crystALL` accepts an uncertainty range for each element individually and systematically explores all prototypical structures within the specified compositional subspace, covering both low- and high-index anonymous formulas.

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
