# Features

**crystALL** provides a practical, science‑driven pipeline to generate, featurize, and prioritize candidate crystal structures for a target composition. At a high level the workflow gathers prototype structures (from local files or MPDD), computes consistent ML descriptors, applies one or more property predictors, and selects compact, experimentally actionable candidate sets.

Prototype generation and featurization are designed to be reproducible and lightweight: the code performs anonymized elemental substitutions and deduplication to build a controlled prototype ensemble (the default generation is intentionally constrained). Each prototype is then embedded into KS2022 descriptor space via pySIPFENN, yielding a uniform representation suitable for clustering and downstream models.

To illustrate the scoring step, the bar plot below shows formation energies sorted from lowest to highest for a focused ensemble of 100 NaMg(BH4)3 prototypes. This view helps rapidly identify low‑energy regions in the ensemble and informs which clusters deserve further attention.

<div align="center">
  <img src="_static/barplot.png" alt="Sorted bar chart of formation energies" width="750"/>
</div>

Property prediction in crystALL is modular. pySIPFENN models are available natively and external calculators plug in through adapter classes. ALIGNN and CHGNet are provided as examples of such adapters: ALIGNN exposes multi‑head formation energy outputs (MP and JARVIS) which we filter according to user request, and CHGNet reports static and relaxed energies computed independently so users can skip relaxation when it is not needed. The adapters accept an explicit list of objectives, enabling selective computation and reduced runtime for expensive predictors.

After featurization and property assignment, crystALL clusters the prototype ensemble (t‑SNE for embedding + KMeans/DBSCAN for grouping) and annotates clusters with objective statistics. The figure below shows the t‑SNE embedding for the same NaMg(BH4)3 set: seven clusters are recovered and correspond to distinct structural families (for example, clusters 1 and 2 are enriched in hexagonal motifs, cluster 3 in triclinic arrangements, clusters 4, 5 and 7 in trigonal lattices, and cluster 6 contains perovskite‑type geometries). This separation illustrates how descriptor‑based clustering can disentangle fundamentally different local chemistries and geometries.

<div align="center">
  <img src="_static/2dcluster.png" alt="2D t-SNE of prototypes colored by clusters" width="750"/>
</div>

The final selection step produces Top Candidates by selecting the lowest‑energy representative from each cluster for every available objective; these candidates provide a compact, diverse set for higher‑accuracy DFT relaxation or experimental validation. Outputs are exportable as CSV (`all-prototypes` and `top-candidates`), and an optional MPDD convex‑hull check (energy‑above‑hull) can be performed to further prioritize thermodynamically plausible entries.
