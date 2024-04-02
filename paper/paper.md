---
title: 'crystALL: Purely data-driven automated structure prediction for unidentified chemical compounds'
tags:
  - Python
  - materials science
  - structure prediction
  - machine learning
  - materials discovery
authors:
  - name: Adam M. Krajewski
    orcid: 0000-0002-2266-0099
    affiliation: 1
    corresponding: true
  - name: Ricardo Amaral
    orcid: 0000-0003-1070-9964
    affiliation: 1
  - name: Jonathan W. Siegel
    orcid: 0000-0002-1493-4889
    affiliation: 2
  - name: Rushi Gong
    orcid: 0009-0004-0543-581X
    affiliation: 1
  - name: ShunLi Shang
    orcid: 0000-0002-6524-8897
    affiliation: 1
  - name: Zi-Kui Liu
    orcid: 0000-0003-3346-3696
    affiliation: 1
affiliations:
  - name: Department of Materials Science and Engineering, The Pennsylvania State University, USA
    index: 1
  - name: Department of Mathematics, Texas A&M University, USA
    index: 2
date: March 31st 2024
#bibliography: [paper/paper.bib]
bibliography: paper.bib
---

# Statement of need

The ability to determine atomic structure of existing or hypothetical solids formed by chemical elements is one of the fundamental challanges in theoritical crystallography and in materials science as a whole. For the last several decades [@Hawthorne1994], the two remaining challenges are predictions of (a) compositions forming compounds of interest and (b) candidate topologies or approximate atomic arrangements they can form. Given these, the exact structure can be solved experimentally, e.g., by matching X-ray diffraction profile to candidates using method by [@LeBail1988] and interpreting the profile, or by similar investigation of the Raman spectrum [@Ferrari2013]. For hypothetical compounds in their pure quantum states, the exact structure can also be solved using ab initio quantum mechanical methods, such as Kohn-Sham density functional theory (DFT) [@Kohn1965], which are computationally expensive but can provide accurate results and continue to improve [@Kothakonda2023].

The latter problem of proposing the topologies has been traditionally approached by finding geometries satisfying constraints, like the principles systematized by [@Pauling1929], which included then-novel concepts like coördination. Over decades, this has evolved into automated computational tools like the `GRINSP` by [@LeBail2005] which can propose such structures. More recently, increasing computational power shifted focus towards testing many options, with some of the methods starting from random arrangements and collapsing them into local minima, like `AIRSS` by [@Pickard2011], or by evolutionary techniques, like `USPEX` [@Glass2006]. 

Several efforts have been made to tackle it simultaneously with the composition problem, such as high-throughput construction of the `AFLOW` [@Curtarolo2012] DFT Database [@Toher2018] (and others, see [#Methods] Section) built by sustematicly populating the earlier, yet continously growing, library of prototype structures [@Mehl2016] with new chemistries based on the expert knowledge. However, even for a limited set of elements, this problem becomes combinatorically challenging (see [@Krajewski2024Nimplex]). This prompted recent efforts into (1) systematic similarity-driven prediction of substitutions which would result in finding new low-energy structures [@Wang2021], (2) into brute-force substitutions filtered by machine learning models [@Schmidt2023], and (3) the combination thereof [@Merchant2023] leading to rapid discovery of new structures stable at 0K temperature and 0Pa pressure against competing arrangements.

While finding such $T=0$K $P=0$ hypothetical structures certainly opens the world for new discoveries, it warrants some key questions. First, *To what degree have we biased the discovery of new structures?* through many steps of biasing the search to our expectations, which is philosophically analogous to how retraining Large Language Models (LLMs) and other machine learnig (ML) models on generated data underestimate improbable events causing, so called, *Model Collapse* [@Shumailov2023], and drift from the real distribution of the data [@Alemohammad2023].

*Do we need diverse energy landscapes to find new structures?*


# Summary

- (2 sentences) Motivation for materials discovery
    - CALPHAD [@Olson2023]
    - Going beyond stability. It is not enough [@Zunger2019] because of the complexities involved in the synthesis of new materials. Phase coexistance can take properties to extreme, e.g. infinitely negative thermal expansion [@Liu2022]. Superconductors [@Ferreira2023]
- Typically done through combinatoric search based on protoypes and DFT
    - Similarity driven prediction of substitutions 
    - Compositional complexity grows quickly with dimensionality, considering thhat substitutions 

- (3 sentences) Past deployment on the Nd-Bi system [@Im2022] and Figure 7a of [@Im2022] showing excellent ordering of the candidates when evaluated vs different DFT methods. Past deployment in Al-Fe [@Shang2021].

- Motivation slightly reiterating the summary with more focus on impact and need
- (**Rushi**, 3 sentences) *(Importance of Ru-Zn in the study of catalysts. Importance of KCl-LaCl3 system in the study of molten salts. How `crystALL` is continuously used in the lab to predict structures of unknown compounds and if the automated experience has been positive.)*
To enhance nylon manufacturing efficiency, Ru-based catalysts are recognized as highly effective catalysts in synthesizing the key chemical intermediate through the benzene partial hydrogenation reaction [@Zhang2013]. Achieving precise control over this process is important to ensure optimal outcomes. Improvement of hydrogenation selectivity involves designing diverse active site ensembles in intermetallic catalysts [@Dasgupta2022]. Using crystALL, we have explored over 165, 000 configurations of Ru-Zn intermetallic compounds, including Ru${_3}$Zn, RuZn, Ru${_4}$Zn${_9}$, RuZn${_3}$, Ru${_3}$Zn$_{10}$, RuZn${_6}$, RuZn$_{10}$, and RuZn$_{13}$. We are currently examining their structures and stability through pySIPFENN [@Krajewski2022], while simultaneously evaluating active site ensembles to design novel catalysts with high activity and selectivity. In the journey towards clean energy and carbon footprint reduction, the development of Molten Salt Reactors (MSRs) is essential, which requires high reliability and efficiency. crystALL serves as a valuable tool for identifying structures for complex molten salts, particularly in cases of limited or conflicting experimental data. For example, crystALL facilitated the prediction of structures for KLa${_3}$Cl$_{10}$ and K${_3}$La${_5}$Cl$_{18}$ compounds in the KCl-LaCl${_3}$ system, thereby validating their stability reported in various literature sources [@Seifert1985] [@Song1995].
- (**Adam**, 1 sentence) Conclusion on the importance of automation for user tool.

# Methods

## MPDD Database

Contains:
- OQMD [@Shen2022]
- AFLOW Database [@Toher2018]
- Materials Project [@Jain2013]
- NIST-JARVIS [@Choudhary2020]
- COD [@Grazulis2019]
- GNoME [@Merchant2023]

API:
- native pymongo
- OPTIMADE [@Evans2024]

- use of `pymatgen` [@Ong2013] for data processing

## Predictive Modeling Pipeline

(about the pySIPFENN, MPDD-ALIGNN, and other models)

- pySIPFENN software [@Krajewski2022] and models on Zenodo [@Krajewski2020SIPFENNModels]
- ALIGNN [@Choudhary2021]

(schematic of crystALL software)

## Compound-Based Structure Prediction

(about the structure prediction)

![Compound-Based Structure Search. Reproduced from [@Im2022] under permission. \label{fig:compoundSearch}](assets/NdBi2_GraphicalAbstract_V4.png){width="300pt"}

## Composition-Based Structure Prediction

(uncertainty in compound composition extends the previous method)

![Composition-Based Structure Search. \label{fig:compositionalSearch}](assets/crystALL_composition_diagram_V3.png){width="300pt"}

# Acknowledgements

- (a loooot of grants over 5 years: NSF superalloys, PSU Seed, all from Nd-Bi study, DOE ARPA-E, NSF-POSE, ONR FGM)
- All Phases Research Lab members who tested the software in production
- All Phases Research Lab members who provided feedback on the software
