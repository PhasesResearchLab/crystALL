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

# Summary

- (2 sentences) Motivation for materials discovery
    - CALPHAD [@Olson2023]
    - Going beyond stability. It is not enough [@Zunger2019] because of the complexities involved in the synthesis of new materials. Phase coexistance can take properties to extreme, e.g. infinitely negative thermal expansion [@Liu2022].
- Typically done through combinatoric search based on protoypes and DFT
    - Proposing new structures for compounds is not a new idea based on geometrical constraints [@LeBail2005]
    - AFLOW Database [@Toher2018] built by expert-intuition driven population of AFLOW Prototype Library [@Mehl2016] with new chemistries in a systematic way
    - Similarity driven prediction of substitutions [@Wang2021]
    - Compositional complexity grows quickly with dimensionality, considering thhat substitutions [@Krajewski2024Nimplex]

- (3 sentences) Past deployment on the Nd-Bi system [@Im2022] and Figure 7a of [@Im2022] showing excellent ordering of the candidates when evaluated vs different DFT methods. Past deployment in Al-Fe [@Shang2021].

# Statement of need

(statement of need)

- Motivation slightly reiterating the summary with more focus on impact and need
- (**Rushi**, 3 sentences) *(Importance of Ru-Zn in the study of catalysts. Importance of KCl-LaCl3 system in the study of molten salts. How `crystALL` is continuously used in the lab to predict structures of unknown compounds and if the automated experience has been positive.)*
To enhance nylon manufacturing efficiency, Ru-based catalysts are recognized as highly effective catalysts in synthesizing the key chemical intermediate through the benzene partial hydrogenation reaction [@Zhang2013]. Achieving precise control over this process is important to ensure optimal outcomes. Improvement of hydrogenation selectivity involves designing diverse active site ensembles in intermetallic catalysts [@Dasgupta2022]. Using crystALL, we have explored over 165, 000 configurations of Ru-Zn intermetallic compounds, including Ru${_3}$Zn, RuZn, Ru${_4}$Zn${_9}$, RuZn${_3}$, Ru${_3}$Zn${_10}$, RuZn${_6}$, RuZn${_10}$, and RuZn${_13}$. We are currently examining their structures and stability through pySIPFENN [@Krajewski2022], while simultaneously evaluating active site ensembles to design novel catalysts with high activity and selectivity. In the journey towards clean energy and carbon footprint reduction, the development of Molten Salt Reactors (MSRs) is essential, which requires high reliability and efficiency. crystALL serves as a valuable tool for identifying structures for complex molten salts, particularly in cases of limited or conflicting experimental data. For example, crystALL facilitated the prediction of structures for KLa${_3}$Cl${_10}$ and K${_3}$La${_5}$Cl${_18}$ compounds in the KCl-LaCl${_3}$ system, thereby validating their stability reported in various literature sources [@Seifert1985] [@Song1995].
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