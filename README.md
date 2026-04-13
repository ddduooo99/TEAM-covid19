# TEAM-covid19

This repository contains the implementation of **TEAM**, a Time-Enhanced Attention-based Mutation prediction framework for SARS-CoV-2 viral sequences. The model combines phylogenetic temporal sampling and protein language model embeddings to enable mutation prediction along viral evolution paths. 

## Repository Structure

```
TEAM-covid19/
├── data/ # Processed mutation data and lineage annotations (e.g., HA/NA sequences, GTF files, lineage notes)
├── embedding/ # ESM embedding pipeline
├── prediction/ # Prediction model and evaluation scripts
├── sampling/ # Time-enhanced phylogenetic sampling code
└── .gitignore
```
