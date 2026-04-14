# TEAM-covid19

This repository contains the implementation of **TEAM**, a Time-Enhanced Attention-based Mutation prediction framework for SARS-CoV-2 viral sequences. The model combines phylogenetic temporal sampling and protein language model embeddings to enable mutation prediction along viral evolution paths. 

## Data
Large data files (ESM embeddings, sequence datasets, metadata) are hosted on Google Drive due to GitHub size limits:
[Download Data](https://drive.google.com/drive/folders/1uuf5kojb8FYWXk0x-6X0ph6csJ6QmD6N?usp=sharing)

## Repository Structure

```
TEAM-covid19/
├── data/ # Processed mutation data and lineage annotations (e.g., HA/NA sequences, GTF files, lineage notes)
├── embedding/ # ESM embedding pipeline
├── prediction/ # Prediction model and evaluation scripts
├── sampling/ # Time-enhanced phylogenetic sampling code
└── .gitignore
```
