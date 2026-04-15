# TEAM-covid19: Implementation and Usage Guide

This repository contains the official implementation of **TEAM**, a Time-Enhanced Attention-based Mutation prediction framework. The pipeline is modularized into three core stages: **Sequence Sampling**, **Protein Embedding**, and **Mutation Prediction**.

## Quick Start Instructions

### 1. Prerequisites & Data Setup
First, clone the repository and install the required dependencies. Due to file size limits, please download the processed data from the link below and place it into the `data/` directory.

* **Data Download:** Large data files (ESM embeddings, sequence datasets, metadata) are hosted on Google Drive due to GitHub size limits: [Download Data](https://drive.google.com/drive/folders/1uuf5kojb8FYWXk0x-6X0ph6csJ6QmD6N?usp=sharing)
* **Environment:** Python 3.8+ and PyTorch recommended.

---

### 2. Step-by-Step Execution

#### **Stage I: Sequence Sampling**
The sampling module processes raw evolution paths and generates training/testing pairs.
* **Input:** `.fasta` files (located in `data/Processed/Covid19/`).
* **Action:** Run `Sampling.py` to generate 3gram-protvec sets and required files for ESM embedding.
* **Output:** 
    1. **3gram-protvec datasets:** Training and testing sets using trigram encoding.
    2. **ESM & Site Encoding files:** 
        * `esm_input_train.fasta`: Input for `ESMembedding.py`.
        * `key2keys_train.json`: Mapping file for `ESMsite.py` to filter site-specific embeddings.

#### **Stage II: Protein Embedding**
This stage generates high-dimensional biological representations using the ESM language model.
* Use `ESMembedding.py` to generate Raw Embeddings (uses esm_input_*.fasta from Stage I)
* Use `ESMsite.py` to filter Site-specific Embeddings (uses key2keys_*.json)

#### **Stage III: Mutation Prediction**
The final stage performs mutation prediction.
* Configuration: Before running, you can modify `isletter` or classification types within the script to toggle between binary and multi-class tasks.
* Execution: Run the training script and redirect logs to a text file for analysis.


## Repository Structure

```
TEAM-covid19/
├── data/ # Processed mutation data and lineage annotations (e.g., SARS-CoV-2 and HA/NA sequences, GTF files, lineage notes)
├── embedding/ # ESM embedding pipeline
├── prediction/ # Prediction model and evaluation scripts
├── Sampling.py # Time-enhanced phylogenetic sampling code
└── .gitignore
```


