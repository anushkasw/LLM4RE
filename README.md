# LLM4RE: A Data-centric Feasibility Study for Relation Extraction

This repository contains code associated with the paper ["LLM4RE: A Data-centric Feasibility Study for Relation Extraction"](https://aclanthology.org/2025.coling-main.447/) published in the Proceedings of the 31st International Conference on Computational Linguistics.

## Repository Structure
```bash
LLM4RE/
├── evaluations/         # Traditional and GenRES evaluation metric calculation
├── prompts/             # Prompts used for the pipeline
├── src_jre/             # Source code for joint relation extraction
├── src_re/              # Source code for relation classification
└── requirements.txt     # Python dependencies
```

## Installation
To clone this repository and set up the environment:

```bash
git clone https://github.com/anushkasw/LLM4RE.git
cd LLM4RE
pip install -r requirements.txt
```

## Getting Started
- Model Inference: Use the main.py file in the src_jre and src_re folders to make inference for joint relation extraction and relation classification.
```bash
python ./src_re/main.py \
--task "NYT10" \
-d "knn" \
-m "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--k "5" \
--prompt "open" \
-dir <Data Directory> \
-out <Output Directory> \
--prompt_dir ./prompts

```
- Evaluation: This folder contains code to extract traditional and GenRES metrics. The CS, US, and TS metric calculation code has been derived from the [GenRES](https://github.com/pat-jj/GenRES) repository which is based this [paper](https://aclanthology.org/2024.naacl-long.155/)

### Cite
If you plan on using the code please cite the paper:

```
@inproceedings{swarup-etal-2025-llm4re,
    title = "{LLM}4{RE}: A Data-centric Feasibility Study for Relation Extraction",
    author = "Swarup, Anushka  and
      Pan, Tianyu  and
      Wilson, Ronald  and
      Bhandarkar, Avanti  and
      Woodard, Damon",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.447/",
    pages = "6670--6691",
    abstract = "Relation Extraction (RE) is a multi-task process that is a crucial part of all information extraction pipelines. With the introduction of the generative language models, Large Language Models (LLMs) have showcased significant performance boosts for complex natural language processing and understanding tasks. Recent research in RE has also started incorporating these advanced machines in their pipelines. However, the full extent of the LLM`s potential for extracting relations remains unknown. Consequently, this study aims to conduct the first feasibility analysis to explore the viability of LLMs for RE by investigating their robustness to various complex RE scenarios stemming from data-specific characteristics. By conducting an exhaustive analysis of five state-of-the-art LLMs backed by more than 2100 experiments, this study posits that LLMs are not robust enough to tackle complex data characteristics for RE, and additional research efforts focusing on investigating their behaviors at extracting relationships are needed. The source code for the evaluation pipeline can be found at https://aaig.ece.ufl.edu/projects/relation-extraction ."
}
```
