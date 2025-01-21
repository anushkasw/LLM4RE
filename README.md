# LLM4RE: A Data-centric Feasibility Study for Relation Extraction

This repository contains code associated with the paper ["LLM4RE: A Data-centric Feasibility Study for Relation Extraction"]([https://ieeexplore.ieee.org/document/10747504](https://aclanthology.org/2025.coling-main.447/)) published in the Proceedings of the 31st International Conference on Computational Linguistics.

## Repository Structure
```bash
LLM4RE/
├── evaluations/             # Traditional and GenRES evaluation metric calculation
├── prompts/                # Prompts used for the pipeline
├── src_jre/                # Source code for evaluation joint relation extractors
├── src_re/              # Source code for evaluation joint relation classifiers
└── requirements.txt     # Python dependencies
```

## Installation
To clone this repository and set up the environment:

```bash
git clone https://github.com/anushkasw/LLM4RE.git
cd LLM4RE
pip install -r requirements.txt
```
Ensure you have Python 3.8+ and a CUDA-compatible GPU for best performance.

## Getting Started
### Data Preparation: Place your datasets in the data/ folder. Follow the format guidelines in docs/data_format.md.
Model Training: Run scripts/train_model.py to train the relation extraction model:
```bash
python scripts/train_model.py --config configs/default.yaml
```
### Evaluation: Use scripts/evaluate.py to test the model on benchmark datasets:
```bash
python scripts/evaluate.py --model checkpoints/best_model.pth
```
### Visualization: Generate knowledge graphs using the visualization/ scripts:
```bash
python visualization/graph_builder.py --input data/sample_relations.csv
```

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
