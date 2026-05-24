# Master_thesis_2026

This repository contains the coode used for the experiments in the master`s thesis *Evaluating Safety Alignment and Mitigation in Norwegian Language Models*. The project investigates safety behaviour in Norwegian large language models and evaluates whether system prompting and supervised fine-tuned can reduce harmful model outputs. 

The experiments are based on the TryggLLM dataset, which contains harmful Norwegian prompts and safety-aligned target responses. The study evaluated two Norwegian instruction-tuned language models: 

- NorMistral-7B-warm-instruct
- NorwAI-Mistral-7B-Instruct

The models are evaluated in three different settings:
1. Base model
2. System-prompted model
3. Fine-tuned model

The main goal of the project is to evaluate how Norwegian language models respond to harmful prompts, and whether mitigation strategies can improve their safety behaviour.

The project consists of the following main parts: 
- Data preparation and dataset splitting
- Basline response generation
- System prompting experiments
- Training and evaluating an automatic safety classificator
- Evaluation with ROUGE-L and BERTScore
- Addition evaluation of harmless prompts to examine over-refusal 

## Safety notice 
This project involves safety evaluation of large language models using harmful prompts. Some generated model outputs may contain unsafe or sensitive content. To avoid sharing potentially harmful information, raw generated outputs are not included in this repository. 

## Repository structure 

.
├── Notebooks/
│   ├── 01_prepare_data.ipynb
│   ├── 02_system_prompting.ipynb
│   ├── 03_grid_search.ipynb
│   ├── 04_fine_tuning.ipynb
│   ├── 05_build_classifier.ipynb
│   ├── 06_generate.ipynb
│   └── 07_evaluate.ipynb
│
├── utils_evaluation.py
├── utils_formatting.py
├── utils_generation.py
└── README.md
