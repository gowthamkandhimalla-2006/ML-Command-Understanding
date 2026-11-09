# ML Command Understanding

## Project Overview

ML Command Understanding is a machine learning project designed to understand natural-language commands and convert them into structured JSON outputs. The project uses the Pythia-70M language model and a prepared command dataset for training and experimentation.

## GitHub Repository

**Project Source Code:**  
https://github.com/gowthamkandhimalla-2006/ML-Command-Understanding

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Pythia-70M
- JSONL Dataset

## Project Directory Structure

```text
ML-Command-Understanding/
│
├── training/
│   ├── train.py
│   ├── config.yaml
│   └── loss.csv
│
├── models/
│   └── v1/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── generation_config.json
│       └── training_args.bin
│
├── README.md
└── requirements.txt
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Status

The training script completed and model artifacts were saved. The model and tokenizer loaded successfully. However, training stability and prediction quality still require further testing and validation.

## Submission Details

- Project: ML Command Understanding
- Model: Pythia-70M
- Repository: https://github.com/gowthamkandhimalla-2006/ML-Command-Understanding
