# BERTimentalAspect

BERTimentalAspect is an NLP project designed to perform Sentiment Orientation Extraction (SOE) and Aspect Term Extraction (ATE) using BERT, NER, and pair sentence classification techniques. This repository contains the source code and resources necessary to train and evaluate models for extracting sentiment and aspect terms from text data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Contributing](#contributing)
  
## Introduction

Sentiment analysis and aspect term extraction are critical components of understanding user opinions and feedback. BERTimentalAspect leverages BERT for efficient and accurate extraction of sentiment orientations and aspect terms from text. Named Entity Recognition (NER) is used for ATE, and pair sentence classification is applied for SOE.

## Features

- **Sentiment Orientation Extraction (SOE)**: Classify the sentiment orientation (positive, negative, neutral) of given text and aspect term.
- **Aspect Term Extraction (ATE)**: Identify and extract aspect terms related to the sentiment.
- **BERT-based Models**: Utilize the power of BERT for state-of-the-art performance.
- **NER Integration**: Incorporate Named Entity Recognition for precise aspect term extraction.
- **Pair Sentence Classification**: Improve sentiment orientation classification with advanced techniques.

## Installation

To install the necessary dependencies, clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/BERTimentalAspect.git
cd BERTimentalAspect
pip install -r requirements.txt
```

## Usage
To train and evaluate the models, follow these steps:

Prepare Datasets: Place your datasets in the data/ directory.
Train Models: Run the training script with the appropriate parameters.
```bash
python train.py \
  --model_name bert-base-uncased \
  --data_dir ./data \
  --sentence_level \
  --do_lower_case \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --epochs 5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --warmup_steps 500 \
  --save_total_limit 5 \
  --evaluation_strategy steps \
  --logging_strategy steps \
  --save_strategy steps
```
### Argument Descriptions
- --model_name (str, required): The name of the pretrained BERT model to use.
- --data_dir (str, required): Directory where the dataset is stored.
- --sentence_level (flag): Whether to perform sentence-level classification.
- --do_lower_case (flag): Whether to lowercase the input text (useful for uncased models).
- --batch_size (int, default=8): Batch size for training.
- --gradient_accumulation_steps (int, default=1): Number of updates steps to accumulate before performing a backward/update pass.
- --learning_rate (float, default=3e-5): Learning rate for the optimizer.
- --epochs (int, default=10): Number of training epochs.
- --weight_decay (float, default=0.1): Weight decay for the optimizer.
- --warmup_ratio (float, default=0.1): Warmup ratio for the learning rate scheduler.
- --warmup_steps (int, default=1000): Number of warmup steps for the learning rate scheduler.
- --save_total_limit (int, default=3): Maximum number of model checkpoints to keep.
- --evaluation_strategy (str, default='steps'): Evaluation strategy to use during training.
- --logging_strategy (str, default='steps'): Logging strategy to use during training.
- --save_strategy (str, default='steps'): Save strategy to use during training.
# Project Structure
```
BERTimentalAspect/
│
├── data/                 # Dataset files
├── scripts/              # Scripts for training and evaluation
│   ├── ATE/              # Aspect Term Extraction scripts
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── data_utils.py
│   │   │   ├── eval_utils.py
│   │   └── run_ate.py
│   └── SOE/              # Sentiment Orientation Extraction scripts
│       ├── utils/
│       │   ├── data_utils.py
│       │   ├── eval_utils.py
│       └── run_soe.py
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
```


# Datasets
Datasets should be placed in the data/ directory. The repository supports various datasets, and instructions for formatting and preprocessing are provided in the data/ section.

# Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss your proposed changes.

