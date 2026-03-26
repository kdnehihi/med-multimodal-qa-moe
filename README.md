# medical-moe-assistant

Research scaffold for a lightweight multimodal medical QA project with a two-stage MoE-inspired architecture.

## Overview

This project studies a compact medical assistant that supports three QA settings:

- `symptom`: clinical image + question -> answer
- `xray`: radiology image + question -> answer
- `qa`: text-only medical question -> answer

The repository is intentionally research-oriented and lightweight. The current codebase focuses on:

- clean data preparation
- reproducible dataset conversion
- a minimal model scaffold with clear extension points

It is not a production medical system.

## Current Data Setup

The project currently uses three sources:

1. `HealMQA`
   - task: `symptom`
   - local dataset under `data/raw/share healmqa/`
   - processed output:
     - `data/processed/healmqa_500.jsonl`
     - `data/processed/healmqa_500_images/`

2. `VQA-RAD`
   - task: `xray`
   - loaded from Hugging Face: `flaviagiammarino/vqa-rad`
   - processed output:
     - `data/processed/vqa_rad_500.jsonl`
     - `data/processed/vqa_rad_500_images/`

3. `MedQuAD`
   - task: `qa`
   - loaded from Hugging Face: `Tonic/medquad`
   - processed output:
     - `data/processed/medquad_500.jsonl`

After preparation, the three datasets can be merged and shuffled into:

- `data/processed/train_merged.jsonl`

This merged file keeps only the three core fields:

- `image`
- `question`
- `answer`

## Data Format

Each processed sample follows:

```json
{
  "image": "path_or_none",
  "question": "What is shown in this image?",
  "answer": "Free-text answer"
}
```

For text-only QA, `image` is `null`.

## MoE Direction

The current project direction is a two-stage MoE-inspired design:

### Stage 1: Soft visual routing

For image-question samples:

- the image is encoded by two parallel image encoders
- the question is encoded by a shared text encoder
- a soft router uses the question embedding to assign weights over the two visual embeddings
- the weighted visual feature is fused with the question feature

For text-only samples:

- the first-stage visual routing is skipped
- the shared text embedding is used directly

### Stage 2: Hard expert routing

After a shared representation is obtained:

- a hard router selects exactly one downstream expert
- the selected expert performs task-specific reasoning before answer generation

This design aims to balance:

- adaptability across different image types
- specialization across tasks
- parameter efficiency for low-resource training

## Repository Structure

```text
medical-moe-assistant/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── prepare_healmqa.py
│   ├── prepare_medquad.py
│   ├── prepare_textvqa_medical_subset.py
│   ├── merge_datasets.py
│   ├── preview_data.py
│   └── train.py
├── models/
├── datasets/
├── trainers/
├── utils/
└── notebooks/
```

Note:

- `prepare_textvqa_medical_subset.py` now reuses the old script path but currently prepares a `VQA-RAD` subset.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation Workflow

### 1. Prepare HealMQA

```bash
python scripts/prepare_healmqa.py
```

### 2. Prepare VQA-RAD

```bash
python scripts/prepare_textvqa_medical_subset.py
```

Or use multiple splits:

```bash
python scripts/prepare_textvqa_medical_subset.py --splits train,test
```

### 3. Prepare MedQuAD

```bash
python scripts/prepare_medquad.py
```

### 4. Merge and shuffle all datasets

```bash
python scripts/merge_datasets.py
```

## Previewing Data

Preview processed files:

```bash
python scripts/preview_data.py \
  --source jsonl \
  --jsonl-path data/processed/healmqa_500.jsonl \
  --limit 5
```

Preview the merged dataset:

```bash
python scripts/preview_data.py \
  --source jsonl \
  --jsonl-path data/processed/train_merged.jsonl \
  --limit 10
```

## Training

The current training code is still a scaffold. The intended entry point is:

```bash
python scripts/train.py --config config/config.yaml
```

Model and trainer modules intentionally contain TODO comments for:

- image encoder replacement
- routing logic
- multimodal fusion
- expert selection
- generation and optimization

## Notes

- This repository is still a scaffold, not a finished implementation.
- Data preparation is more complete than model training at the moment.
- The main goal is to keep the code readable, modular, and easy to extend for experiments.
