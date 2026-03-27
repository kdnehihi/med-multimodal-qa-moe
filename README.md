# medical-moe-assistant

Lightweight multimodal medical QA research repo with a runnable baseline and a two-stage MoE-inspired direction.

## Overview

This project studies a compact medical assistant that supports three QA settings:

- `symptom`: clinical image + question -> answer
- `xray`: radiology image + question -> answer
- `qa`: text-only medical question -> answer

The repository is intentionally research-oriented and lightweight. The current codebase now includes:

- clean and reproducible data preparation
- a merged multi-task training set
- a runnable baseline using `BiomedCLIP + FLAN-T5-base`
- utilities for checkpointing and qualitative prediction preview

It is not a production medical system.

## Current Data Setup

The project currently uses three sources:

1. `HealMQA`
   - task: `symptom`
   - local dataset under `data/raw/share healmqa/`
   - processed output:
     - `data/processed/healmqa_1500.jsonl`
     - `data/processed/healmqa_500_images/`

2. `VQA-RAD`
   - task: `xray`
   - loaded from Hugging Face: `flaviagiammarino/vqa-rad`
   - processed output:
     - `data/processed/vqa_rad_1500.jsonl`
     - `data/processed/vqa_rad_1500_images/`

3. `MedQuAD`
   - task: `qa`
   - loaded from Hugging Face: `Tonic/medquad`
   - processed output:
     - `data/processed/medquad_2000.jsonl`

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

Optional: augment each HealMQA image with two extra independent QA pairs:

```bash
python scripts/augment_healmqa.py
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
  --jsonl-path data/processed/healmqa_1500.jsonl \
  --limit 5
```

Preview the merged dataset:

```bash
python scripts/preview_data.py \
  --source jsonl \
  --jsonl-path data/processed/train_merged.jsonl \
  --limit 10
```

## Baseline Training

The current runnable baseline is:

- vision encoder: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- text model: `google/flan-t5-base`
- fusion: one projected visual prefix token prepended to the text stream
- training strategy:
  - freeze the vision encoder
  - freeze the T5 encoder
  - train the decoder and lightweight fusion layers

Default config is tuned for a relatively safe local run on Apple Silicon `mps`.

Train with:

```bash
python scripts/train.py --config config/config.yaml
```

Key defaults:

- `batch_size: 2`
- `gradient_accumulation_steps: 4`
- `epochs: 1`
- `max_train_steps: 300`
- `device: mps`

Checkpoints are written under:

- `outputs/checkpoints/baseline_biomedclip_flan_t5_base/`

## Prediction Preview

After training, preview a few validation predictions with the best checkpoint:

```bash
python scripts/predict_samples.py --config config/config.yaml
```

This prints:

- image path
- question
- expected answer
- predicted answer

and saves a JSONL preview under:

- `outputs/predictions/baseline_preview_predictions.jsonl`

## Notes

- The baseline is intentionally simple so it can serve as a fair comparison point for the later MoE model.
- The proposed MoE architecture has not been implemented yet; the current repo establishes the data pipeline and baseline training loop first.
- The main goal is to keep the code readable, modular, and easy to extend for experiments.
