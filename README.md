# medical-moe-assistant

Minimal research scaffold for a lightweight multimodal medical QA project with a simplified MoE-inspired design.

## Project Idea

This repository is a clean starter codebase for a medical QA system that can support:

- `symptom`: image + question -> free-text answer
- `text_image`: image containing text -> free-text answer
- `qa`: text-only medical question -> free-text answer

The current repository is intentionally a scaffold, not a finished implementation. Most modules contain clear TODO comments showing where model logic, routing, fusion, tokenization, and training should be added later.

## Architecture Overview

Text-only architecture:

1. `VisionEncoder`
   - shared visual backbone placeholder
2. `VisionMoE`
   - two virtual vision experts:
     - `natural_image_head`
     - `text_image_head`
3. `Router`
   - predicts:
     - vision weights of size 2
     - adapter weights of size 3
4. `LanguageMoE`
   - placeholder language backbone with adapter slots
5. `MedicalMoEModel`
   - ties together instruction embedding, routing, fusion, and generation

## Data Format

Each JSONL sample should look like:

```json
{
  "image": "path_or_none",
  "question": "What is shown in this image?",
  "answer": "Free-text answer",
  "task_type": "symptom"
}
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run placeholder scripts:

```bash
python scripts/download_data.py
python scripts/preprocess.py --input data/raw/your_data.jsonl
python scripts/train.py --config config/config.yaml
```

Prepare HealMQA symptom-image QA data:

```bash
python scripts/prepare_healmqa.py
```

Prepare text-only medical QA data:

```bash
python scripts/prepare_medquad.py
```

Preview raw TextVQA samples:

```bash
python scripts/preview_data.py --source hf --splits train --limit 5
```

Preview exported medical subset samples:

```bash
python scripts/preview_data.py --source jsonl --jsonl-path data/processed/textvqa_medical_500.jsonl --limit 5
```

Prepare a medical subset from VQA-RAD:

```bash
python scripts/prepare_textvqa_medical_subset.py --config config/config.yaml
```

This script is intended to:

- load VQA-RAD directly from Hugging Face `datasets`
- sample medical radiology VQA examples
- normalize them into `image / question / answer / task_type`
- save selected images locally under `data/processed/`

Use `train` only by default, or include test too:

```bash
python scripts/prepare_textvqa_medical_subset.py --splits train,test
```

## Notes

- This project is a scaffold, not a finished system.
- The code is intentionally minimal and incomplete.
- TODO comments explain expected inputs, outputs, and extension points.
