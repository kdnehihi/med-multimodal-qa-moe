"""Create more natural independent QA augmentations for HealMQA samples."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, load_config, load_jsonl, write_jsonl


DISEASE_TERMS = [
    "lipoma",
    "cyst",
    "eczema",
    "pink eye",
    "conjunctivitis",
    "rash",
    "infection",
    "fungal infection",
    "allergic reaction",
    "irritation",
    "stye",
    "acne",
    "wart",
    "boil",
    "cellulitis",
    "psoriasis",
]

TREATMENT_TERMS = [
    "treat",
    "treatment",
    "cream",
    "ointment",
    "drops",
    "compress",
    "antibiotic",
    "hydrocortisone",
    "ketoconazole",
    "wash",
    "hygiene",
    "medicine",
    "medication",
]

SEVERITY_TERMS = [
    "worry",
    "serious",
    "danger",
    "concern",
    "benign",
    "doctor",
    "consult",
    "checked",
    "not getting better",
    "painful",
    "changes in size",
]

CAUSE_TERMS = [
    "cause",
    "causing",
    "viral",
    "bacterial",
    "allergic",
    "infection",
    "related",
    "trigger",
]

SYMPTOM_TERMS = [
    "pain",
    "redness",
    "itch",
    "swelling",
    "symptom",
    "symptoms",
    "soft",
    "lump",
    "bump",
    "discomfort",
]

GENERIC_QUESTION_BANK = [
    "What does this image most likely show?",
    "What problem could this image be related to?",
    "What kind of treatment is commonly suggested for this issue?",
    "When should someone get this checked by a doctor?",
    "Is this usually considered serious?",
    "What symptoms are most relevant in this case?",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Augment HealMQA with two natural QA pairs per image.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to project config.")
    parser.add_argument("--input", type=str, default=None, help="Input HealMQA JSONL path.")
    parser.add_argument("--output", type=str, default=None, help="Augmented output JSONL path.")
    return parser.parse_args()


def sentence_split(text: str) -> list[str]:
    """Split free text into clean sentence units."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def normalize_text(text: str) -> str:
    """Normalize whitespace in free text."""
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, keywords: list[str]) -> bool:
    """Check whether any keyword appears in text."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def find_best_sentence(answer: str, keywords: list[str]) -> str | None:
    """Find the best sentence in an answer for a given keyword family."""
    sentences = sentence_split(answer)
    for sentence in sentences:
        if contains_any(sentence, keywords):
            return normalize_text(sentence)
    return None


def first_sentence(answer: str) -> str:
    """Return the first sentence or the normalized answer."""
    sentences = sentence_split(answer)
    return normalize_text(sentences[0] if sentences else answer)


def first_two_sentences(answer: str) -> str:
    """Return the first two sentences or the normalized answer."""
    sentences = sentence_split(answer)
    if not sentences:
        return normalize_text(answer)
    return normalize_text(" ".join(sentences[:2]))


def derive_condition_question(question: str, answer: str) -> tuple[str, str] | None:
    """Create a diagnosis-style independent question-answer pair."""
    condition_sentence = find_best_sentence(answer, DISEASE_TERMS)
    if condition_sentence is None:
        if contains_any(question, ["what is this", "what could this be", "what is it", "what exactly is this"]):
            condition_sentence = first_sentence(answer)
        elif contains_any(question + " " + answer, DISEASE_TERMS):
            condition_sentence = first_sentence(answer)
    if condition_sentence is None:
        return None
    return ("What condition might this image be showing?", condition_sentence)


def derive_treatment_question(question: str, answer: str) -> tuple[str, str] | None:
    """Create a treatment-style independent question-answer pair."""
    treatment_sentence = find_best_sentence(answer, TREATMENT_TERMS)
    if treatment_sentence is None and contains_any(question + " " + answer, TREATMENT_TERMS):
        treatment_sentence = first_two_sentences(answer)
    if treatment_sentence is None:
        return None
    return ("What kind of treatment may help in this case?", treatment_sentence)


def derive_severity_question(question: str, answer: str) -> tuple[str, str] | None:
    """Create a severity / when-to-worry question-answer pair."""
    severity_sentence = find_best_sentence(answer, SEVERITY_TERMS)
    if severity_sentence is None and contains_any(question + " " + answer, ["worry", "serious", "concern"]):
        severity_sentence = first_two_sentences(answer)
    if severity_sentence is None:
        return None
    return ("Is this usually something serious or worth worrying about?", severity_sentence)


def derive_cause_question(question: str, answer: str) -> tuple[str, str] | None:
    """Create a cause-style independent question-answer pair."""
    cause_sentence = find_best_sentence(answer, CAUSE_TERMS)
    if cause_sentence is None:
        return None
    return ("What could be causing this problem?", cause_sentence)


def derive_symptom_question(question: str, answer: str) -> tuple[str, str] | None:
    """Create a symptom-focused independent question-answer pair."""
    symptom_sentence = find_best_sentence(answer, SYMPTOM_TERMS)
    if symptom_sentence is None and contains_any(question + " " + answer, SYMPTOM_TERMS):
        symptom_sentence = first_sentence(answer)
    if symptom_sentence is None:
        return None
    return ("What symptoms or visible signs are important in this image?", symptom_sentence)


def derive_generic_pairs(answer: str) -> list[tuple[str, str]]:
    """Provide fallback QA pairs if no specific pattern is found."""
    summary = first_sentence(answer)
    extended = first_two_sentences(answer)
    return [
        (GENERIC_QUESTION_BANK[0], summary),
        (GENERIC_QUESTION_BANK[2], extended),
        (GENERIC_QUESTION_BANK[4], extended),
    ]


def select_two_pairs(question: str, answer: str) -> list[tuple[str, str]]:
    """Select two natural independent QA pairs for one sample."""
    candidate_builders = [
        derive_condition_question,
        derive_treatment_question,
        derive_severity_question,
        derive_cause_question,
        derive_symptom_question,
    ]

    candidates: list[tuple[str, str]] = []
    seen_questions = set()
    for builder in candidate_builders:
        result = builder(question, answer)
        if result is None:
            continue
        aug_question, aug_answer = result
        aug_question = normalize_text(aug_question)
        aug_answer = normalize_text(aug_answer)
        if not aug_question or not aug_answer or aug_question in seen_questions:
            continue
        seen_questions.add(aug_question)
        candidates.append((aug_question, aug_answer))

    if len(candidates) < 2:
        for fallback_question, fallback_answer in derive_generic_pairs(answer):
            fallback_question = normalize_text(fallback_question)
            fallback_answer = normalize_text(fallback_answer)
            if fallback_question in seen_questions or not fallback_answer:
                continue
            seen_questions.add(fallback_question)
            candidates.append((fallback_question, fallback_answer))
            if len(candidates) >= 2:
                break

    return candidates[:2]


def augment_record(record: dict) -> list[dict]:
    """Return original plus two augmented records for a single HealMQA sample."""
    original = dict(record)
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))

    augmented_records = [original]
    extra_pairs = select_two_pairs(question=question, answer=answer)

    for aug_index, (aug_question, aug_answer) in enumerate(extra_pairs, start=1):
        new_record = dict(record)
        new_record["question"] = aug_question
        new_record["answer"] = aug_answer
        if "id" in new_record:
            new_record["id"] = f"{new_record['id']}_aug{aug_index}"
        new_record["source_split"] = "custom_augmented"
        augmented_records.append(new_record)

    return augmented_records


def main() -> None:
    """Create a 3x augmented HealMQA file."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    healmqa_cfg = config["data"]["healmqa"]

    input_path = PROJECT_ROOT / (args.input or healmqa_cfg["output_path"])
    output_path = PROJECT_ROOT / (args.output or healmqa_cfg["augmented_output_path"])

    records = load_jsonl(input_path)
    augmented_records = []
    for record in records:
        augmented_records.extend(augment_record(record))

    ensure_dir(output_path.parent)
    write_jsonl(augmented_records, output_path)

    print(f"Loaded {len(records)} base HealMQA records from {input_path}")
    print(f"Exported {len(augmented_records)} total records to {output_path}")


if __name__ == "__main__":
    main()
