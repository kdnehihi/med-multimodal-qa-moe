"""Normalize HealMQA symptom QA into a more consistent training set."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, load_config, load_jsonl, write_jsonl


CONDITION_TERMS = [
    "lipoma",
    "cyst",
    "baker",
    "eczema",
    "pink eye",
    "conjunctivitis",
    "rash",
    "infection",
    "fungal",
    "allergic",
    "stye",
    "acne",
    "wart",
    "boil",
    "cellulitis",
    "psoriasis",
    "lump",
    "bump",
    "growth",
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
    "rest",
    "ice",
    "sunscreen",
    "surgery",
    "therapy",
    "injection",
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
    "check-up",
    "not getting better",
    "painful",
    "changes in size",
    "abrupt changes",
    "safe",
]

CAUSE_TERMS = [
    "cause",
    "causing",
    "caused",
    "viral",
    "bacterial",
    "allergic",
    "trigger",
    "blocked",
    "buildup",
    "strain",
    "irritation",
]

QUESTION_TEMPLATES = {
    "condition": "What condition might this image be showing?",
    "treatment": "What kind of treatment or care may help in this case?",
    "severity": "Is this usually serious or something that should be checked by a doctor?",
    "cause": "What could be causing this problem?",
    "symptom": "What symptoms or visible signs are important in this image?",
    "general": "What problem could this image be related to, and what care may help?",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Normalize HealMQA symptom QA pairs.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to project config.")
    parser.add_argument("--input", type=str, default="data/processed/healmqa_1500.jsonl", help="Input JSONL path.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/healmqa_1500_normalized.jsonl",
        help="Normalized output JSONL path.",
    )
    parser.add_argument(
        "--drop-low-quality",
        action="store_true",
        help="Drop records marked as low quality instead of keeping them with flags.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", text).strip()


def cleanup_punctuation(text: str) -> str:
    """Reduce punctuation noise while preserving meaning."""
    text = text.replace("…", "...")
    text = re.sub(r"[!?]{2,}", "?", text)
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
    return normalize_whitespace(text)


def capitalize_sentence(text: str) -> str:
    """Capitalize the first alphabetical character in a sentence."""
    if not text:
        return text
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def sentence_split(text: str) -> list[str]:
    """Split free text into simple sentence units."""
    cleaned = cleanup_punctuation(text)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [capitalize_sentence(part.strip()) for part in parts if part.strip()]


def contains_any(text: str, keywords: list[str]) -> bool:
    """Check whether any keyword appears in the text."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def dedupe_sentences(sentences: list[str]) -> list[str]:
    """Remove near-duplicate sentences while keeping order."""
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in sentences:
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def remove_chatty_phrases(question: str) -> str:
    """Trim colloquial filler without changing the medical intent."""
    text = cleanup_punctuation(question)
    patterns = [
        r"\bplease help\b",
        r"\bhelp me\b",
        r"\blol\b",
        r"\bthanks\b\.?",
        r"\bany ideas\b",
        r"\bi was wondering if anyone had.*$",
        r"\bhas anybody else been through this surgery\b\.?",
        r"\bbut seriously\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = normalize_whitespace(text)
    text = text.strip(" -,.")
    return capitalize_sentence(text)


def infer_question_type(question: str, answer: str) -> str:
    """Map a free-form question to a small intent set."""
    question_lower = question.lower()
    merged = f"{question} {answer}".lower()
    if contains_any(question_lower, ["what is this", "what exactly is this", "what could this be", "what is it", "what condition", "diagnosis"]):
        return "condition"
    if contains_any(question_lower, ["worry", "serious", "danger", "concern", "doctor", "checked"]):
        return "severity"
    if contains_any(question_lower, ["cause", "causing", "why", "what causes"]):
        return "cause"
    if contains_any(question_lower, ["treat", "treatment", "help", "remove", "go away", "care", "surgery"]):
        return "treatment"
    if contains_any(question_lower, ["symptom", "sign", "look like", "visible"]):
        return "symptom"
    if contains_any(merged, CONDITION_TERMS):
        return "condition"
    return "general"


def standardize_question(question: str, answer: str) -> tuple[str, str]:
    """Produce a consistent question template and the inferred intent."""
    cleaned = remove_chatty_phrases(question)
    question_type = infer_question_type(cleaned, answer)
    standardized = QUESTION_TEMPLATES[question_type]
    return standardized, question_type


def find_best_sentence(answer: str, keywords: list[str]) -> str | None:
    """Find the first sentence matching a keyword family."""
    for sentence in sentence_split(answer):
        if contains_any(sentence, keywords):
            return sentence
    return None


def sentence_quality_flags(question: str, answer: str) -> list[str]:
    """Detect common quality issues in the original pair."""
    flags: list[str] = []
    q_key = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    a_key = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    if q_key and a_key and (q_key == a_key or q_key in a_key):
        flags.append("answer_duplicates_question")
    if len(sentence_split(answer)) == 1 and len(answer.split()) > 35:
        flags.append("long_single_sentence_answer")
    if len(question.split()) > 45:
        flags.append("very_long_question")
    if not contains_any(answer, CONDITION_TERMS + TREATMENT_TERMS + SEVERITY_TERMS + CAUSE_TERMS):
        flags.append("weak_medical_signal")
    return flags


def soften_condition_sentence(sentence: str) -> str:
    """Keep diagnosis language appropriately uncertain for low-resource QA."""
    lowered = sentence.lower()
    if lowered.startswith(("if ", "since ", "when ", "unless ", "apply ", "use ", "rest ", "avoid ")):
        return sentence
    if lowered.startswith(("this may be", "this could be", "this might be", "it may be", "it could be", "it might be")):
        return sentence
    if lowered.startswith(("a ", "an ", "the ")) and " is " in lowered:
        return sentence
    if "could be" in lowered or "might be" in lowered or "may be" in lowered:
        return sentence
    if lowered.startswith(("you have", "you are having", "you might be having", "your")):
        return sentence
    return f"This may be {sentence[0].lower() + sentence[1:]}" if sentence else sentence


def standardize_answer(question: str, answer: str) -> tuple[str, str, list[str]]:
    """Normalize answer into a shorter and more consistent style."""
    flags = sentence_quality_flags(question, answer)
    sentences = dedupe_sentences(sentence_split(answer))
    if not sentences:
        return "", "low", flags + ["empty_answer_after_cleanup"]

    question_type = infer_question_type(question, answer)

    condition_sentence = find_best_sentence(answer, CONDITION_TERMS)
    treatment_sentence = find_best_sentence(answer, TREATMENT_TERMS)
    severity_sentence = find_best_sentence(answer, SEVERITY_TERMS)
    cause_sentence = find_best_sentence(answer, CAUSE_TERMS)

    selected: list[str] = []
    if question_type == "condition" and condition_sentence:
        selected.append(soften_condition_sentence(condition_sentence))
    elif question_type == "treatment" and treatment_sentence:
        selected.append(treatment_sentence)
        if severity_sentence:
            selected.append(severity_sentence)
    elif question_type == "severity" and severity_sentence:
        selected.append(severity_sentence)
        if treatment_sentence:
            selected.append(treatment_sentence)
    elif question_type == "cause" and cause_sentence:
        selected.append(cause_sentence)
        if treatment_sentence:
            selected.append(treatment_sentence)
    elif question_type == "symptom":
        if condition_sentence:
            selected.append(soften_condition_sentence(condition_sentence))
        else:
            selected.append(sentences[0])
    else:
        if condition_sentence:
            selected.append(soften_condition_sentence(condition_sentence))
        if treatment_sentence:
            selected.append(treatment_sentence)
        if severity_sentence:
            selected.append(severity_sentence)

    if not selected:
        selected = sentences[:2]

    selected = dedupe_sentences(selected)[:2]
    normalized = normalize_whitespace(" ".join(selected))

    quality = "high"
    if "answer_duplicates_question" in flags or "empty_answer_after_cleanup" in flags:
        quality = "low"
    elif flags:
        quality = "medium"

    return normalized, quality, flags


def normalize_record(record: dict) -> dict | None:
    """Normalize one HealMQA record while preserving the original content."""
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    if not question or not answer:
        return None

    normalized_question, question_type = standardize_question(question, answer)
    normalized_answer, quality, flags = standardize_answer(question, answer)
    if not normalized_answer:
        return None

    normalized = dict(record)
    normalized["question"] = normalized_question
    normalized["answer"] = normalized_answer
    normalized["question_type"] = question_type
    normalized["normalization_quality"] = quality
    normalized["normalization_flags"] = flags
    normalized["original_question"] = question
    normalized["original_answer"] = answer
    normalized["source_split"] = f"{record.get('source_split', 'custom')}_normalized"
    return normalized


def main() -> None:
    """Normalize HealMQA and write a separate JSONL file."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    _ = config

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    records = load_jsonl(input_path)
    normalized_records: list[dict] = []
    dropped = 0
    quality_counts = {"high": 0, "medium": 0, "low": 0}

    for record in records:
        normalized = normalize_record(record)
        if normalized is None:
            dropped += 1
            continue
        if args.drop_low_quality and normalized["normalization_quality"] == "low":
            dropped += 1
            continue
        quality_counts[normalized["normalization_quality"]] += 1
        normalized_records.append(normalized)

    ensure_dir(output_path.parent)
    write_jsonl(normalized_records, output_path)

    print(f"Loaded {len(records)} records from {input_path}")
    print(f"Wrote {len(normalized_records)} normalized records to {output_path}")
    print(f"Dropped {dropped} records")
    print(
        "Quality counts: "
        f"high={quality_counts['high']} "
        f"medium={quality_counts['medium']} "
        f"low={quality_counts['low']}"
    )


if __name__ == "__main__":
    main()
