import os
os.environ.setdefault("HF_HOME", os.environ.get("TRANSFORMERS_CACHE", os.environ.get("HF_HOME", "")))
import json
import argparse
import time
from typing import List, Dict, Any, Iterable, Optional, Tuple, Type

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
# Defer heavy deps (vllm, easysteer) to call sites to keep light imports working


class BaseQADataset:
    """Base class for QA datasets expected by the steering pipeline."""

    name: str = "base"

    def __init__(self, path: str, split: str | None = None) -> None:
        self.path = path
        self.split = split

    def iter_rows(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a normalized row or None to skip it."""
        raise NotImplementedError

    def load_rows(self, max_samples: int | None = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in self.iter_rows():
            normalized = self.normalize_row(row)
            if normalized is None:
                continue
            rows.append(normalized)
            if max_samples and len(rows) >= max_samples:
                break
        return rows


class BarExamQA(BaseQADataset):
    """Bar exam multiple-choice QA dataset (JSONL)."""

    name = "barexamqa"

    def iter_rows(self) -> Iterable[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row

    def normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not row:
            return None

        normalized = {
            "prompt": row.get("prompt"),
            "question": row.get("question"),
            "choice_a": row.get("choice_a"),
            "choice_b": row.get("choice_b"),
            "choice_c": row.get("choice_c"),
            "choice_d": row.get("choice_d"),
            "gold_passage": row.get("gold_passage"),
            "answer": row.get("answer") or row.get("label"),
        }

        choices = row.get("choices")
        if isinstance(choices, dict):
            for label in ["A", "B", "C", "D"]:
                key = f"choice_{label.lower()}"
                if not normalized.get(key) and isinstance(choices.get(label), str):
                    normalized[key] = choices[label]

        if not isinstance(normalized.get("question"), str) and not isinstance(normalized.get("prompt"), str):
            return None

        return normalized

class HousingQA(BaseQADataset):
    name = "housingqa"

    def iter_rows(self) -> Iterable[Dict[str, Any]]:
        split = self.split or "test"
        dataset = load_dataset("reglab/housing_qa", "questions", split=split, trust_remote_code=True)
        for item in dataset:
            yield dict(item)

    def normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Implement normalization logic specific to HousingQA if needed
        return row


class TriviaQA(BaseQADataset):
    name = "triviaqa"

    def iter_rows(self) -> Iterable[Dict[str, Any]]:
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="test")
        for item in dataset:
            yield dict(item)

    def normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return row


DATASET_REGISTRY: Dict[str, Type[BaseQADataset]] = {
    BarExamQA.name: BarExamQA,
    "barexam": BarExamQA,
    HousingQA.name: HousingQA,
    "housingqa": HousingQA,
    TriviaQA.name: TriviaQA,
    "triviaqa": TriviaQA,
}


def _parse_dataset_spec(value: str) -> Tuple[str, str]:
    if ":" in value:
        candidate, path = value.split(":", 1)
        if candidate in DATASET_REGISTRY:
            return candidate, path
    if value in DATASET_REGISTRY:
        return value, ""
    return BarExamQA.name, value


def load_dataset_rows(
    path: str,
    max_samples: int | None = None,
    split: str | None = None,
) -> List[Dict[str, Any]]:
    dataset_name, dataset_path = _parse_dataset_spec(path)
    dataset_cls = DATASET_REGISTRY.get(dataset_name, BarExamQA)
    return dataset_cls(dataset_path, split=split).load_rows(max_samples=max_samples)


def load_barexam_rows(
    path: str,
    max_samples: int | None = None,
    split: str | None = None,
) -> List[Dict[str, Any]]:
    return load_dataset_rows(path, max_samples=max_samples, split=split)


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    parts = [p for p in parts if p]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def _extract_triviaqa_contexts(row: Dict[str, Any]) -> List[str]:
    contexts: List[str] = []
    entries = row.get("entity_pages")
    if isinstance(entries, dict):
        value = entries.get("wiki_context")
        if isinstance(value, str) and value.strip():
            contexts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    contexts.append(item)
    elif isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            value = entry.get("wiki_context")
            if isinstance(value, str) and value.strip():
                contexts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        contexts.append(item)
    return contexts


def load_triviaqa_paragraphs(
    max_samples: int | None = None,
    max_paragraphs: int | None = None,
    split: str = "train",
) -> List[str]:
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split=split)
    paragraphs: List[str] = []
    for idx, row in enumerate(dataset):
        contexts = _extract_triviaqa_contexts(dict(row))
        for ctx in contexts:
            for paragraph in _split_paragraphs(ctx):
                if len(paragraph) > 2000:
                    paragraph = paragraph[:2000]
                if paragraph:
                    paragraphs.append(paragraph)
            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                return paragraphs[:max_paragraphs]
        if max_samples and (idx + 1) >= max_samples:
            break
    return paragraphs


def build_gold_passage_texts(rows: List[Dict[str, Any]], max_length: int = 512) -> List[str]:
    """Extract gold passages and truncate them to avoid OOM.
    
    Args:
        rows: Dataset rows
        max_length: Maximum number of characters per passage (default 512)
    """
    texts: List[str] = []
    for r in rows:
        gp = r.get("gold_passage")
        if isinstance(gp, str) and gp.strip():
            # Truncate to avoid OOM with long passages
            truncated = gp[:max_length] if len(gp) > max_length else gp
            texts.append(truncated)
    return texts


def build_qa_prompt(row: Dict[str, Any]) -> str:
    """Build a QA prompt with question and choices."""
    parts: List[str] = []
    
    # Add prompt if available
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        parts.append(f"{row['prompt']}\n\n")
    
    # Add question
    if isinstance(row.get("question"), str) and row["question"].strip():
        parts.append(f"{row['question']}\n\n")
    
    # Add choices
    for label in ["A", "B", "C", "D"]:
        key = f"choice_{label.lower()}"
        if isinstance(row.get(key), str) and row[key].strip():
            parts.append(f"{label}) {row[key]}\n")
    
    return "".join(parts).strip()


def build_housingqa_prompt(row: Dict[str, Any]) -> str:
    """Build a yes/no prompt for HousingQA rows."""
    state = row.get("state")
    question = row.get("question")

    state_text = state.strip() if isinstance(state, str) and state.strip() else "the specified state"
    question_text = question.strip() if isinstance(question, str) and question.strip() else ""

    prompt = (
        f"Consider statutory law for {state_text} in the year 2021. {question_text}\n\n"
        'Answer "Yes" or "No".\n'
        "Answer:"
    )
    return prompt.strip()


def normalize_answer_label(row: Dict[str, Any]) -> str:
    """Return the multiple-choice label in uppercase or empty if missing."""
    for key in ("answer", "label"):
        if isinstance(row.get(key), str) and row[key].strip():
            label = row[key].strip().upper()
            if label in {"A", "B", "C", "D"}:
                return label
    return ""


def build_full_prompt(row: Dict[str, Any], include_gold_passage: bool = True) -> str:
    """Build a QA prompt for evaluation (BarExam QA)."""
    parts: List[str] = []
    if include_gold_passage:
        gp = row.get("gold_passage")
        if isinstance(gp, str) and gp.strip():
            parts.append("Legal Context\n")
            parts.append(f"{gp.strip()}\n\n")
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        parts.append(f"{row['prompt'].strip()}\n\n")
    if isinstance(row.get("question"), str) and row["question"].strip():
        parts.append(f"{row['question'].strip()}\n\n")
    for label in "ABCD":
        key = f"choice_{label.lower()}"
        if isinstance(row.get(key), str) and row[key].strip():
            parts.append(f"{label}) {row[key].strip()}\n")
    return "".join(parts).strip()
