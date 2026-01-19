#!/usr/bin/env python3
import os

# Ensure local EasySteer package is importable when running from repo
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EASYSTEER_SRC = _REPO_ROOT / "EasySteer"
if _EASYSTEER_SRC.exists() and str(_EASYSTEER_SRC) not in sys.path:
    sys.path.insert(0, str(_EASYSTEER_SRC))

# Set GPU (optional)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import gc
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Literal, Optional

import bm25s
import Stemmer
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional
    wandb = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x, **kwargs):
        return x

from dataset_utils import build_qa_prompt, build_full_prompt, normalize_answer_label


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class QAExample:
    prompt: str
    negatives: List["NegativeContext"]
    answer: str
    positive_context: str | None = None


@dataclass
class EvalExample:
    prompt: str
    label: str
    negatives: List[str]


@dataclass
class NegativeContext:
    text: str
    weight: float = 1.0
    is_soft: bool = False


def passage_matches_gold(passage: str, gold_passage: str | None) -> bool:
    """Check if a retrieved passage overlaps with the known gold passage."""
    if not gold_passage:
        return False
    gold = gold_passage.strip()
    passage = passage.strip()
    if not gold or not passage:
        return False
    return gold in passage or passage in gold


def retrieve_hard_negatives(
    retriever: bm25s.BM25,
    query_text: str,
    gold_passage: str | None,
    k: int,
    stemmer: Stemmer.Stemmer,
) -> List[str]:
    """Retrieve top-k passages and drop any that look like the gold passage."""
    tokens = bm25s.tokenize(query_text, stemmer=stemmer)
    results, _ = retriever.retrieve(tokens, k=k)

    negatives: List[str] = []
    seen = set()
    for hit in results[0]:
        text = hit["text"] if isinstance(hit, dict) and "text" in hit else str(hit)
        if passage_matches_gold(text, gold_passage):
            continue
        key = text.strip()
        if not key or key in seen:
            continue
        negatives.append(text.strip())
        seen.add(key)
    return negatives


def format_prompt(
    base_prompt: str,
    context: str | None = None,
    *,
    thinking: bool = False,
    final_answer_cue: str | None = None,
) -> str:
    parts: List[str] = []
    if context:
        parts.append(f"Context:\n{context.strip()}\n\n")
    parts.append("Question:\n")
    parts.append(base_prompt)
    if thinking:
        parts.append("\n\nAssistant: <think>\n</think>\n")
        cue = final_answer_cue if final_answer_cue is not None else "Final answer: "
        parts.append(cue)
    else:
        parts.append("\n\nAnswer (A/B/C/D):")
        parts.append("\nRespond with a single letter only: A, B, C, or D.")
    return "".join(parts)


def prepare_eval_examples(
    rows: List[Dict[str, Any]],
    retriever: bm25s.BM25,
    stemmer: Stemmer.Stemmer,
    k: int,
    max_negatives: int,
    include_gold_passage: bool = True,
) -> List[EvalExample]:
    examples: List[EvalExample] = []
    for row in tqdm(rows, desc="Preparing eval examples", leave=False):
        label = normalize_answer_label(row)
        if not label:
            continue
        base_prompt = build_full_prompt(row, include_gold_passage=include_gold_passage)
        if not base_prompt:
            continue
        negatives = retrieve_hard_negatives(
            retriever,
            query_text=row.get("question", base_prompt),
            gold_passage=row.get("gold_passage"),
            k=k,
            stemmer=stemmer,
        )
        clipped = negatives[:max_negatives] if negatives else []
        examples.append(EvalExample(prompt=base_prompt, label=label, negatives=clipped))
    return examples


def apply_steer_pre_hook_last_token_only(
    block: torch.nn.Module,
    steer_vector: torch.Tensor,
    steer_scale: float,
    last_indices: torch.Tensor,
):
    def hook_fn(module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(h, torch.Tensor):
            return args, kwargs
        if h.dim() != 3:
            return args, kwargs

        batch_size = h.size(0)
        if last_indices.numel() != batch_size:
            return args, kwargs

        sv = (steer_scale * steer_vector).to(device=h.device, dtype=h.dtype)
        h2 = h.clone()
        rows = torch.arange(batch_size, device=h.device)
        idx = last_indices.to(device=h.device)
        h2[rows, idx, :] = h2[rows, idx, :] + sv

        if args:
            if len(args) == 1:
                return (h2,), kwargs
            return (h2, *args[1:]), kwargs

        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = h2
        return args, new_kwargs

    return block.register_forward_pre_hook(hook_fn, with_kwargs=True)




AggMode = Literal["max", "logsumexp"]


@dataclass
class CandidateScoreResult:
    label_scores: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor
    per_label_variant_scores: Dict[str, List[float]]


class ChoiceScorer:
    """Robust multi-token choice scoring for A/B/C/D by scoring candidate strings."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        agg: AggMode = "max",
        include_plain_letter: bool = True,
    ):
        self.tokenizer = tokenizer
        self.agg: AggMode = agg
        self.include_plain_letter = include_plain_letter
        self.candidate_map: Dict[str, List[str]] = self._build_candidate_strings()

    def _build_candidate_strings(self) -> Dict[str, List[str]]:
        templates = [
            " {L}",
            "{L}",
            "\n{L}",
            "\n\nAnswer: {L}",
            "\nAnswer: {L}",
            " Answer: {L}",
            "\n\nAnswer ({L}): {L}",
            "\n\nAnswer (A/B/C/D): {L}",
            "\nAnswer (A/B/C/D): {L}",
            " Answer (A/B/C/D): {L}",
            " {L}.",
            "{L}.",
            " {L})",
            "{L})",
            " **{L}**",
            "\n**{L}**",
            "\n\n**Answer: {L}**",
            "\n**Answer: {L}**",
            " **Answer: {L}**",
        ]

        mapping: Dict[str, List[str]] = {}
        for label in "ABCD":
            variants = [tpl.format(L=label) for tpl in templates]
            if self.include_plain_letter:
                variants += [label, f" {label}"]

            uniq: List[str] = []
            seen = set()
            for s in variants:
                if s not in seen:
                    uniq.append(s)
                    seen.add(s)
            mapping[label] = uniq
        return mapping

    def score_candidates_teacher_forcing(
        self,
        model: AutoModelForCausalLM,
        prompt: str,
        candidates: Sequence[str],
        device: Optional[torch.device] = None,
        max_length: Optional[int] = None,
        apply_steer_hook: Optional[Callable[[torch.Tensor, float], Any]] = None,
        steer_scale: Optional[float] = None,
        chunk_size: int = 1,
    ) -> torch.Tensor:
        device = device or next(model.parameters()).device
        scores: List[torch.Tensor] = []
        if chunk_size < 1:
            chunk_size = 1

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        for start in range(0, len(candidates), chunk_size):
            batch = candidates[start:start + chunk_size]
            full_texts = [prompt + c for c in batch]

            enc = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            max_prompt_idx = max(min(prompt_len, input_ids.size(1)) - 1, 0)

            handle = None
            if apply_steer_hook is not None and steer_scale is not None and steer_scale != 0.0:
                last_indices = torch.full(
                    (input_ids.size(0),),
                    max_prompt_idx,
                    device=device,
                    dtype=torch.long,
                )
                handle = apply_steer_hook(last_indices, steer_scale)

            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            if handle is not None:
                handle.remove()

            logits = out.logits
            log_denom = torch.logsumexp(logits, dim=-1)

            for b in range(input_ids.size(0)):
                seq_len = int(attn[b].sum().item())
                cand_token_ids = input_ids[b, prompt_len:seq_len]
                if cand_token_ids.numel() == 0:
                    scores.append(torch.tensor(0.0, device=device))
                    continue
                pred_pos = torch.arange(prompt_len - 1, seq_len - 1, device=device)
                if pred_pos.numel() != cand_token_ids.numel():
                    pred_pos = pred_pos[: cand_token_ids.numel()]
                token_logits = logits[b, pred_pos, cand_token_ids]
                token_logps = token_logits - log_denom[b, pred_pos]
                scores.append(token_logps.sum())

        return torch.stack(scores, dim=0)

    def _aggregate_label_scores(self, variant_scores: torch.Tensor) -> torch.Tensor:
        if self.agg == "max":
            return torch.max(variant_scores)
        return torch.logsumexp(variant_scores, dim=0)

    def score_prompt(
        self,
        model: AutoModelForCausalLM,
        prompt: str,
        device: Optional[torch.device] = None,
        max_length: Optional[int] = None,
        return_debug: bool = False,
        apply_steer_hook: Optional[Callable[[torch.Tensor, float], Any]] = None,
        steer_scale: Optional[float] = None,
        chunk_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor] | CandidateScoreResult:
        device = device or next(model.parameters()).device

        per_label_variant_scores: Dict[str, List[float]] = {}
        label_scores: List[torch.Tensor] = []

        for label in "ABCD":
            variants = self.candidate_map[label]
            vs = self.score_candidates_teacher_forcing(
                model=model,
                prompt=prompt,
                candidates=variants,
                device=device,
                max_length=max_length,
                apply_steer_hook=apply_steer_hook,
                steer_scale=steer_scale,
                chunk_size=chunk_size,
            )
            label_score = self._aggregate_label_scores(vs)
            label_scores.append(label_score)

            if return_debug:
                per_label_variant_scores[label] = [float(x) for x in vs.detach().cpu()]

        label_scores_t = torch.stack(label_scores, dim=0)
        probs = torch.softmax(label_scores_t, dim=-1)
        probs = torch.clamp(probs, min=1e-12)
        log_probs = probs.log()

        if return_debug:
            return CandidateScoreResult(
                label_scores=label_scores_t.detach(),
                probs=probs.detach(),
                log_probs=log_probs.detach(),
                per_label_variant_scores=per_label_variant_scores,
            )
        return probs, log_probs


class SingleTokenChoiceScorer:
    """Project next-token logprobs onto A/B/C/D single-token choices.

    Picks the correct single-token ids depending on whether the next token is expected
    to include a leading space/newline or not.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.ids_space = self._build_ids(prefix=" ")     # " A"
        self.ids_plain = self._build_ids(prefix="")      # "A"
        self.ids_newline = self._build_ids(prefix="\n")  # "\nA" (often 2 tokens, but try)

    def _one_token_id(self, s: str) -> Optional[int]:
        toks = self.tokenizer.encode(s, add_special_tokens=False)
        if len(toks) == 1:
            return toks[0]
        return None

    def _build_ids(self, prefix: str) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        for label in "ABCD":
            tid = self._one_token_id(prefix + label)
            if tid is None:
                # if not single-token, just skip (we'll fall back to other prefix types)
                continue
            ids[label] = tid
        return ids

    def _pick_id_map(self, prompt: str) -> Dict[str, int]:
        # If prompt ends with a space, next token for Llama-style outputs is usually " A"
        if prompt.endswith(" "):
            if len(self.ids_space) == 4:
                return self.ids_space

        # If prompt ends with newline, next token might be "\nA" or "A"
        if prompt.endswith("\n"):
            if len(self.ids_newline) == 4:
                return self.ids_newline
            if len(self.ids_plain) == 4:
                return self.ids_plain

        # Default: prefer space form if available, else plain
        if len(self.ids_space) == 4:
            return self.ids_space
        if len(self.ids_plain) == 4:
            return self.ids_plain

        raise ValueError(
            "Could not find a complete single-token mapping for A/B/C/D. "
            f"space={len(self.ids_space)}, plain={len(self.ids_plain)}, newline={len(self.ids_newline)}"
        )

    def project(self, log_probs: torch.Tensor, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._pick_id_map(prompt)
        idxs = torch.tensor([ids[l] for l in "ABCD"], device=log_probs.device, dtype=torch.long)
        selected = log_probs.index_select(dim=-1, index=idxs)
        probs = selected.softmax(dim=-1)
        probs = torch.clamp(probs, min=1e-12)
        return probs, probs.log()


import torch
from typing import Any, Dict, List, Tuple, Optional


class MultiChoiceScorer:
    """
    Fast multi-token choice scoring for A/B/C/D using KV-cache.

    - Prompt is cached once (past_key_values).
    - First continuation token is scored with optional steering (next-token logprobs).
    - Remaining continuation tokens are scored unsteered from cached prompt KV,
      using a *batched* forward pass (not per-token loops).
    - For each label, we take max over variants (robust to formatting).

    Designed for outputs like:
      **Answer: B**
      Answer: B
      B
      B.
      (etc.)
    """

    def __init__(
        self,
        tokenizer,
        variants: Optional[List[str]] = None,
        include_plain_letter: bool = True,
    ):
        self.tokenizer = tokenizer
        self.include_plain_letter = include_plain_letter

        # Variants tuned for your Qwen3 behavior (markdown + "Answer:" wrappers).
        # Keep this as small as possible while still covering observed formats.
        if variants is None:
            variants = [
                # Plain / spaced letter
                " {L}",
                "{L}",
                "\n{L}",
                "\n\n{L}",
                " {L}.",
                "{L}.",
                " {L})",
                "{L})",

                # "Answer:" wrappers (very common)
                "\nAnswer: {L}",
                "\n\nAnswer: {L}",
                " Answer: {L}",
                "\nAnswer: {L}.",
                "\n\nAnswer: {L}.",

                # Bold markdown wrappers (your sample)
                "\n**Answer: {L}**",
                "\n\n**Answer: {L}**",
                " **Answer: {L}**",

                # Sometimes models output bold letter only
                " **{L}**",
                "\n**{L}**",

                # Sometimes "Final answer: X"
                "\nFinal answer: {L}",
                "\n\nFinal answer: {L}",
                " Final answer: {L}",
            ]

        self.variants = variants
        self.candidate_map: Dict[str, List[List[int]]] = self._build_candidate_token_ids()

    def _build_candidate_token_ids(self) -> Dict[str, List[List[int]]]:
        mapping: Dict[str, List[List[int]]] = {}
        for label in "ABCD":
            seqs: List[List[int]] = []
            for tpl in self.variants:
                s = tpl.format(L=label)
                toks = self.tokenizer.encode(s, add_special_tokens=False)
                if len(toks) >= 1:
                    seqs.append(toks)

            if self.include_plain_letter:
                # Explicitly add raw letter and space-letter (often already present, but dedup handles it)
                for s in [label, f" {label}"]:
                    toks = self.tokenizer.encode(s, add_special_tokens=False)
                    if len(toks) >= 1:
                        seqs.append(toks)

            # Dedup by token sequence
            uniq: List[List[int]] = []
            seen = set()
            for t in seqs:
                key = tuple(t)
                if key not in seen:
                    uniq.append(t)
                    seen.add(key)
            mapping[label] = uniq
        return mapping

    @staticmethod
    def _repeat_past_key_values(past_key_values: Any, batch_size: int) -> Any:
        """Repeat cached past_key_values along batch dim to score many continuations at once."""
        if batch_size == 1:
            return past_key_values

        repeated = []
        for layer in past_key_values:
            new_layer = []
            for t in layer:
                if not isinstance(t, torch.Tensor):
                    new_layer.append(t)
                else:
                    new_layer.append(t.repeat(batch_size, *([1] * (t.dim() - 1))))
            repeated.append(tuple(new_layer))
        return tuple(repeated)

    @torch.no_grad()
    def _prompt_cache(
        self, trainer: "InvarianceSteeringTrainer", prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, int]:
        formatted = trainer._format_for_model(prompt, add_generation_prompt=False)
        toks = trainer.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=trainer.max_length,
        )
        input_ids = toks["input_ids"].to(trainer.device)
        attn = toks["attention_mask"].to(trainer.device)

        out = trainer.model(input_ids=input_ids, attention_mask=attn, use_cache=True)

        seq_len = int(attn.sum().item())

        # IMPORTANT: this must match your trainer helper signature.
        # (You used this form in your later paste.)
        first_idx = trainer._first_token_after_context_index(formatted, trainer.max_length)
        first_idx = max(min(first_idx, max(seq_len - 1, 0)), 0)

        return input_ids[0], attn[0], out.past_key_values, first_idx

    @torch.no_grad()
    def _tail_logprob_sums_from_cache_batched(
        self,
        trainer: "InvarianceSteeringTrainer",
        past_key_values: Any,
        tails: List[List[int]],
    ) -> torch.Tensor:
        """
        Compute sum log p(tail tokens) for each tail sequence in one batched forward pass.

        Returns: [B] tensor of tail logprob sums.
        """
        device = trainer.device
        B = len(tails)
        if B == 0:
            return torch.empty(0, device=device)

        maxT = max(len(x) for x in tails)
        if maxT == 0:
            return torch.zeros(B, device=device)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        x = torch.full((B, maxT), pad_id, device=device, dtype=torch.long)
        mask = torch.zeros((B, maxT), device=device, dtype=torch.float32)
        for i, seq in enumerate(tails):
            if not seq:
                continue
            L = len(seq)
            x[i, :L] = torch.tensor(seq, device=device, dtype=torch.long)
            mask[i, :L] = 1.0

        pkv = self._repeat_past_key_values(past_key_values, B)

        out = trainer.model(input_ids=x, past_key_values=pkv, use_cache=False)
        logits = out.logits.to(torch.float32)          # [B, T, V]
        logp = torch.log_softmax(logits, dim=-1)       # [B, T, V]
        gathered = logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B, T]
        return (gathered * mask).sum(dim=-1)           # [B]

    @torch.no_grad()
    def score_prompt(
        self,
        trainer: "InvarianceSteeringTrainer",
        prompt: str,
        steer_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          probs: [4] probabilities over A/B/C/D
          logp:  [4] log probabilities
        """
        prompt_ids, prompt_attn, past, first_idx = self._prompt_cache(trainer, prompt)

        # Must return log-probs over vocab for the next token position.
        # If your trainer returns logits instead, replace with log_softmax(logits).
        log_probs_first = trainer._next_token_logprobs_with_optional_steer(
            prompt_ids=prompt_ids,
            attention_mask=prompt_attn,
            steer_scale=steer_scale,
            steer_index=first_idx,
        )  # [V]

        label_scores: List[torch.Tensor] = []
        for label in "ABCD":
            seqs = self.candidate_map[label]
            if not seqs:
                label_scores.append(torch.tensor(-1e9, device=trainer.device))
                continue

            first_ids = [int(s[0]) for s in seqs]
            tails = [[int(x) for x in s[1:]] for s in seqs]

            first_ids_t = torch.tensor(first_ids, device=trainer.device, dtype=torch.long)
            first_lp = log_probs_first.index_select(dim=-1, index=first_ids_t)  # [N]

            tail_sums = self._tail_logprob_sums_from_cache(trainer, past, tails)
            totals = first_lp + tail_sums

            label_scores.append(torch.max(totals))

        scores = torch.stack(label_scores, dim=0)  # [4]
        probs = torch.softmax(scores, dim=-1)
        probs = torch.clamp(probs, min=1e-12)
        return probs, probs.log()

    def _is_hf_cache(self, past_key_values: Any) -> bool:
        # Qwen3 uses HF Cache objects (DynamicCache) that define get_seq_length()
        return hasattr(past_key_values, "get_seq_length") and callable(past_key_values.get_seq_length)

    @torch.no_grad()
    def _tail_logprob_sum_from_cache_single(
        self,
        trainer: "InvarianceSteeringTrainer",
        past_key_values: Any,
        tail: List[int],
    ) -> torch.Tensor:
        """
        Sum log p(tail[t] | prompt, tail[:t]) for one tail sequence,
        in ONE forward pass using the provided past_key_values.
        Works for both tuple PKV and HF Cache objects.
        """
        device = trainer.device
        if not tail:
            return torch.tensor(0.0, device=device)

        x = torch.tensor([tail], device=device, dtype=torch.long)  # [1, T]
        out = trainer.model(input_ids=x, past_key_values=past_key_values, use_cache=False)
        logits = out.logits[0].to(torch.float32)  # [T, V]
        logp = torch.log_softmax(logits, dim=-1)
        idx = torch.tensor(tail, device=device, dtype=torch.long).unsqueeze(-1)  # [T, 1]
        picked = logp.gather(-1, idx).squeeze(-1)  # [T]
        return picked.sum()

    @torch.no_grad()
    def _tail_logprob_sums_from_cache(
        self,
        trainer: "InvarianceSteeringTrainer",
        past_key_values: Any,
        tails: List[List[int]],
    ) -> torch.Tensor:
        """
        Compute tail logprob sums for many tails.

        - If HF Cache object (Qwen3): do per-tail scoring (cannot safely repeat cache).
        - If legacy tuple PKV: do batched scoring with repeated PKV (fastest).
        """
        device = trainer.device
        B = len(tails)
        if B == 0:
            return torch.empty(0, device=device)

        # ---- Qwen3 / Cache path ----
        if self._is_hf_cache(past_key_values):
            sums = [self._tail_logprob_sum_from_cache_single(trainer, past_key_values, t) for t in tails]
            return torch.stack(sums, dim=0)

        # ---- Legacy tuple PKV path (batched) ----
        maxT = max(len(x) for x in tails)
        if maxT == 0:
            return torch.zeros(B, device=device)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        x = torch.full((B, maxT), pad_id, device=device, dtype=torch.long)
        mask = torch.zeros((B, maxT), device=device, dtype=torch.float32)
        for i, seq in enumerate(tails):
            if not seq:
                continue
            L = len(seq)
            x[i, :L] = torch.tensor(seq, device=device, dtype=torch.long)
            mask[i, :L] = 1.0

        pkv = self._repeat_past_key_values(past_key_values, B)
        out = trainer.model(input_ids=x, past_key_values=pkv, use_cache=False)
        logits = out.logits.to(torch.float32)          # [B, T, V]
        logp = torch.log_softmax(logits, dim=-1)       # [B, T, V]
        gathered = logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B, T]
        return (gathered * mask).sum(dim=-1)           # [B]


# -----------------------------
# Data build
# -----------------------------
def build_training_examples(
    rows: List[Dict[str, Any]],
    retriever: bm25s.BM25,
    stemmer: Stemmer.Stemmer,
    k: int,
    max_negatives: int,
    debug: bool = False,
    debug_limit: int = 3,
) -> List[QAExample]:
    examples: List[QAExample] = []
    for idx, row in enumerate(tqdm(rows, desc="Preparing training examples", leave=False)):
        answer = normalize_answer_label(row)
        if not answer:
            if debug and len(examples) < debug_limit:
                print(f"[debug][row {idx}] skipped: missing answer label")
            continue

        base_prompt = build_qa_prompt(row)
        negatives = retrieve_hard_negatives(
            retriever,
            query_text=row.get("question", base_prompt),
            gold_passage=row.get("gold_passage"),
            k=k,
            stemmer=stemmer,
        )

        if not negatives:
            if debug and len(examples) < debug_limit:
                print(f"[debug][row {idx}] skipped: no contexts retrieved (k={k})")
            continue

        clipped = negatives[:max_negatives]
        gold_ctx = row.get("gold_passage")
        if isinstance(gold_ctx, str) and gold_ctx.strip():
            gold_ctx = gold_ctx.strip()
        else:
            gold_ctx = None
        neg_objs = [NegativeContext(text=ctx, weight=1.0, is_soft=False) for ctx in clipped]
        examples.append(
            QAExample(
                prompt=base_prompt,
                negatives=neg_objs,
                answer=answer,
                positive_context=gold_ctx,
            )
        )

        if debug and len(examples) <= debug_limit:
            question = row.get("question", base_prompt).replace("\n", " ")
            print(
                f"[debug][row {idx}] label={answer} contexts={len(negatives)} "
                f"kept={len(clipped)} prompt_len={len(base_prompt)}"
            )
            print(f"         question snippet: {question[:140]}")
            for neg in neg_objs[:2]:
                print(f"         context: {neg.text.replace(chr(10), ' ')[:160]}")
    return examples


def _score_choice_probs(
    trainer: "InvarianceSteeringTrainer",
    prompts: Sequence[str],
    chunk_size: int = 8,   # try 4/8/16 depending on VRAM
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    probs_all: List[torch.Tensor] = []
    logps_all: List[torch.Tensor] = []

    if trainer.choice_mode == "single":
        assert trainer.choice_scorer is not None
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i + chunk_size]
            formatted_chunk = [trainer._format_for_model(p, add_generation_prompt=False) for p in chunk]
            with torch.no_grad():
                lp_batch = trainer._choice_logprobs(chunk, steer_scale=0.0)  # [b, vocab]
            for j in range(lp_batch.size(0)):
                p_i, lp_i = trainer.choice_scorer.project(lp_batch[j], prompt=formatted_chunk[j])
                probs_all.append(p_i)
                logps_all.append(lp_i)
        return probs_all, logps_all

    # multi mode (slower): still chunk to avoid long batches
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i:i + chunk_size]
        for prompt in chunk:
            p_i, lp_i = trainer.choice_probs(prompt, steer_scale=0.0)
            probs_all.append(p_i)
            logps_all.append(lp_i)
    return probs_all, logps_all


def build_flip_mined_examples(
    rows: List[Dict[str, Any]],
    retriever: bm25s.BM25,
    stemmer: Stemmer.Stemmer,
    trainer: "InvarianceSteeringTrainer",
    k_pool: int,
    max_negatives: int,
    min_flips: int,
    fallback_top_r: int,
    soft_weight: float,
    fallback_metric: str,
    debug: bool = False,
    debug_limit: int = 3,
) -> List[QAExample]:
    examples: List[QAExample] = []
    for idx, row in enumerate(tqdm(rows, desc="Flip-mining contexts", leave=False)):
        answer = normalize_answer_label(row)
        if not answer:
            if debug and len(examples) < debug_limit:
                print(f"[debug][row {idx}] skipped: missing answer label")
            continue

        base_prompt = build_qa_prompt(row)
        if not base_prompt:
            if debug and len(examples) < debug_limit:
                print(f"[debug][row {idx}] skipped: empty prompt")
            continue

        clean_prompt = format_prompt(
            base_prompt,
            context=None,
            thinking=trainer.thinking,
            final_answer_cue=trainer.final_answer_cue,
        )
        p0_list, log_p0_list = _score_choice_probs(trainer, [clean_prompt])
        p0 = p0_list[0]
        log_p0 = log_p0_list[0]
        argmax_clean = int(torch.argmax(p0).item())
        gold_idx = "ABCD".index(answer)

        contexts = retrieve_hard_negatives(
            retriever,
            query_text=row.get("question", base_prompt),
            gold_passage=row.get("gold_passage"),
            k=k_pool,
            stemmer=stemmer,
        )
        if not contexts:
            if debug and len(examples) < debug_limit:
                print(f"[debug][row {idx}] skipped: no contexts retrieved (k_pool={k_pool})")
            continue

        ctx_prompts = [
            format_prompt(
                base_prompt,
                context=ctx,
                thinking=trainer.thinking,
                final_answer_cue=trainer.final_answer_cue,
            )
            for ctx in contexts
        ]
        ctx_probs, ctx_log_probs = _score_choice_probs(trainer, ctx_prompts, chunk_size=8)

        hard_negs: List[NegativeContext] = []
        soft_candidates: List[tuple[float, NegativeContext]] = []

        for ctx, pb, log_pb in zip(contexts, ctx_probs, ctx_log_probs):
            ctx_choice = int(torch.argmax(pb).item())
            if ctx_choice != argmax_clean:
                hard_negs.append(NegativeContext(text=ctx, weight=1.0, is_soft=False))
                continue

            if fallback_metric == "gold_drop":
                score = float((log_p0[gold_idx] - log_pb[gold_idx]).item())
            else:
                score = float(torch.sum(p0 * (log_p0 - log_pb)).item())
            soft_candidates.append((score, NegativeContext(text=ctx, weight=soft_weight, is_soft=True)))

        hard_negs = hard_negs[:max_negatives]
        soft_negs: List[NegativeContext] = []
        if len(hard_negs) < min_flips and soft_candidates and max_negatives > len(hard_negs):
            soft_candidates.sort(key=lambda x: x[0], reverse=True)
            remaining = max_negatives - len(hard_negs)
            take = min(fallback_top_r, remaining, len(soft_candidates))
            soft_negs = [c[1] for c in soft_candidates[:take]]

        negatives = hard_negs + soft_negs
        if not negatives:
            if debug and len(examples) < debug_limit:
                print(f"[debug][row {idx}] skipped: no flip-mined contexts")
            continue

        gold_ctx = row.get("gold_passage")
        if isinstance(gold_ctx, str) and gold_ctx.strip():
            gold_ctx = gold_ctx.strip()
        else:
            gold_ctx = None

        examples.append(
            QAExample(
                prompt=base_prompt,
                negatives=negatives,
                answer=answer,
                positive_context=gold_ctx,
            )
        )

        if debug and len(examples) <= debug_limit:
            question = row.get("question", base_prompt).replace("\n", " ")
            hard_count = sum(1 for n in negatives if not n.is_soft)
            soft_count = sum(1 for n in negatives if n.is_soft)
            print(
                f"[debug][row {idx}] label={answer} pool={len(contexts)} "
                f"hard={hard_count} soft={soft_count} prompt_len={len(base_prompt)}"
            )
            print(f"         question snippet: {question[:140]}")
            for neg in negatives[:2]:
                tag = "soft" if neg.is_soft else "hard"
                print(f"         {tag} context: {neg.text.replace(chr(10), ' ')[:160]}")

    return examples


def save_flip_mined_examples(path: str, examples: Sequence[QAExample]) -> None:
    payload = []
    for ex in examples:
        negs = [
            {"text": n.text, "weight": float(n.weight), "is_soft": bool(n.is_soft)}
            for n in ex.negatives
        ]
        payload.append(
            {
                "prompt": ex.prompt,
                "answer": ex.answer,
                "positive_context": ex.positive_context,
                "negatives": negs,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_flip_mined_examples(path: str) -> List[QAExample]:
    examples: List[QAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            prompt = row.get("prompt")
            answer = row.get("answer")
            if not isinstance(prompt, str) or not isinstance(answer, str):
                continue
            negs = []
            for n in row.get("negatives", []):
                if not isinstance(n, dict):
                    continue
                text = n.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                weight = float(n.get("weight", 1.0))
                is_soft = bool(n.get("is_soft", False))
                negs.append(NegativeContext(text=text, weight=weight, is_soft=is_soft))
            if not negs:
                continue
            pos_ctx = row.get("positive_context")
            if not isinstance(pos_ctx, str) or not pos_ctx.strip():
                pos_ctx = None
            examples.append(
                QAExample(
                    prompt=prompt,
                    negatives=negs,
                    answer=answer,
                    positive_context=pos_ctx,
                )
            )
    return examples


@torch.inference_mode()
def score_prompts_with_steer_batch(
    trainer: "InvarianceSteeringTrainer",
    choice_scorer: SingleTokenChoiceScorer,
    prompts: Sequence[str],
    steer_vector: torch.Tensor | None,
    steer_scale: float,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(prompts) == 0:
        raise ValueError("score_prompts_with_steer_batch: prompts is empty")

    formatted = [trainer._format_for_model(p, add_generation_prompt=False) for p in prompts]
    toks = trainer.tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(trainer.device)

    attn = toks["attention_mask"]
    last_indices = (attn.sum(dim=1) - 1).long()

    handle = None
    if steer_vector is not None and steer_scale != 0.0:
        handle = apply_steer_pre_hook_last_token_only(
            block=trainer.blocks[trainer.steer_layer],
            steer_vector=steer_vector,
            steer_scale=steer_scale,
            last_indices=last_indices,
        )

    out = trainer.model(**toks, use_cache=False)
    if handle is not None:
        handle.remove()

    logits = out.logits.to(torch.float32)
    row = torch.arange(logits.size(0), device=logits.device)
    last_logits = logits[row, last_indices, :]
    logp_vocab = torch.log_softmax(last_logits, dim=-1)

    probs_list: List[torch.Tensor] = []
    log_probs_list: List[torch.Tensor] = []
    for i in range(logp_vocab.size(0)):
        p_i, lp_i = choice_scorer.project(logp_vocab[i], prompt=formatted[i])
        probs_list.append(p_i)
        log_probs_list.append(lp_i)

    probs = torch.stack(probs_list, dim=0)
    log_probs = torch.stack(log_probs_list, dim=0)
    return probs, log_probs


def evaluate_on_barexam(
    trainer: "InvarianceSteeringTrainer",
    examples: Sequence[EvalExample],
    steer_vector: torch.Tensor | None,
    steer_scale: float,
    max_length: int,
    debug: bool = False,
    answers_out: str | None = None,
) -> Dict[str, float]:
    choice_scorer = SingleTokenChoiceScorer(trainer.tokenizer)

    kl_without: List[float] = []
    kl_with: List[float] = []
    kl_clean: List[float] = []

    clean_correct_base = 0
    clean_correct_steer = 0
    clean_changed = 0

    neg_correct_base = 0
    neg_correct_steer = 0
    restored = 0
    restored_to_gold = 0
    changed = 0
    total_pairs = 0

    do_steer = (steer_vector is not None) and (steer_scale != 0.0)

    answers_rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(tqdm(examples, desc="Evaluating", leave=False)):
        clean_prompt = format_prompt(
            ex.prompt,
            context=None,
            thinking=trainer.thinking,
            final_answer_cue=trainer.final_answer_cue,
        )

        if debug and idx < 3:
            formatted = trainer._format_for_model(clean_prompt, add_generation_prompt=False)
            toks = trainer.tokenizer(formatted, return_tensors="pt").to(trainer.device)
            with torch.inference_mode():
                gen = trainer.model.generate(
                    **toks,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=trainer.tokenizer.eos_token_id,
                )
            decoded = trainer.tokenizer.decode(gen[0], skip_special_tokens=True)
            response = decoded[len(formatted) :].strip() if decoded.startswith(formatted) else decoded.strip()
            print(f"[debug][example {idx}] model output:")
            print(response)

        gold_idx = "ABCD".index(ex.label)

        p0_batch, _ = score_prompts_with_steer_batch(
            trainer=trainer,
            choice_scorer=choice_scorer,
            prompts=[clean_prompt],
            steer_vector=None,
            steer_scale=0.0,
            max_length=max_length,
        )
        p0 = p0_batch[0]
        base_choice = int(torch.argmax(p0).item())
        clean_correct_base += int(base_choice == gold_idx)

        if do_steer:
            ps0_batch, _ = score_prompts_with_steer_batch(
                trainer=trainer,
                choice_scorer=choice_scorer,
                prompts=[clean_prompt],
                steer_vector=steer_vector,
                steer_scale=steer_scale,
                max_length=max_length,
            )
            ps0 = ps0_batch[0]
        else:
            ps0 = p0

        steer_clean_choice = int(torch.argmax(ps0).item())
        clean_correct_steer += int(steer_clean_choice == gold_idx)
        clean_changed += int(steer_clean_choice != base_choice)
        kl_clean.append(torch.sum(p0 * (p0.log() - ps0.log())).item())

        context_base_choices: List[str] = []
        context_steer_choices: List[str] = []

        if not ex.negatives:
            if answers_out:
                answers_rows.append(
                    {
                        "prompt": ex.prompt,
                        "label": ex.label,
                        "clean_base": "ABCD"[base_choice],
                        "clean_steer": "ABCD"[steer_clean_choice],
                        "contexts": [],
                        "context_base": [],
                        "context_steer": [],
                    }
                )
            continue

        ctx_prompts = [
            format_prompt(
                ex.prompt,
                context=ctx,
                thinking=trainer.thinking,
                final_answer_cue=trainer.final_answer_cue,
            )
            for ctx in ex.negatives
        ]
        total_pairs += len(ctx_prompts)

        pw_batch, _ = score_prompts_with_steer_batch(
            trainer=trainer,
            choice_scorer=choice_scorer,
            prompts=ctx_prompts,
            steer_vector=None,
            steer_scale=0.0,
            max_length=max_length,
        )

        if do_steer:
            ps_batch, _ = score_prompts_with_steer_batch(
                trainer=trainer,
                choice_scorer=choice_scorer,
                prompts=ctx_prompts,
                steer_vector=steer_vector,
                steer_scale=steer_scale,
                max_length=max_length,
            )
        else:
            ps_batch = pw_batch

        for j in range(len(ctx_prompts)):
            pw = pw_batch[j]
            ps = ps_batch[j]

            kl_without.append(torch.sum(p0 * (p0.log() - pw.log())).item())
            kl_with.append(torch.sum(p0 * (p0.log() - ps.log())).item())

            wrong_choice = int(torch.argmax(pw).item())
            steer_choice = int(torch.argmax(ps).item())
            context_base_choices.append("ABCD"[wrong_choice])
            context_steer_choices.append("ABCD"[steer_choice])

            changed += int(wrong_choice != base_choice)
            neg_correct_base += int(wrong_choice == gold_idx)
            neg_correct_steer += int(steer_choice == gold_idx)
            restored += int((wrong_choice != base_choice) and (steer_choice == base_choice))
            restored_to_gold += int((wrong_choice != gold_idx) and (steer_choice == gold_idx))

        if answers_out:
            answers_rows.append(
                {
                    "prompt": ex.prompt,
                    "label": ex.label,
                    "clean_base": "ABCD"[base_choice],
                    "clean_steer": "ABCD"[steer_clean_choice],
                    "contexts": list(ex.negatives),
                    "context_base": context_base_choices,
                    "context_steer": context_steer_choices,
                }
            )

    clean_count = max(len(examples), 1)
    pair_count = total_pairs if total_pairs else 1

    if answers_out:
        with open(answers_out, "w", encoding="utf-8") as f:
            for row in answers_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "clean_examples": float(len(examples)),
        "pairs": float(total_pairs),
        "clean_acc_base": clean_correct_base / clean_count,
        "clean_acc_steer": clean_correct_steer / clean_count,
        "clean_changed_pct": 100.0 * clean_changed / clean_count,
        "neg_acc_base": (neg_correct_base / pair_count) if total_pairs else 0.0,
        "neg_acc_steer": (neg_correct_steer / pair_count) if total_pairs else 0.0,
        "kl_clean": float(sum(kl_clean) / max(len(kl_clean), 1)),
        "kl_without": float(sum(kl_without) / max(len(kl_without), 1)),
        "kl_with": float(sum(kl_with) / max(len(kl_with), 1)),
        "changed_pct": 100.0 * changed / pair_count if total_pairs else 0.0,
        "restored_pct": 100.0 * restored / pair_count if total_pairs else 0.0,
        "restored_to_gold_pct": 100.0 * restored_to_gold / pair_count if total_pairs else 0.0,
    }


# -----------------------------
# Export
# -----------------------------
def save_control_vector(
    vector: torch.Tensor,
    model_name: str,
    layer: int,
    steer_scale: float,
    output_path: str,
    notes: str,
) -> None:
    """Export the learned vector. Prefers GGUF via EasySteer; falls back to .pt."""
    direction = vector.detach().cpu().float().numpy()
    try:
        from easysteer.steer.utils import StatisticalControlVector  # type: ignore

        control_vector = StatisticalControlVector(
            model_type=model_name,
            method="invariance",
            directions={layer: direction},
            metadata={"steer_scale": steer_scale, "notes": notes, "layer": layer},
        )
        control_vector.export_gguf(output_path)
        print(f"Saved GGUF steering vector to {output_path}")
    except Exception as e:
        base, _ext = os.path.splitext(output_path)
        alt_path = base + ".pt"
        torch.save(
            {
                "model_type": model_name,
                "method": "invariance",
                "layer": layer,
                "steer_scale": steer_scale,
                "notes": notes,
                "direction": vector.detach().cpu(),
            },
            alt_path,
        )
        print(
            f"EasySteer/gguf export unavailable ({e}). Saved PyTorch tensor to {alt_path}. "
            f"Install dependencies to export GGUF."
        )



# -----------------------------
# Main
# -----------------------------

def load_control_vector(path: str) -> tuple[torch.Tensor, int, float]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        payload = torch.load(path, map_location="cpu")
        vector = payload["direction"]
        layer = int(payload.get("layer", 0))
        steer_scale = float(payload.get("steer_scale", 0.0))
        return vector, layer, steer_scale

    try:
        from easysteer.steer.utils import StatisticalControlVector  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"GGUF vector loading requires easysteer; failed to import: {exc}") from exc

    control = StatisticalControlVector.import_gguf(path)
    if not control.directions:
        raise ValueError(f"No directions found in {path}")
    if control.metadata and "layer" in control.metadata:
        layer = int(control.metadata["layer"])
    else:
        layer = sorted(control.directions.keys())[0]
    direction = control.directions[layer]
    steer_scale = 0.0
    if control.metadata and "steer_scale" in control.metadata:
        steer_scale = float(control.metadata["steer_scale"])
    return torch.tensor(direction), layer, steer_scale


def cleanup_trainer(trainer: "InvarianceSteeringTrainer") -> None:
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
