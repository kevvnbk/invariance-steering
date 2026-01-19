#!/usr/bin/env python3
"""Helper functions for testing invariance steering vectors."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

# Re-export commonly used classes/functions from the library
from invariance_steering_lib import (
    SingleTokenChoiceScorer,
    format_prompt,
)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# Model utilities
# -----------------------------
def locate_transformer_blocks(model: AutoModelForCausalLM) -> List[torch.nn.Module]:
    """Locate transformer blocks for various model architectures."""
    if hasattr(model, "model"):
        inner = getattr(model, "model")
        if hasattr(inner, "layers"):
            layers = getattr(inner, "layers")
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return list(layers)
    if hasattr(model, "transformer"):
        tr = getattr(model, "transformer")
        if hasattr(tr, "h"):
            h = getattr(tr, "h")
            if isinstance(h, (list, torch.nn.ModuleList)):
                return list(h)
    return []


def _format_for_model(tokenizer: Any, prompt: str, add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                return prompt
        except Exception:
            return prompt
    return prompt

def _format_for_scoring(tokenizer: Any, prompt: str) -> str:
    # When scoring next-token logits, chat models need the assistant prefix.
    if hasattr(tokenizer, "apply_chat_template"):
        return _format_for_model(tokenizer, prompt, add_generation_prompt=True)
    return prompt

def _first_token_after_context_index(tokenizer: Any, prompt: str, max_length: int) -> int:
    marker = "Question:\n"
    idx = prompt.find(marker)
    prefix_raw = prompt[: idx + len(marker)] if idx >= 0 else ""
    formatted_prefix = _format_for_model(tokenizer, prefix_raw, add_generation_prompt=False)
    toks = tokenizer(
        formatted_prefix,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return int(toks["input_ids"].shape[1])


def apply_steer_pre_hook_span(
    block: torch.nn.Module,
    steer_vector: torch.Tensor,
    steer_scale: float,
    start_indices: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    def hook_fn(_module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(h, torch.Tensor):
            return args, kwargs
        if h.dim() != 3:
            return args, kwargs

        batch_size, seq_len, hidden_dim = h.shape
        if start_indices.numel() != batch_size:
            return args, kwargs

        sv = (steer_scale * steer_vector).to(device=h.device, dtype=h.dtype)
        idx = start_indices.to(device=h.device).clamp(0, seq_len - 1)
        t = torch.arange(seq_len, device=h.device).unsqueeze(0)
        mask = (t >= idx.unsqueeze(1)).to(h.dtype)
        if attention_mask is not None:
            am = attention_mask.to(device=h.device, dtype=h.dtype)
            mask = mask * am

        sv_expanded = sv.view(1, 1, hidden_dim).expand(batch_size, seq_len, hidden_dim)
        mask_expanded = mask.unsqueeze(-1).expand(batch_size, seq_len, hidden_dim)
        h2 = h + sv_expanded * mask_expanded

        if args:
            if len(args) == 1:
                return (h2,), kwargs
            return (h2, *args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = h2
        return args, new_kwargs

    return block.register_forward_pre_hook(hook_fn, with_kwargs=True)

@torch.no_grad()
def score_batch_with_optional_span_steer(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    choice_scorer: SingleTokenChoiceScorer,
    block: torch.nn.Module,
    prompts: Sequence[str],              # UNFORMATTED prompts (your format_prompt output)
    steer_vector: torch.Tensor | None,
    steer_scale: float,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    device = next(model.parameters()).device

    formatted = [_format_for_model(tokenizer, p, add_generation_prompt=False) for p in prompts]
    toks = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    attn = toks["attention_mask"]

    # boundary indices per prompt
    start_indices = torch.tensor(
        [_first_token_after_context_index(tokenizer, p, max_length) for p in prompts],
        device=device,
        dtype=torch.long,
    )
    seq_lens = attn.sum(dim=1).long()
    start_indices = torch.minimum(start_indices, torch.clamp(seq_lens - 1, min=0))

    handle = None
    if steer_vector is not None and steer_scale != 0.0:
        handle = apply_steer_pre_hook_span(
            block=block,
            steer_vector=steer_vector,
            steer_scale=steer_scale,
            start_indices=start_indices,
            attention_mask=attn,
        )

    out = model(**toks)
    if handle is not None:
        handle.remove()

    logits = out.logits.to(torch.float32)
    last_indices = (attn.sum(dim=1) - 1).long()
    rows = torch.arange(logits.size(0), device=device)
    last_logits = logits[rows, last_indices, :]
    logp_vocab = torch.log_softmax(last_logits, dim=-1)

    probs_list, logps_list = [], []
    for i in range(logp_vocab.size(0)):
        p_i, lp_i = choice_scorer.project(logp_vocab[i], prompt=formatted[i])
        probs_list.append(p_i)
        logps_list.append(lp_i)

    return torch.stack(probs_list, dim=0), torch.stack(logps_list, dim=0), formatted


# -----------------------------
# Evaluation data structures
# -----------------------------
@dataclass
class EvalExample:
    """Evaluation example with prompt, label, and contexts."""
    prompt: str
    label: str
    contexts: List[str]
    gold_passage: str | None = None


def prepare_examples(
    rows: List[Dict[str, Any]],
    retriever: Any,
    stemmer: Optional[Any],
    k: int,
    max_negatives: int,
) -> List[EvalExample]:
    """Build evaluation examples with BM25 or embedding retrieved contexts."""
    from dataset_utils import load_dataset_rows, normalize_answer_label, build_full_prompt
    import bm25s
    
    examples: List[EvalExample] = []
    for row in rows:
        label = normalize_answer_label(row)
        if not label:
            continue
        base_prompt = build_full_prompt(row, include_gold_passage=False)
        if not base_prompt:
            continue
        
        gold_passage = row.get("gold_passage")
        if not isinstance(gold_passage, str) or not gold_passage.strip():
            gold_passage = None
        
        query_text = row.get("question", base_prompt)
        if stemmer is None and hasattr(retriever, "retrieve_text"):
            hits = retriever.retrieve_text(query_text, k=k)
        else:
            tokens = bm25s.tokenize(query_text, stemmer=stemmer)
            results, _ = retriever.retrieve(tokens, k=k)
            hits = results[0]
        
        contexts: List[str] = []
        for hit in hits:
            text = hit["text"] if isinstance(hit, dict) and "text" in hit else str(hit)
            text = text.strip()
            if text and (not gold_passage or text != gold_passage):
                contexts.append(text)
        
        clipped = contexts[:max_negatives] if contexts else []
        examples.append(
            EvalExample(
                prompt=base_prompt,
                label=label,
                contexts=clipped,
                gold_passage=gold_passage,
            )
        )
    return examples


def build_context_examples(
    examples: List[EvalExample],
    config: str,
    mixed_min: int,
    mixed_max: int,
    injected_contexts: Optional[Sequence[str]] = None,
    injected_per_example: int = 0,
) -> List[EvalExample]:
    """Build evaluation examples with different context configurations.
    
    Args:
        examples: Base evaluation examples
        config: One of "clean", "gold-only", "rag-topk", "rag-topk+injected", "gold+rag"
        mixed_min: Minimum distractors for mixed config
        mixed_max: Maximum distractors for mixed config
        injected_contexts: Optional contexts to inject (used for rag-topk+injected)
        injected_per_example: Number of injected contexts per example
    """
    import random
    
    if config == "clean":
        return [EvalExample(prompt=ex.prompt, label=ex.label, contexts=[], gold_passage=ex.gold_passage) 
                for ex in examples]
    
    elif config == "gold-only":
        return [EvalExample(prompt=ex.prompt, label=ex.label, 
                           contexts=[ex.gold_passage] if ex.gold_passage else [],
                           gold_passage=ex.gold_passage)
                for ex in examples]
    
    elif config == "rag-topk":
        return examples

    elif config == "rag-topk+injected":
        if not injected_contexts or injected_per_example <= 0:
            return examples
        result = []
        for ex in examples:
            contexts = list(ex.contexts)
            for _ in range(injected_per_example):
                contexts.append(random.choice(injected_contexts))
            result.append(EvalExample(prompt=ex.prompt, label=ex.label, 
                                     contexts=contexts, gold_passage=ex.gold_passage))
        return result
    
    elif config in {"gold+bm25", "gold+rag"}:
        result = []
        for ex in examples:
            contexts = list(ex.contexts)
            if ex.gold_passage:
                contexts = [ex.gold_passage] + contexts
            result.append(EvalExample(prompt=ex.prompt, label=ex.label, 
                                     contexts=contexts, gold_passage=ex.gold_passage))
        return result
    
    else:
        raise ValueError(f"Unknown config: {config}")


def filter_flip_contexts(
    examples: List[EvalExample],
    model: AutoModelForCausalLM,
    tokenizer: Any,
    choice_scorer: SingleTokenChoiceScorer,
    blocks: List[torch.nn.Module],
    layer_id: int,
    max_length: int,
    debug: bool = False,
) -> List[EvalExample]:
    """Filter contexts to only include those that flip the model's prediction.
    
    Returns examples with only contexts that change the model's answer.
    """
    filtered: List[EvalExample] = []
    
    for ex in tqdm(examples, desc="Filtering flip contexts", leave=False):
        # Get clean prediction
        FINAL_ANSWER_CUE = "Final answer: "

        clean_prompt = format_prompt(ex.prompt, context=None, final_answer_cue=FINAL_ANSWER_CUE)
        formatted = _format_for_scoring(tokenizer, clean_prompt)
        
        toks = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
        toks = {k: v.to(next(model.parameters()).device) for k, v in toks.items()}
        
        with torch.no_grad():
            out = model(**toks)
            logits = out.logits[:, -1, :].to(torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)[0]
            probs, _ = choice_scorer.project(log_probs, prompt=formatted)
            clean_choice = int(torch.argmax(probs).item())
        
        # Check each context
        flip_contexts: List[str] = []
        for ctx in ex.contexts:
            ctx_prompt   = format_prompt(ex.prompt, context=ctx, final_answer_cue=FINAL_ANSWER_CUE)
            ctx_formatted = _format_for_scoring(tokenizer, ctx_prompt)
            
            ctx_toks = tokenizer(ctx_formatted, return_tensors="pt", truncation=True, max_length=max_length)
            ctx_toks = {k: v.to(next(model.parameters()).device) for k, v in ctx_toks.items()}
            
            with torch.no_grad():
                ctx_out = model(**ctx_toks)
                ctx_logits = ctx_out.logits[:, -1, :].to(torch.float32)
                ctx_log_probs = torch.log_softmax(ctx_logits, dim=-1)[0]
                ctx_probs, _ = choice_scorer.project(ctx_log_probs, prompt=ctx_formatted)
                ctx_choice = int(torch.argmax(ctx_probs).item())
            
            if ctx_choice != clean_choice:
                flip_contexts.append(ctx)
        
        if flip_contexts:
            filtered.append(EvalExample(
                prompt=ex.prompt,
                label=ex.label,
                contexts=flip_contexts,
                gold_passage=ex.gold_passage,
            ))
    
    return filtered


def build_mixed_from_flip(
    examples: List[EvalExample],
    flip_examples: List[EvalExample],
    mixed_min: int,
    mixed_max: int,
) -> List[EvalExample]:
    """Build mixed examples combining flip contexts with gold passages.
    
    Args:
        examples: Original examples with gold passages
        flip_examples: Filtered examples with only flip contexts
        mixed_min: Minimum number of flip contexts
        mixed_max: Maximum number of flip contexts
    """
    import random
    
    # Map prompts to flip contexts
    flip_map: Dict[str, List[str]] = {}
    for ex in flip_examples:
        flip_map[ex.prompt] = ex.contexts
    
    result: List[EvalExample] = []
    for ex in examples:
        flip_ctxs = flip_map.get(ex.prompt, [])
        if not flip_ctxs:
            continue
        
        # Sample flip contexts
        n_flips = min(len(flip_ctxs), random.randint(mixed_min, mixed_max))
        sampled_flips = random.sample(flip_ctxs, n_flips)
        
        # Combine with gold passage if available
        contexts = list(sampled_flips)
        if ex.gold_passage:
            contexts.append(ex.gold_passage)
            random.shuffle(contexts)
        
        result.append(EvalExample(
            prompt=ex.prompt,
            label=ex.label,
            contexts=contexts,
            gold_passage=ex.gold_passage,
        ))
    
    return result


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    choice_scorer: SingleTokenChoiceScorer,
    blocks: List[torch.nn.Module],
    steer_vector: torch.Tensor | None,
    steer_scale: float,
    layer_id: int,
    examples: List[EvalExample],
    max_length: int,
    debug: bool = False,
    answers_out: str | None = None,
) -> Dict[str, float]:
    """Evaluate steering vector on examples.

    Updated to match the trainer's "span injection" behavior:
      - compute inject_idx = first token after context boundary ("Question:\\n")
      - when steering: add sv to ALL tokens t >= inject_idx (masked by attention_mask)

    Returns metrics including accuracy, KL divergence, and restoration rates.
    """
    import json

    device = next(model.parameters()).device
    do_steer = (steer_vector is not None) and (steer_scale != 0.0)

    kl_clean: List[float] = []
    kl_without: List[float] = []
    kl_with: List[float] = []

    clean_acc_base = 0
    clean_acc_steer = 0
    clean_changed = 0

    ctx_acc_base = 0
    ctx_acc_steer = 0
    changed = 0
    restored = 0
    restored_to_gold = 0
    total_pairs = 0

    answers_rows: List[Dict[str, Any]] = []

    def _capture_block_input(block: torch.nn.Module, toks: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        captured: Dict[str, torch.Tensor] = {}

        def cap_hook(_module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if isinstance(h, torch.Tensor):
                captured["h"] = h.detach().clone()
            return args, kwargs

        handle = block.register_forward_pre_hook(cap_hook, with_kwargs=True)
        model(**toks)
        handle.remove()
        return captured.get("h")

    def _score_one_prompt(
        prompt_text: str,
        steer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, Dict[str, Any]]:
        """
        Returns:
          probs (A/B/C/D), log_probs(A/B/C/D), choice_idx, inject_idx_used, debug_info
        """
        formatted = _format_for_scoring(tokenizer, prompt_text)

        toks = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
        toks = {k: v.to(device) for k, v in toks.items()}

        attn = toks.get("attention_mask")
        seq_len = int(attn.sum(dim=1).item()) if attn is not None else toks["input_ids"].shape[1]
        inject_idx = _first_token_after_context_index(tokenizer, formatted, max_length)
        inject_idx = max(min(inject_idx, max(seq_len - 1, 0)), 0)

        handle = None
        if steer and do_steer:
            start_indices = torch.tensor([inject_idx], device=device, dtype=torch.long)
            handle = apply_steer_pre_hook_span(
                blocks[layer_id],
                steer_vector,
                steer_scale,
                start_indices,
                attention_mask=attn,
            )

        with torch.no_grad():
            out = model(**toks)
            logits = out.logits[:, -1, :].to(torch.float32)
            log_probs_vocab = torch.log_softmax(logits, dim=-1)[0]
            probs, log_probs = choice_scorer.project(log_probs_vocab, prompt=formatted)
            choice_idx = int(torch.argmax(probs).item())

        if handle is not None:
            handle.remove()

        dbg = {
            "formatted": formatted,
            "seq_len": seq_len,
            "inject_idx": inject_idx,
            "last_idx": seq_len - 1,
            "tail_token_ids": toks["input_ids"][0, max(0, seq_len - 4) : seq_len].tolist(),
        }
        return probs, log_probs, choice_idx, inject_idx, dbg

    for idx, ex in enumerate(tqdm(examples, desc="Evaluating", leave=False)):
        gold_idx = "ABCD".index(ex.label)

        # -----------------
        # Clean prompt
        # -----------------
        FINAL_ANSWER_CUE = "Final answer: "

        clean_prompt = format_prompt(ex.prompt, context=None, final_answer_cue=FINAL_ANSWER_CUE)

        p0, log_p0, clean_choice, clean_inject_idx, dbg0 = _score_one_prompt(clean_prompt, steer=False)
        clean_acc_base += int(clean_choice == gold_idx)

        if debug and idx == 0:
            tail_tokens = [tokenizer.decode([tid]) for tid in dbg0["tail_token_ids"]]
            formatted = dbg0["formatted"]
            tail = formatted[-100:] if len(formatted) > 100 else formatted
            has_options = all(f"{c})" in formatted for c in "ABCD")
            has_final_answer = "Final answer:" in formatted
            print(
                f"[debug][diag] clean seq_len={dbg0['seq_len']} last_idx={dbg0['last_idx']} "
                f"inject_idx={dbg0['inject_idx']}"
            )
            print(f"[debug][diag] clean tail tokens: {' | '.join(tail_tokens)}")
            print(f"[debug][diag] formatted tail (last 100 chars): {tail}")
            print(f"[debug][diag] formatted has options A/B/C/D: {has_options}")
            print(f"[debug][diag] formatted has 'Final answer:': {has_final_answer}")

        if do_steer:
            if debug and idx == 0:
                # Show that the hook actually changes the block input at the boundary token
                formatted = dbg0["formatted"]
                toks = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
                toks = {k: v.to(device) for k, v in toks.items()}
                base_h = _capture_block_input(blocks[layer_id], toks)

                start_indices = torch.tensor([clean_inject_idx], device=device, dtype=torch.long)
                steer_handle_dbg = apply_steer_pre_hook_span(
                    blocks[layer_id],
                    steer_vector,
                    steer_scale,
                    start_indices,
                    attention_mask=toks.get("attention_mask"),
                )
                steer_h = _capture_block_input(blocks[layer_id], toks)
                steer_handle_dbg.remove()

                if base_h is not None and steer_h is not None:
                    delta = (steer_h - base_h)[0, int(clean_inject_idx), :].norm().item()
                    print(f"[debug][diag] hook delta norm at inject token: {delta:.6f}")

            ps0, log_ps0, steer_clean_choice, _inj, _dbg = _score_one_prompt(clean_prompt, steer=True)
        else:
            ps0, log_ps0 = p0, log_p0
            steer_clean_choice = clean_choice

        # KEEP THESE METRICS (unchanged)
        clean_acc_steer += int(steer_clean_choice == gold_idx)
        clean_changed += int(steer_clean_choice != clean_choice)
        kl_clean.append(float(torch.sum(p0 * (log_p0 - log_ps0)).item()))

        if not ex.contexts:
            if answers_out:
                answers_rows.append(
                    {
                        "prompt": ex.prompt,
                        "label": ex.label,
                        "clean_base": "ABCD"[clean_choice],
                        "clean_steer": "ABCD"[steer_clean_choice],
                        "contexts": [],
                        "context_base": [],
                        "context_steer": [],
                    }
                )
            continue

        # -----------------
        # Context prompts
        # -----------------
        ctx_base_choices: List[str] = []
        ctx_steer_choices: List[str] = []

        for ctx in ex.contexts:
            ctx_prompt   = format_prompt(ex.prompt, context=ctx,  final_answer_cue=FINAL_ANSWER_CUE)

            pb, log_pb, ctx_choice, ctx_inject_idx, dbgc = _score_one_prompt(ctx_prompt, steer=False)

            if debug and idx == 0:
                tail_tokens = [tokenizer.decode([tid]) for tid in dbgc["tail_token_ids"]]
                print(
                    f"[debug][diag] ctx seq_len={dbgc['seq_len']} last_idx={dbgc['last_idx']} "
                    f"inject_idx={dbgc['inject_idx']}"
                )
                print(f"[debug][diag] ctx tail tokens: {' | '.join(tail_tokens)}")

            if do_steer:
                ps, log_ps, steer_choice, _inj2, _dbg2 = _score_one_prompt(ctx_prompt, steer=True)
            else:
                ps, log_ps = pb, log_pb
                steer_choice = ctx_choice

            ctx_base_choices.append("ABCD"[ctx_choice])
            ctx_steer_choices.append("ABCD"[steer_choice])

            # KLs are always relative to CLEAN p0
            kl_without.append(float(torch.sum(p0 * (log_p0 - log_pb)).item()))
            kl_with.append(float(torch.sum(p0 * (log_p0 - log_ps)).item()))

            changed += int(ctx_choice != clean_choice)
            ctx_acc_base += int(ctx_choice == gold_idx)
            ctx_acc_steer += int(steer_choice == gold_idx)
            restored += int((ctx_choice != clean_choice) and (steer_choice == clean_choice))
            restored_to_gold += int((ctx_choice != gold_idx) and (steer_choice == gold_idx))
            total_pairs += 1

        if answers_out:
            answers_rows.append(
                {
                    "prompt": ex.prompt,
                    "label": ex.label,
                    "clean_base": "ABCD"[clean_choice],
                    "clean_steer": "ABCD"[steer_clean_choice],
                    "contexts": list(ex.contexts),
                    "context_base": ctx_base_choices,
                    "context_steer": ctx_steer_choices,
                }
            )

    if answers_out:
        with open(answers_out, "w", encoding="utf-8") as f:
            for row in answers_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    clean_count = max(len(examples), 1)
    pair_count = max(total_pairs, 1)

    return {
        "clean_examples": float(len(examples)),
        "pairs": float(total_pairs),
        "clean_acc_base": clean_acc_base / clean_count,
        "clean_acc_steer": clean_acc_steer / clean_count,
        "clean_changed_pct": 100.0 * clean_changed / clean_count,
        "ctx_acc_base": ctx_acc_base / pair_count if total_pairs else 0.0,
        "ctx_acc_steer": ctx_acc_steer / pair_count if total_pairs else 0.0,
        "kl_clean": float(sum(kl_clean) / max(len(kl_clean), 1)),
        "kl_without": float(sum(kl_without) / max(len(kl_without), 1)),
        "kl_with": float(sum(kl_with) / max(len(kl_with), 1)),
        "changed_pct": 100.0 * changed / pair_count if total_pairs else 0.0,
        "restored_pct": 100.0 * restored / pair_count if total_pairs else 0.0,
        "restored_to_gold_pct": 100.0 * restored_to_gold / pair_count if total_pairs else 0.0,
    }
