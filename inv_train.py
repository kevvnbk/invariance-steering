#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import gc
from typing import Any, Callable, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from invariance_steering_lib import (
    ChoiceScorer,
    MultiChoiceScorer,
    QAExample,
    NegativeContext,
    SingleTokenChoiceScorer,
    build_flip_mined_examples,
    build_training_examples,
    format_prompt,
    load_flip_mined_examples,
    save_control_vector,
    save_flip_mined_examples,
    set_seed,
)
from dataset_utils import load_dataset_rows, load_triviaqa_paragraphs

import bm25s
import Stemmer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x, **kwargs):
        return x


class InvarianceSteeringTrainer:
    """Learn a single steering vector injected at a chosen transformer block."""

    def __init__(
        self,
        model_name: str,
        steer_scale: float = 0.3,
        lr: float = 5e-3,
        max_length: int = 2048,
        device: str | None = None,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        choice_scorer: str = "multi",
        steer_layer: int = -1,
        steer_span: bool = True,
        thinking: bool = False,
        final_answer_cue: str | None = "Final answer: ",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device or "auto",
            trust_remote_code=True,
        )
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.steer_scale = steer_scale
        self.grad_clip = grad_clip
        self.max_length = max_length
        self.steer_span = steer_span
        self.thinking = thinking
        self.final_answer_cue = final_answer_cue

        hidden_size = self.model.config.hidden_size
        self.vector = torch.nn.Parameter(torch.zeros(hidden_size, device=self.device))
        self.optimizer = torch.optim.AdamW([self.vector], lr=lr, weight_decay=weight_decay)

        self.choice_mode = choice_scorer
        if self.choice_mode == "single":
            self.choice_scorer = SingleTokenChoiceScorer(self.tokenizer)
        else:
            self.choice_scorer = MultiChoiceScorer(self.tokenizer)

        self.blocks = self._locate_transformer_blocks()
        if not self.blocks:
            raise RuntimeError("Could not locate transformer blocks for this model architecture.")

        if steer_layer < 0:
            steer_layer = len(self.blocks) - 1
        if steer_layer < 0 or steer_layer >= len(self.blocks):
            raise ValueError(f"Invalid steer_layer={steer_layer}. Model has {len(self.blocks)} blocks.")
        self.steer_layer = steer_layer

    def _locate_transformer_blocks(self) -> List[torch.nn.Module]:
        if hasattr(self.model, "model"):
            inner = getattr(self.model, "model")
            if hasattr(inner, "layers"):
                layers = getattr(inner, "layers")
                if isinstance(layers, (list, torch.nn.ModuleList)):
                    return list(layers)
        if hasattr(self.model, "transformer"):
            tr = getattr(self.model, "transformer")
            if hasattr(tr, "h"):
                h = getattr(tr, "h")
                if isinstance(h, (list, torch.nn.ModuleList)):
                    return list(h)
        return []

    def _apply_steer_hook(self, inject_indices: torch.Tensor, steer_scale: float, attention_mask: torch.Tensor | None = None):
        """Pre-hook a specific block input and add steer vector at the selected token positions per batch."""
        block = self.blocks[self.steer_layer]
        steer_vec = self.vector

        def hook_fn(_module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(h, torch.Tensor):
                return args, kwargs

            sv = (steer_scale * steer_vec).to(device=h.device, dtype=h.dtype)

            if self.steer_span:
                # "Boundary injection": add steering from inject_idx through end of sequence
                if h.dim() != 3 or inject_indices.numel() != h.size(0):
                    return args, kwargs
                batch_size, seq_len, hidden_dim = h.shape

                idx = inject_indices.to(device=h.device).clamp(0, seq_len - 1)  # [B]

                # mask[b, t] = 1 if t >= idx[b], else 0
                t = torch.arange(seq_len, device=h.device).unsqueeze(0)          # [1, S]
                mask = (t >= idx.unsqueeze(1)).to(h.dtype)                       # [B, S]

                if attention_mask is not None:
                    am = attention_mask.to(device=h.device, dtype=h.dtype)  # [B, S]
                    mask = mask * am

                h2 = h + mask.unsqueeze(-1) * sv.view(1, 1, hidden_dim)
            else:
                # broadcast to all tokens (old behavior)
                h2 = h + sv

            if args:
                if len(args) == 1:
                    return (h2,), kwargs
                return (h2, *args[1:]), kwargs
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = h2
            return args, new_kwargs

        return block.register_forward_pre_hook(hook_fn, with_kwargs=True)

    def _format_for_model(self, prompt: str, add_generation_prompt: bool = False) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=False,
                )
            except TypeError:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                except Exception:
                    return prompt
            except Exception:
                return prompt
        return prompt

    def _first_token_after_context_index(
        self,
        formatted_prompt: str,
        max_length: int,
    ) -> int:
        marker = "Question:\n"
        idx = formatted_prompt.find(marker)
        if idx < 0:
            return 0

        prefix = formatted_prompt[: idx + len(marker)]
        toks = self.tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return toks["input_ids"].shape[1]

    def _next_token_logprobs_with_optional_steer(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        steer_scale: float,
        steer_index: int | None = None,
    ) -> torch.Tensor:
        input_ids = prompt_ids.unsqueeze(0).to(self.device)
        attn = attention_mask.unsqueeze(0).to(self.device)
        seq_len = int(attn.sum(dim=1).item())
        if steer_index is None:
            inject_indices = (attn.sum(dim=1) - 1).long()
        else:
            idx = max(min(steer_index, max(seq_len - 1, 0)), 0)
            inject_indices = torch.tensor([idx], device=self.device, dtype=torch.long)

        handle = None
        if steer_scale is not None and steer_scale != 0.0:
            handle = self._apply_steer_hook(inject_indices=inject_indices, steer_scale=steer_scale, attention_mask=attn)

        out = self.model(input_ids=input_ids, attention_mask=attn)
        if handle is not None:
            handle.remove()

        logits = out.logits[:, -1, :].to(torch.float32)
        return torch.log_softmax(logits, dim=-1)[0]

    def _choice_logprobs(self, prompts: Sequence[str], steer_scale: float | None) -> torch.Tensor:
        formatted = [self._format_for_model(p, add_generation_prompt=False) for p in prompts]
        toks = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        attn = toks["attention_mask"]
        last_indices = (attn.sum(dim=1) - 1).long()
        inject_indices = torch.tensor(
            [self._first_token_after_context_index(p, self.max_length) for p in formatted],
            device=self.device,
            dtype=torch.long,
        )
        seq_lens = attn.sum(dim=1).long()
        inject_indices = torch.minimum(inject_indices, torch.clamp(seq_lens - 1, min=0))

        handle = None
        if steer_scale is not None and steer_scale != 0.0:
            handle = self._apply_steer_hook(inject_indices=inject_indices, steer_scale=steer_scale, attention_mask=attn)

        out = self.model(**toks)
        if handle is not None:
            handle.remove()

        logits = out.logits
        row = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[row, last_indices, :].to(torch.float32)
        return torch.log_softmax(last_logits, dim=-1)

    def choice_probs(self, prompt: str, steer_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.choice_mode == "multi":
            return self.choice_scorer.score_prompt(self, prompt=prompt, steer_scale=steer_scale)

        log_probs = self._choice_logprobs([prompt], steer_scale=steer_scale)[0]
        formatted = self._format_for_model(prompt, add_generation_prompt=False)
        return self.choice_scorer.project(log_probs, prompt=formatted)

    def debug_last_token(self, prompt: str) -> Dict[str, Any]:
        formatted = self._format_for_model(prompt, add_generation_prompt=False)
        toks = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=self.max_length)
        input_ids = toks["input_ids"][0]
        last_idx = int(toks["attention_mask"][0].sum().item() - 1)
        tail_ids = input_ids[max(0, last_idx - 3) : last_idx + 1].tolist()
        tail_str = [self.tokenizer.decode([tid]) for tid in tail_ids]
        return {"last_index": last_idx, "tail_token_ids": tail_ids, "tail_tokens": tail_str}

    def batch_step_loss(
        self,
        batch: Sequence[QAExample],
        clean_penalty: str = "kl",
        clean_weight: float = 1.0,
        gold_weight: float = 0.1,
        target_norm: float = 0.0,
        debug_hook: Callable[[Dict[str, Any]], None] | None = None,
    ) -> tuple[float, Dict[str, float]]:
        if len(batch) == 0:
            return 0.0, {
                "pairs": 0.0,
                "kl_contam_sum": 0.0,
                "kl_steer_sum": 0.0,
                "changed": 0.0,
                "restored": 0.0,
                "restored_to_gold": 0.0,
                "neg_acc_base": 0.0,
                "neg_acc_steer": 0.0,
                "clean_acc_base": 0.0,
                "clean_acc_steer": 0.0,
                "clean_changed": 0.0,
                "pos_count": 0.0,
            }

        clean_prompts = [
            format_prompt(
                ex.prompt,
                context=None,
                thinking=self.thinking,
                final_answer_cue=self.final_answer_cue,
            )
            for ex in batch
        ]
        gold_indices: List[int | None] = [
            ("ABCD".index(ex.answer) if ex.answer in {"A", "B", "C", "D"} else None) for ex in batch
        ]
        positive_prompts: List[str] = []
        positive_owner: List[int] = []
        for i, ex in enumerate(batch):
            if ex.positive_context:
                positive_prompts.append(
                    format_prompt(
                        ex.prompt,
                        context=ex.positive_context,
                        thinking=self.thinking,
                        final_answer_cue=self.final_answer_cue,
                    )
                )
                positive_owner.append(i)

        if self.choice_mode == "single":
            formatted_clean = [self._format_for_model(p, add_generation_prompt=False) for p in clean_prompts]

            with torch.no_grad():
                base_lp_batch = self._choice_logprobs(clean_prompts, steer_scale=0.0)
            steer_lp_batch = self._choice_logprobs(clean_prompts, steer_scale=self.steer_scale)

            p0_list, log_p0_list, ps0_list, log_ps0_list = [], [], [], []
            for i in range(base_lp_batch.size(0)):
                p0_i, lp0_i = self.choice_scorer.project(base_lp_batch[i], prompt=formatted_clean[i])
                ps0_i, lps0_i = self.choice_scorer.project(steer_lp_batch[i], prompt=formatted_clean[i])
                p0_list.append(p0_i.detach())
                log_p0_list.append(lp0_i.detach())
                ps0_list.append(ps0_i)
                log_ps0_list.append(lps0_i)

        else:
            p0_list, log_p0_list, ps0_list, log_ps0_list = [], [], [], []
            for cp in clean_prompts:
                with torch.no_grad():
                    p0_i, lp0_i = self.choice_probs(cp, steer_scale=0.0)
                ps0_i, lps0_i = self.choice_probs(cp, steer_scale=self.steer_scale)
                p0_list.append(p0_i.detach())
                log_p0_list.append(lp0_i.detach())
                ps0_list.append(ps0_i)
                log_ps0_list.append(lps0_i)

        clean_changed = 0
        clean_acc_base = 0
        clean_acc_steer = 0
        clean_loss = torch.tensor(0.0, device=self.device)
        for i in range(len(batch)):
            base_choice = int(torch.argmax(p0_list[i]).item())
            steer_clean_choice = int(torch.argmax(ps0_list[i]).item())
            clean_changed += int(steer_clean_choice != base_choice)
            gi = gold_indices[i]
            if gi is not None:
                clean_acc_base += int(base_choice == gi)
                clean_acc_steer += int(steer_clean_choice == gi)
            if clean_weight > 0:
                if clean_penalty == "kl":
                    clean_loss = clean_loss + torch.sum(p0_list[i] * (log_p0_list[i] - log_ps0_list[i]))
                elif clean_penalty == "ce":
                    clean_loss = clean_loss + (-log_ps0_list[i][base_choice])

        pos_loss = torch.tensor(0.0, device=self.device)
        pos_count = 0
        if positive_prompts and gold_weight > 0:
            if self.choice_mode == "single":
                formatted_pos = [self._format_for_model(p, add_generation_prompt=False) for p in positive_prompts]
                pos_lp_batch = self._choice_logprobs(positive_prompts, steer_scale=self.steer_scale)
                for i in range(pos_lp_batch.size(0)):
                    p_pos, log_p_pos = self.choice_scorer.project(pos_lp_batch[i], prompt=formatted_pos[i])
                    owner = positive_owner[i]
                    gi = gold_indices[owner]
                    if gi is not None:
                        pos_loss = pos_loss + (-log_p_pos[gi])
                        pos_count += 1
            else:
                for i, prompt in enumerate(positive_prompts):
                    _p_pos, log_p_pos = self.choice_probs(prompt, steer_scale=self.steer_scale)
                    owner = positive_owner[i]
                    gi = gold_indices[owner]
                    if gi is not None:
                        pos_loss = pos_loss + (-log_p_pos[gi])
                        pos_count += 1

        ctx_prompts: List[str] = []
        ctx_owner: List[int] = []
        ctx_weights: List[float] = []
        for i, ex in enumerate(batch):
            for ctx in ex.negatives:
                ctx_prompts.append(
                    format_prompt(
                        ex.prompt,
                        context=ctx.text,
                        thinking=self.thinking,
                        final_answer_cue=self.final_answer_cue,
                    )
                )
                ctx_owner.append(i)
                ctx_weights.append(float(ctx.weight))

        total_pairs = len(ctx_prompts)
        if total_pairs == 0:
            total_loss = torch.tensor(0.0, device=self.device)
            if clean_weight > 0 and clean_penalty != "none":
                total_loss = total_loss + clean_weight * (clean_loss / max(len(batch), 1))
            if pos_count > 0 and gold_weight > 0:
                total_loss = total_loss + gold_weight * (pos_loss / max(pos_count, 1))
            total_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_([self.vector], self.grad_clip)
            self.optimizer.step()
            if target_norm and target_norm > 0:
                with torch.no_grad():
                    vnorm = self.vector.norm()
                    if vnorm > target_norm:
                        self.vector.mul_(target_norm / (vnorm + 1e-12))
            self.optimizer.zero_grad()
            return float(total_loss.detach().cpu()), {
                "pairs": 0.0,
                "pair_weight": 0.0,
                "kl_contam_sum": 0.0,
                "kl_steer_sum": 0.0,
                "changed": 0.0,
                "restored": 0.0,
                "restored_to_gold": 0.0,
                "neg_acc_base": 0.0,
                "neg_acc_steer": 0.0,
                "clean_acc_base": float(clean_acc_base),
                "clean_acc_steer": float(clean_acc_steer),
                "clean_changed": float(clean_changed),
                "pos_count": float(pos_count),
            }

        if self.choice_mode == "single":
            with torch.no_grad():
                ctx_base_lp = self._choice_logprobs(ctx_prompts, steer_scale=0.0)
            ctx_steer_lp = self._choice_logprobs(ctx_prompts, steer_scale=self.steer_scale)
            pb_list: List[torch.Tensor] = []
            log_pb_list: List[torch.Tensor] = []
            pw_list: List[torch.Tensor] = []
            log_pw_list: List[torch.Tensor] = []
            
            formatted_ctx = [self._format_for_model(p, add_generation_prompt=False) for p in ctx_prompts]
            for i in range(ctx_base_lp.size(0)):
                pb_i, lpb_i = self.choice_scorer.project(ctx_base_lp[i], prompt=formatted_ctx[i])
                pw_i, lpw_i = self.choice_scorer.project(ctx_steer_lp[i], prompt=formatted_ctx[i])
                pb_list.append(pb_i.detach())
                log_pb_list.append(lpb_i.detach())
                pw_list.append(pw_i)
                log_pw_list.append(lpw_i)
        else:
            pb_list, log_pb_list, pw_list, log_pw_list = [], [], [], []
            for prompt in ctx_prompts:
                with torch.no_grad():
                    pb_i, lpb_i = self.choice_probs(prompt, steer_scale=0.0)
                    pb_i = pb_i.detach()
                    lpb_i = lpb_i.detach()
                pw_i, lpw_i = self.choice_probs(prompt, steer_scale=self.steer_scale)
                pb_list.append(pb_i)
                log_pb_list.append(lpb_i)
                pw_list.append(pw_i)
                log_pw_list.append(lpw_i)

        total_loss = torch.tensor(0.0, device=self.device)
        kl_contam_sum = 0.0
        kl_steer_sum = 0.0
        changed = 0
        restored = 0
        restored_to_gold = 0
        neg_acc_base = 0
        neg_acc_steer = 0

        total_weight = float(sum(ctx_weights)) if ctx_weights else float(total_pairs)
        for j in range(total_pairs):
            owner = ctx_owner[j]
            weight = ctx_weights[j]
            p0 = p0_list[owner]
            log_p0 = log_p0_list[owner]
            pb = pb_list[j]
            log_pb = log_pb_list[j]
            pw = pw_list[j]
            log_pw = log_pw_list[j]
            # Invariance loss: KL(p0 || pw)
            # p0 = clean prompt, unsteered (no grad)
            # pw = context prompt, steered (with grad)
            # Goal: under context, steering should recover original clean behavior
            kl_invariance = torch.sum(p0 * (log_p0 - log_pw))
            loss = kl_invariance

            kl_contam = torch.sum(p0 * (log_p0 - log_pb))
            kl_contam_sum += float((kl_contam * weight).detach().cpu())
            kl_steer_sum += float((kl_invariance * weight).detach().cpu())

            base_choice = int(torch.argmax(p0).item())
            ctx_choice = int(torch.argmax(pb).item())
            steer_choice = int(torch.argmax(pw).item())
            changed += int(ctx_choice != base_choice)
            restored += int((ctx_choice != base_choice) and (steer_choice == base_choice))

            gi = gold_indices[owner]
            if gi is not None:
                neg_acc_base += int(ctx_choice == gi)
                neg_acc_steer += int(steer_choice == gi)
                restored_to_gold += int((ctx_choice != gi) and (steer_choice == gi))

            if debug_hook and j < 1:
                debug_hook(
                    {
                        "base_prompt": clean_prompts[owner],
                        "context": ctx_prompts[j],
                        "p0": p0.detach().cpu(),
                        "pw": pw.detach().cpu(),
                        "loss": float(loss.detach().cpu()),
                        "last_prompt_debug": self.debug_last_token(clean_prompts[owner]),
                    }
                )

            total_loss = total_loss + (weight * loss)

        total_loss = total_loss / max(total_weight, 1.0)

        if clean_weight > 0 and clean_penalty != "none":
            total_loss = total_loss + clean_weight * (clean_loss / max(len(batch), 1))
        if pos_count > 0 and gold_weight > 0:
            total_loss = total_loss + gold_weight * (pos_loss / max(pos_count, 1))

        total_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_([self.vector], self.grad_clip)
        self.optimizer.step()
        if target_norm and target_norm > 0:
            with torch.no_grad():
                vnorm = self.vector.norm()
                if vnorm > target_norm:
                    self.vector.mul_(target_norm / (vnorm + 1e-12))
        self.optimizer.zero_grad()

        metrics = {
            "pairs": float(total_pairs),
            "pair_weight": float(total_weight),
            "kl_contam_sum": kl_contam_sum,
            "kl_steer_sum": kl_steer_sum,
            "changed": float(changed),
            "restored": float(restored),
            "restored_to_gold": float(restored_to_gold),
            "neg_acc_base": float(neg_acc_base),
            "neg_acc_steer": float(neg_acc_steer),
            "clean_acc_base": float(clean_acc_base),
            "clean_acc_steer": float(clean_acc_steer),
            "clean_changed": float(clean_changed),
            "pos_count": float(pos_count),
        }
        return float(total_loss.detach().cpu()), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an invariance steering vector with retriever-based hard contexts.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="barexam_train.jsonl",
        help="QA dataset JSONL (optionally '<dataset>:/path').",
    )
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split for HF datasets.")
    parser.add_argument("--model", type=str, required=True, help="HF model id or path for training.")
    parser.add_argument("--retriever", type=str, default="barexamqa_train_passage_bm25", help="bm25s index path.")
    parser.add_argument("--max-samples", type=int, default=200, help="Limit dataset rows.")
    parser.add_argument("--retrieval-k", type=int, default=8, help="Top-k passages to retrieve.")
    parser.add_argument("--max-negatives", type=int, default=4, help="Contexts per question.")
    parser.add_argument("--flip-min-negatives", type=int, default=1, help="Minimum flip-mined contexts per question.")
    parser.add_argument("--flip-fallback-top-r", type=int, default=2, help="Soft contexts to add when flips are few.")
    parser.add_argument("--flip-soft-weight", type=float, default=0.3, help="Loss weight for soft contexts.")
    parser.add_argument(
        "--flip-fallback-metric",
        type=str,
        choices=["gold_drop", "kl"],
        default="gold_drop",
        help="Ranking metric for soft contexts.",
    )
    parser.add_argument(
        "--flip-cache-path",
        type=str,
        default="flip_mined_contexts.jsonl",
        help="Cache path for flip-mined contexts.",
    )
    parser.add_argument("--flip-cache-rebuild", action="store_true", help="Rebuild flip-mined contexts cache.")
    parser.add_argument(
        "--use-original-contexts",
        action="store_true",
        help="Train with original BM25 contexts instead of flip-mined contexts.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs over the dataset.")
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size (QA examples per step).")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for the steering vector.")
    parser.add_argument("--steer-scale", type=float, default=0.1, help="Scale applied to the steering vector.")
    parser.add_argument("--steer-layer", type=int, default=-1, help="Transformer block index to steer at (-1 = last).")
    parser.add_argument(
        "--steer-span",
        action="store_true",
        dest="steer_span",
        help="Apply steering vector from the context->question boundary through end of sequence (default).",
    )
    parser.add_argument(
        "--steer-broadcast",
        action="store_false",
        dest="steer_span",
        help="Apply steering vector to the full sequence.",
    )
    parser.set_defaults(steer_span=True)
    parser.add_argument(
        "--steer-at-boundary",
        action="store_true",
        dest="steer_span",
        help="Deprecated: use --steer-span.",
    )
    parser.add_argument(
        "--clean-penalty",
        type=str,
        choices=["kl", "ce", "none"],
        default="kl",
        help="Penalty for clean prompts: KL(p0 || p_clean_steered) or CE to base argmax.",
    )
    parser.add_argument("--clean-weight", type=float, default=0.2, help="Weight for clean do-no-harm KL (beta).")
    parser.add_argument(
        "--gold-weight",
        type=float,
        default=0.1,
        help="Weight for positive supervision on gold context (alpha).",
    )
    parser.add_argument("--choice-scorer", type=str, choices=["single", "multi"], default="single")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument(
        "--vector-norm",
        type=float,
        default=0.0,
        help="Project steering vector to a fixed norm after each step (0 disables).",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length for tokenization.")
    parser.add_argument("--vector-out", type=str, default="invariance_steer.gguf", help="Output path for gguf vector.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--debug-limit", type=int, default=3)
    parser.add_argument("--debug-train-steps", type=int, default=0)
    parser.add_argument(
        "--triviaqa-contexts-per-example",
        type=int,
        default=0,
        help="Add N random TriviaQA paragraphs per example (0 disables).",
    )
    parser.add_argument(
        "--triviaqa-split",
        type=str,
        default="train",
        help="TriviaQA split to sample paragraphs from.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace, wandb_run=None) -> InvarianceSteeringTrainer:
    set_seed(0)

    rows = load_dataset_rows(args.dataset, max_samples=args.max_samples, split=args.dataset_split)
    if not rows:
        raise ValueError(f"No rows loaded from {args.dataset}")
    print(f"Loaded {len(rows)} rows from {args.dataset}")

    retriever = bm25s.BM25.load(args.retriever, load_corpus=True)
    stemmer = Stemmer.Stemmer("english")

    triviaqa_paragraphs: List[str] = []
    if args.triviaqa_contexts_per_example > 0:
        triviaqa_paragraphs = load_triviaqa_paragraphs()
        if not triviaqa_paragraphs:
            raise ValueError("TriviaQA paragraphs are empty; set --triviaqa-contexts-per-example=0 to disable.")
        print(f"Loaded {len(triviaqa_paragraphs)} TriviaQA paragraphs for contamination")

    trainer = InvarianceSteeringTrainer(
        model_name=args.model,
        steer_scale=args.steer_scale,
        lr=args.lr,
        max_length=args.max_length,
        grad_clip=args.grad_clip,
        choice_scorer=args.choice_scorer,
        steer_layer=args.steer_layer,
        steer_span=args.steer_span,
    )

    if args.use_original_contexts:
        examples = build_training_examples(
            rows,
            retriever=retriever,
            stemmer=stemmer,
            k=args.retrieval_k,
            max_negatives=args.max_negatives,
            debug=args.debug,
            debug_limit=args.debug_limit,
        )
        if not examples:
            raise ValueError("No training examples with BM25 contexts were found.")
        print(f"Prepared {len(examples)} training examples with BM25 contexts")
    else:
        cache_path = args.flip_cache_path
        if cache_path and os.path.exists(cache_path) and not args.flip_cache_rebuild:
            examples = load_flip_mined_examples(cache_path)
            print(f"Loaded {len(examples)} flip-mined contexts from {cache_path}")
        else:
            examples = build_flip_mined_examples(
                rows,
                retriever=retriever,
                stemmer=stemmer,
                trainer=trainer,
                k_pool=args.retrieval_k,
                max_negatives=args.max_negatives,
                min_flips=args.flip_min_negatives,
                fallback_top_r=args.flip_fallback_top_r,
                soft_weight=args.flip_soft_weight,
                fallback_metric=args.flip_fallback_metric,
                debug=args.debug,
                debug_limit=args.debug_limit,
            )
            if cache_path:
                save_flip_mined_examples(cache_path, examples)
                print(f"Saved flip-mined contexts to {cache_path}")
        if not examples:
            raise ValueError("No training examples with flip-mined contexts were found.")
        print(f"Prepared {len(examples)} training examples with flip-mined contexts")

    step_counter = 0

    if args.debug and examples:
        sample = examples[0]
        clean_prompt = format_prompt(
            sample.prompt,
            context=None,
            thinking=trainer.thinking,
            final_answer_cue=trainer.final_answer_cue,
        )
        formatted = trainer._format_for_model(clean_prompt, add_generation_prompt=True)
        tail = formatted[-100:] if len(formatted) > 100 else formatted
        has_options = all(f"{c})" in formatted for c in "ABCD")
        has_final_answer = "Final answer:" in formatted
        toks = trainer.tokenizer(formatted, return_tensors="pt").to(trainer.device)
        with torch.no_grad():
            gen = trainer.model.generate(
                **toks,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=trainer.tokenizer.eos_token_id,
            )
        decoded = trainer.tokenizer.decode(gen[0], skip_special_tokens=True)
        response = decoded[len(formatted) :].strip() if decoded.startswith(formatted) else decoded.strip()
        print("[debug][first-output] model response:")
        print(response)
        print("[debug][diag] formatted tail (last 100 chars):", tail)
        print("[debug][diag] formatted has options A/B/C/D:", has_options)
        print("[debug][diag] formatted has 'Final answer:':", has_final_answer)
        print("[debug][diag] last-token info:", trainer.debug_last_token(clean_prompt))
        with torch.no_grad():
            orig_vec = trainer.vector.detach().clone()
            trainer.vector.fill_(0.5)

            p_clean, _ = trainer.choice_probs(clean_prompt, steer_scale=0.0)
            p_clean_steer, _ = trainer.choice_probs(clean_prompt, steer_scale=trainer.steer_scale)
            delta_clean = float(torch.max(torch.abs(p_clean_steer - p_clean)).item())
            print(f"[debug][diag] clean max prob delta: {delta_clean:.6f}")

            toks = trainer.tokenizer(formatted, return_tensors="pt").to(trainer.device)
            attn = toks["attention_mask"]
            seq_len = int(attn.sum(dim=1).item())
            first_idx = trainer._first_token_after_context_index(clean_prompt, trainer.max_length)
            first_idx = max(min(first_idx, max(seq_len - 1, 0)), 0)
            last_idx = torch.tensor([first_idx], device=trainer.device, dtype=torch.long)
            captured = {}

            def cap_hook(_module, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                if isinstance(h, torch.Tensor):
                    captured["pre"] = h.detach().clone()
                return args, kwargs

            h0 = trainer.blocks[trainer.steer_layer].register_forward_pre_hook(cap_hook, with_kwargs=True)
            trainer.model(**toks)
            h0.remove()
            base_h = captured.get("pre")

            captured = {}
            h1 = trainer.blocks[trainer.steer_layer].register_forward_pre_hook(cap_hook, with_kwargs=True)
            steer_h = trainer._apply_steer_hook(inject_indices=last_idx, steer_scale=trainer.steer_scale)
            trainer.model(**toks)
            steer_h.remove()
            h1.remove()
            steered_h = captured.get("pre")
            if base_h is not None and steered_h is not None:
                delta_vec = (steered_h - base_h)[0, int(last_idx[0].item()), :].norm().item()
                print(f"[debug][diag] hook delta norm at target token: {delta_vec:.6f}")

            layer_candidates = [0, len(trainer.blocks) // 2, len(trainer.blocks) - 1]
            layer_candidates = [l for l in layer_candidates if 0 <= l < len(trainer.blocks)]
            base_layer = trainer.steer_layer
            for l in layer_candidates:
                trainer.steer_layer = l
                p_l, _ = trainer.choice_probs(clean_prompt, steer_scale=trainer.steer_scale)
                delta_l = float(torch.max(torch.abs(p_l - p_clean)).item())
                print(f"[debug][diag] layer {l} max prob delta: {delta_l:.6f}")
            trainer.steer_layer = base_layer

            p_strong, _ = trainer.choice_probs(clean_prompt, steer_scale=1.0)
            delta_strong = float(torch.max(torch.abs(p_strong - p_clean)).item())
            print(f"[debug][diag] steer_scale=1.0 max prob delta: {delta_strong:.6f}")

            trainer.vector.copy_(orig_vec)

        def training_debug_hook(payload):
            nonlocal step_counter
            step_counter += 1
            if args.debug_train_steps == 0 or step_counter > args.debug_train_steps:
                return
            p0 = [float(x) for x in payload["p0"]]
            pw = [float(x) for x in payload["pw"]]
            base_probs = ", ".join(f"{c}:{p:.3f}" for c, p in zip("ABCD", p0))
            steered_probs = ", ".join(f"{c}:{p:.3f}" for c, p in zip("ABCD", pw))
            ctx_snippet = payload["context"].replace("\n", " ")[:160]
            print(f"[debug][step {step_counter}] loss={payload['loss']:.4f} base[{base_probs}] steered[{steered_probs}]")
            print(f"         context snippet: {ctx_snippet}")
            lp = payload.get("last_prompt_debug")
            if isinstance(lp, dict):
                tail_tokens = lp.get("tail_tokens")
                last_index = lp.get("last_index")
                if isinstance(tail_tokens, list) and last_index is not None:
                    tail_str = " | ".join(str(t) for t in tail_tokens)
                    print(f"         last prompt idx: {last_index} | tail tokens: {tail_str}")

    def add_triviaqa_contexts(
        batch: Sequence[QAExample],
        paragraphs: Sequence[str],
        n_per_example: int,
        weight: float,
    ) -> List[QAExample]:
        if n_per_example <= 0 or not paragraphs:
            return list(batch)
        import random

        augmented: List[QAExample] = []
        for ex in batch:
            extras: List[NegativeContext] = []
            for _ in range(n_per_example):
                ctx = random.choice(paragraphs)
                extras.append(NegativeContext(text=ctx, weight=weight, is_soft=True))
            augmented.append(
                QAExample(
                    prompt=ex.prompt,
                    negatives=ex.negatives + extras,
                    answer=ex.answer,
                    positive_context=ex.positive_context,
                )
            )
        return augmented

    for epoch in range(args.epochs):
        import random

        random.shuffle(examples)
        running = 0.0
        loss_steps = 0.0
        total_pairs = 0.0
        total_pair_weight = 0.0
        kl_contam_sum = 0.0
        kl_steer_sum = 0.0
        changed = 0.0
        restored = 0.0
        restored_to_gold = 0.0
        neg_acc_base = 0.0
        neg_acc_steer = 0.0
        clean_acc_base = 0.0
        clean_acc_steer = 0.0
        clean_changed = 0.0

        bs = max(1, args.batch_size)
        total_steps = (len(examples) + bs - 1) // bs

        for start in tqdm(
            range(0, len(examples), bs),
            total=total_steps,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False,
        ):
            batch = examples[start : start + bs]
            if epoch >= 1 and args.triviaqa_contexts_per_example > 0:
                batch = add_triviaqa_contexts(
                    batch=batch,
                    paragraphs=triviaqa_paragraphs,
                    n_per_example=args.triviaqa_contexts_per_example,
                    weight=0.1,
                )
            loss_val, metrics = trainer.batch_step_loss(
                batch=batch,
                clean_penalty=args.clean_penalty,
                clean_weight=args.clean_weight,
                gold_weight=args.gold_weight,
                target_norm=args.vector_norm,
                debug_hook=training_debug_hook if args.debug and args.debug_train_steps > 0 else None,
            )
            running += loss_val
            loss_steps += 1.0
            total_pairs += metrics["pairs"]
            total_pair_weight += metrics.get("pair_weight", metrics["pairs"])
            kl_contam_sum += metrics["kl_contam_sum"]
            kl_steer_sum += metrics["kl_steer_sum"]
            changed += metrics["changed"]
            restored += metrics["restored"]
            restored_to_gold += metrics["restored_to_gold"]
            neg_acc_base += metrics["neg_acc_base"]
            neg_acc_steer += metrics["neg_acc_steer"]
            clean_acc_base += metrics["clean_acc_base"]
            clean_acc_steer += metrics["clean_acc_steer"]
            clean_changed += metrics["clean_changed"]

        avg_loss = running / max(loss_steps, 1.0)
        contam_kl = kl_contam_sum / max(total_pair_weight, 1.0)
        steer_kl = kl_steer_sum / max(total_pair_weight, 1.0)

        print(f"[Epoch {epoch+1}/{args.epochs}] avg loss: {avg_loss:.4f}")
        print(
            f"    KL clean->context: {contam_kl:.4f} | KL clean->steered: {steer_kl:.4f} | "
            f"changed by negs: {100.0 * changed / max(total_pairs, 1.0):.1f}% | "
            f"restored: {100.0 * restored / max(total_pairs, 1.0):.1f}% | "
            f"restored->gold: {100.0 * restored_to_gold / max(total_pairs, 1.0):.1f}%"
        )
        if total_pairs > 0:
            print(f"    Neg accuracy base/steer: {neg_acc_base / total_pairs:.4f} / {neg_acc_steer / total_pairs:.4f}")
        if len(examples) > 0:
            print(
                f"    Clean acc base/steer: {clean_acc_base / len(examples):.4f} / "
                f"{clean_acc_steer / len(examples):.4f} | clean changed: {100.0 * clean_changed / len(examples):.1f}%"
            )
        with torch.no_grad():
            vnorm = float(trainer.vector.norm().detach().cpu())
        print(f"    vector ||v||: {vnorm:.4f} | steer_layer: {trainer.steer_layer}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "avg_loss": avg_loss,
                    "kl_clean_context": contam_kl,
                    "kl_clean_steered": steer_kl,
                    "changed_pct": 100.0 * changed / max(total_pairs, 1.0),
                    "restored_pct": 100.0 * restored / max(total_pairs, 1.0),
                    "restored_to_gold_pct": 100.0 * restored_to_gold / max(total_pairs, 1.0),
                    "neg_acc_base": neg_acc_base / max(total_pairs, 1.0),
                    "neg_acc_steer": neg_acc_steer / max(total_pairs, 1.0),
                    "clean_acc_base": clean_acc_base / max(len(examples), 1.0),
                    "clean_acc_steer": clean_acc_steer / max(len(examples), 1.0),
                    "clean_changed_pct": 100.0 * clean_changed / max(len(examples), 1.0),
                    "vector_norm": vnorm,
                    "epoch": epoch + 1,
                }
            )

    vector_out = args.vector_out
    if wandb_run is not None and vector_out:
        base, ext = os.path.splitext(vector_out)
        vector_out = f"{base}_{wandb_run.id}{ext}"

    save_control_vector(
        trainer.vector,
        model_name=args.model,
        layer=trainer.steer_layer,
        steer_scale=args.steer_scale,
        output_path=vector_out,
        notes="Invariance steering vector trained with BM25 hard contexts (hook-based layer steering).",
    )
    print(f"Saved steering vector to {vector_out}")
    return trainer


def cleanup_trainer(trainer: InvarianceSteeringTrainer) -> None:
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    trainer = run_training(args)
    cleanup_trainer(trainer)


if __name__ == "__main__":
    main()
