#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EASYSTEER_SRC = _REPO_ROOT / "EasySteer"
if _EASYSTEER_SRC.exists() and str(_EASYSTEER_SRC) not in sys.path:
    sys.path.insert(0, str(_EASYSTEER_SRC))
_LEGAL_RAG_SRC = _REPO_ROOT / "legal-rag"
if _LEGAL_RAG_SRC.exists() and str(_LEGAL_RAG_SRC) not in sys.path:
    sys.path.append(str(_LEGAL_RAG_SRC))

import argparse
import os.path
import random
from typing import Any, Dict, List

import bm25s
import Stemmer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easysteer.steer.utils import StatisticalControlVector
from dataset_utils import load_dataset_rows, load_triviaqa_paragraphs
from invariance_steering_lib import load_control_vector
import test_invariance_steering as tis
from test_invariance_steering import (
    SingleTokenChoiceScorer,
    build_context_examples,
    build_mixed_from_flip,
    evaluate,
    filter_flip_contexts,
    locate_transformer_blocks,
    prepare_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a steering vector on barexam QA with clean prompts and BM25 contexts (batched)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="barexam_test.jsonl",
        help="QA dataset JSONL (optionally '<dataset>:/path').",
    )
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split for HF datasets.")
    parser.add_argument("--model", type=str, required=True, help="HF model id/path for evaluation.")
    parser.add_argument("--vector", type=str, required=True, help="Path to gguf/pt steering vector.")
    parser.add_argument("--retriever", type=str, default="barexamqa_test_passage_bm25", help="bm25s index path.")
    parser.add_argument(
        "--retriever-type",
        type=str,
        choices=["bm25", "qwen3-embed"],
        default="bm25",
        help="Retriever backend to use for evaluation contexts.",
    )
    parser.add_argument(
        "--retriever-tsv",
        type=str,
        default=str(_REPO_ROOT / "legal-rag" / "test.tsv"),
        help="TSV document source for Qwen3 embedding retriever.",
    )
    parser.add_argument(
        "--retriever-cache",
        type=str,
        default=str(SCRIPT_DIR / "qwen3_ft_embed_test.pt"),
        help="Cache path for Qwen3 embedding retriever.",
    )
    parser.add_argument(
        "--retriever-embed-model",
        type=str,
        default=str(_REPO_ROOT / "qwen3-embedding-fine-tuned" / "ft_qwen_kd_100p0" / "best_epoch"),
        help="Model id/path for Qwen3 embedding retriever.",
    )
    parser.add_argument("--max-samples", type=int, default=117, help="Max dataset rows to score.")
    parser.add_argument("--retrieval-k", type=int, default=6, help="Top-k contexts per question before filtering.")
    parser.add_argument("--max-negatives", type=int, default=3, help="Contexts per question to evaluate.")
    parser.add_argument("--max-length", type=int, default=8192, help="Max sequence length.")
    parser.add_argument("--steer-scale", type=float, default=None, help="Override steer scale (defaults to metadata or 1.0).")
    parser.add_argument("--device", type=str, default=None, help="Device map for HF model (e.g., 'auto', 'cuda').")
    parser.add_argument(
        "--scale-sweep",
        type=str,
        default=None,
        help="Optional comma-separated list of scales to benchmark (baseline 0 is added automatically).",
    )
    parser.add_argument(
        "--context-configs",
        type=str,
        default="clean,gold-only,rag-topk,rag-topk+injected,flip-only,mixed,gold+rag",
        help="Comma-separated list: clean,gold-only,rag-topk,rag-topk+injected,flip-only,mixed,gold+rag",
    )
    parser.add_argument("--mixed-min-distractors", type=int, default=1, help="Min distractors for mixed config.")
    parser.add_argument("--mixed-max-distractors", type=int, default=3, help="Max distractors for mixed config.")
    parser.add_argument("--answers-out-base", type=str, default="", help="Base path to save per-config answers JSONL.")
    parser.add_argument("--debug", action="store_true", help="Print model outputs for first few examples.")
    # parser.add_argument(
    #     "--system-prompt",
    #     type=str,
    #     default="You are a helpful assistant.",
    #     help="System prompt used for chat templates (matches benchmark_llama.py).",
    # )
    parser.add_argument(
        "--inject-triviaqa-contexts-per-example",
        type=int,
        default=0,
        help="Inject N random TriviaQA paragraphs for rag-topk+injected (0 disables).",
    )
    parser.add_argument(
        "--inject-triviaqa-split",
        type=str,
        default="train",
        help="TriviaQA split to sample injected paragraphs from.",
    )
    parser.add_argument(
        "--inject-triviaqa-max-paragraphs",
        type=int,
        default=0,
        help="Optional cap on TriviaQA paragraphs loaded (0 = no cap).",
    )
    return parser.parse_args()


# def _install_llama_chat_template(system_prompt: str) -> None:
#     if not system_prompt:
#         return

#     def format_for_model_llama(tokenizer, prompt: str) -> str:
#         if hasattr(tokenizer, "apply_chat_template"):
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt},
#             ]
#             try:
#                 return tokenizer.apply_chat_template(
#                     messages,
#                     tokenize=False,
#                     add_generation_prompt=True,
#                 )
#             except Exception:
#                 try:
#                     return tokenizer.apply_chat_template(
#                         messages,
#                         tokenize=False,
#                         add_generation_prompt=True,
#                         enable_thinking=False,
#                     )
#                 except Exception:
#                     return prompt
#         return prompt

#     tis.format_for_model = format_for_model_llama


def _load_vector_and_scale(args: argparse.Namespace, trainer=None) -> tuple[Dict[int, torch.Tensor], int, float, List[float]]:
    direction_map: Dict[int, torch.Tensor] = {}
    default_scale = 1.0
    layer_id = None

    if getattr(args, "vector", ""):
        vector_path = args.vector
        ext = os.path.splitext(vector_path)[1].lower()
        if ext == ".pt":
            vector, layer_id, saved_scale = load_control_vector(vector_path)
            direction_map = {int(layer_id): vector}
            default_scale = saved_scale or 1.0
        else:
            control = StatisticalControlVector.import_gguf(vector_path)
            direction_map = {int(k): torch.tensor(v) for k, v in control.directions.items()}
            if control.metadata and "layer" in control.metadata:
                layer_id = int(control.metadata["layer"])
            else:
                layer_id = sorted(direction_map.keys())[0]
            default_scale = float(control.metadata.get("steer_scale", 1.0) if control.metadata else 1.0)
    elif trainer is not None:
        direction_map = {int(trainer.steer_layer): trainer.vector.detach().cpu()}
        layer_id = int(trainer.steer_layer)
        default_scale = float(trainer.steer_scale)
    else:
        raise ValueError("vector is required when no trainer is provided")

    if args.steer_scale is not None:
        default_scale = args.steer_scale

    scale_candidates: List[float] = []
    if args.scale_sweep:
        scale_candidates.extend(float(s.strip()) for s in args.scale_sweep.split(",") if s.strip())
    else:
        scale_candidates.append(default_scale)

    seen_scales = set()
    scales: List[float] = []
    for s in scale_candidates:
        if s not in seen_scales:
            scales.append(s)
            seen_scales.add(s)

    return direction_map, int(layer_id), default_scale, scales


def run_eval(args: argparse.Namespace, trainer=None) -> List[Dict[str, Any]]:
    random.seed(0)
    # if args.model.lower().startswith("meta-llama"):
    #     _install_llama_chat_template(args.system_prompt)

    rows = load_dataset_rows(args.dataset, max_samples=args.max_samples, split=args.dataset_split)
    if not rows:
        raise ValueError(f"No data loaded from {args.dataset}")
    print(f"Loaded {len(rows)} rows from {args.dataset}")

    if args.retriever_type == "qwen3-embed":
        from qwen3_embedding_retriever import Qwen3EmbeddingRetriever

        retriever = Qwen3EmbeddingRetriever(
            tsv_path=args.retriever_tsv,
            cache_path=args.retriever_cache,
            model_id=args.retriever_embed_model,
        )
        stemmer = None
    else:
        retriever = bm25s.BM25.load(args.retriever, load_corpus=True)
        stemmer = Stemmer.Stemmer("english")
    examples = prepare_examples(
        rows,
        retriever=retriever,
        stemmer=stemmer,
        k=args.retrieval_k,
        max_negatives=args.max_negatives,
    )

    if not examples:
        raise ValueError("No evaluatable examples were built.")
    total_pairs = sum(len(ex.contexts) for ex in examples)
    print(f"Prepared {len(examples)} examples with {total_pairs} negative contexts")

    direction_map, layer_id, _default_scale, scales = _load_vector_and_scale(args, trainer=trainer)
    vector_path = getattr(args, "vector", "")
    if vector_path:
        print(f"Loaded steering vector from {vector_path} (layer {layer_id}); scales to test: {scales}")

    owns_model = trainer is None
    if trainer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device or "auto",
            trust_remote_code=True,
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.9,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        blocks = locate_transformer_blocks(model)
        if not blocks:
            raise RuntimeError("Could not locate transformer blocks in this model.")
    else:
        model = trainer.model
        tokenizer = trainer.tokenizer
        blocks = getattr(trainer, "blocks", None) or locate_transformer_blocks(model)

    direction = direction_map[layer_id].to(next(model.parameters()).device)
    choice_scorer = SingleTokenChoiceScorer(tokenizer)

    config_raw = [c.strip() for c in args.context_configs.split(",") if c.strip()]
    configs = []
    seen = set()
    for c in config_raw:
        if c in seen:
            continue
        configs.append(c)
        seen.add(c)

    injected_contexts: List[str] = []
    if "rag-topk+injected" in configs and args.inject_triviaqa_contexts_per_example > 0:
        max_paragraphs = args.inject_triviaqa_max_paragraphs or None
        injected_contexts = load_triviaqa_paragraphs(
            max_paragraphs=max_paragraphs,
            split=args.inject_triviaqa_split,
        )
        if not injected_contexts:
            raise ValueError("TriviaQA paragraphs are empty; set --inject-triviaqa-contexts-per-example=0 to disable.")
        print(f"Loaded {len(injected_contexts)} TriviaQA paragraphs for injected contexts")

    eval_sets: List[tuple[str, List[Any]]] = []
    flip_source_config = "rag-topk"
    if "rag-topk+injected" in configs and args.inject_triviaqa_contexts_per_example > 0:
        flip_source_config = "rag-topk+injected"
    flip_source_examples = build_context_examples(
        examples,
        config=flip_source_config,
        mixed_min=args.mixed_min_distractors,
        mixed_max=args.mixed_max_distractors,
        injected_contexts=injected_contexts if flip_source_config == "rag-topk+injected" else None,
        injected_per_example=args.inject_triviaqa_contexts_per_example if flip_source_config == "rag-topk+injected" else 0,
    )
    for cfg in configs:
        if cfg == "flip-only":
            flip_examples = filter_flip_contexts(
                examples=flip_source_examples,
                model=model,
                tokenizer=tokenizer,
                choice_scorer=choice_scorer,
                blocks=blocks,
                layer_id=layer_id,
                max_length=args.max_length,
                debug=args.debug,
            )
            flip_pairs = sum(len(ex.contexts) for ex in flip_examples)
            print(f"Prepared {len(flip_examples)} flip examples with {flip_pairs} negative contexts")
            eval_sets.append(("flip-only", flip_examples))
        elif cfg == "mixed":
            flip_examples = filter_flip_contexts(
                examples=flip_source_examples,
                model=model,
                tokenizer=tokenizer,
                choice_scorer=choice_scorer,
                blocks=blocks,
                layer_id=layer_id,
                max_length=args.max_length,
                debug=args.debug,
            )
            ctx_examples = build_mixed_from_flip(
                examples=examples,
                flip_examples=flip_examples,
                mixed_min=args.mixed_min_distractors,
                mixed_max=args.mixed_max_distractors,
            )
            pair_count = sum(len(ex.contexts) for ex in ctx_examples)
            print(f"Prepared {len(ctx_examples)} mixed examples with {pair_count} negative contexts")
            eval_sets.append(("mixed", ctx_examples))
        else:
            ctx_examples = build_context_examples(
                examples,
                config=cfg,
                mixed_min=args.mixed_min_distractors,
                mixed_max=args.mixed_max_distractors,
                injected_contexts=injected_contexts if cfg == "rag-topk+injected" else None,
                injected_per_example=args.inject_triviaqa_contexts_per_example if cfg == "rag-topk+injected" else 0,
            )
            pair_count = sum(len(ex.contexts) for ex in ctx_examples)
            print(f"Prepared {len(ctx_examples)} {cfg} examples with {pair_count} negative contexts")
            eval_sets.append((cfg, ctx_examples))

    results: List[Dict[str, Any]] = []
    for scale in scales:
        do_steer = (scale != 0.0)
        steer_vec = direction if do_steer else None
        for label, eval_examples in eval_sets:
            answers_out = None
            if args.answers_out_base:
                base, ext = os.path.splitext(args.answers_out_base)
                ext = ext or ".jsonl"
                scale_tag = f"{scale:.3f}".replace(".", "p")
                answers_out = f"{base}_{label}_s{scale_tag}{ext}"
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                choice_scorer=choice_scorer,
                blocks=blocks,
                steer_vector=steer_vec,
                steer_scale=scale,
                layer_id=layer_id,
                examples=eval_examples,
                max_length=args.max_length,
                debug=args.debug,
                answers_out=answers_out,
            )
            if answers_out:
                print(f"Saved answers to {answers_out}")
            metrics["steer_scale"] = scale
            metrics["eval_set"] = label
            results.append(metrics)

            print(f"\n=== Invariance + Accuracy ({label}) ===")
            print(f"Steer scale:          {scale}")
            if label == "clean":
                acc_base = metrics["clean_acc_base"]
                acc_steer = metrics["clean_acc_steer"]
            else:
                acc_base = metrics["ctx_acc_base"]
                acc_steer = metrics["ctx_acc_steer"]
            print(f"Accuracy:             base={acc_base:.4f} | steered={acc_steer:.4f}")

            if label == "clean":
                print(f"Clean KL shift:       {metrics['kl_clean']:.4f}")
                continue

            print(f"Clean KL shift:       {metrics['kl_clean']:.4f}")
            print(f"KL w/o steering:      {metrics['kl_without']:.4f}")
            print(f"KL with steering:     {metrics['kl_with']:.4f}")

            if label in {"rag-topk", "rag-topk+injected", "flip-only"}:
                print(
                    f"Changed by contexts: {metrics['changed_pct']:.1f}% | "
                    f"Restored (base pred): {metrics['restored_pct']:.1f}% | "
                    f"Restored to gold: {metrics['restored_to_gold_pct']:.1f}%"
                )
                if label == "flip-only":
                    print(f"Pairs:                {int(metrics['pairs'])}")
            elif label in {"mixed", "gold+rag"}:
                print(f"Restored to gold:     {metrics['restored_to_gold_pct']:.1f}%")

    print("\n=== Summary Table ===")
    header = f"{'set':>10} | {'scale':>7} | {'acc_base':>9} | {'acc_steer':>10} | {'kl_with':>8}"
    print(header)
    print("-" * len(header))
    for m in results:
        if m["eval_set"] == "clean":
            acc_base = m["clean_acc_base"]
            acc_steer = m["clean_acc_steer"]
        else:
            acc_base = m["ctx_acc_base"]
            acc_steer = m["ctx_acc_steer"]
        print(
            f"{m['eval_set']:>10} | {m['steer_scale']:7.3f} | "
            f"{acc_base:9.4f} | {acc_steer:10.4f} | {m['kl_with']:8.4f}"
        )

    if owns_model:
        del model
        torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
