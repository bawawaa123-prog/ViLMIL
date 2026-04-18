#!/usr/bin/env python3
"""Fill missing RAG cache entries using the exact model forward path (no backprop).

This script aligns with online training behavior by:
1) building the same ViLa-MIL-BiomedCLIP model components,
2) running per-slide forward to obtain real retrieval_debug from image features,
3) letting the in-model RAG rewriter generate and append cache entries.
"""

import argparse
import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict cache fill via real model forward (no backward), compatible with online main.py cache format."
    )

    parser.add_argument("--task", type=str, default="task_adenocarcinoma",
                        choices=["task_adenocarcinoma", "task_tcga_lung_subtyping", "task_tcga_rcc_subtyping"])
    parser.add_argument("--mode", type=str, default="transformer", choices=["transformer"])

    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--data_folder_s", type=str, required=True)
    parser.add_argument("--data_folder_l", type=str, required=True)

    parser.add_argument("--split_csv", type=str, default=None,
                        help="Direct split CSV path, e.g. splits/Yifuyuan_strict/splits_0.csv")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Split directory, used with --fold to resolve splits_{fold}.csv")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index for split_dir mode")
    parser.add_argument("--which_splits", type=str, default="all", choices=["all", "train", "val", "test"],
                        help="Which subset(s) to process")

    parser.add_argument("--text_prompt_path", type=str, required=True,
                        help="CSV containing low/high prompt columns")
    parser.add_argument("--prototype_number", type=int, default=16)

    parser.add_argument("--enable_dynamic_prompt", action="store_true", default=True,
                        help="Keep enabled for strict alignment with online chain")
    parser.add_argument("--prompt_pool_path", type=str, required=True)
    parser.add_argument("--retrieval_topk", type=int, default=3)
    parser.add_argument("--retrieval_temp", type=float, default=0.15)
    parser.add_argument("--dynamic_prompt_mix", type=float, default=0.9)

    parser.add_argument("--enable_vcp", action="store_true", default=False,
                        help="Optional: keep OFF by default during cache construction")
    parser.add_argument("--vcp_beta", type=float, default=0.1)
    parser.add_argument("--vcp_dropout", type=float, default=0.1)

    parser.add_argument("--enable_rag_rewrite", action="store_true", default=True)
    parser.add_argument("--rag_mode", type=str, choices=["online", "hybrid"], default="hybrid")
    parser.add_argument("--rag_cache_path", type=str, required=True)
    parser.add_argument("--rag_topk", type=int, default=3)
    parser.add_argument("--rag_ollama_model", type=str, default="qwen2.5:14b-instruct")
    parser.add_argument("--rag_ollama_url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument("--rag_temperature", type=float, default=0.0)
    parser.add_argument("--rag_max_tokens", type=int, default=256)
    parser.add_argument("--rag_timeout_sec", type=int, default=60)
    parser.add_argument("--rag_max_retries", type=int, default=2)
    parser.add_argument("--rag_retry_delay_sec", type=float, default=0.5)
    parser.add_argument("--rag_failure_log_path", type=str, default=None)
    parser.add_argument("--rag_fallback", type=str, choices=["dynamic", "static"], default="dynamic")

    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional model checkpoint for closer alignment with a specific training state")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N missing slides (0 means all)")

    return parser.parse_args()


def resolve_split_csv(args):
    if args.split_csv and os.path.isfile(args.split_csv):
        return args.split_csv

    if args.split_dir:
        split_dir = args.split_dir
        if not os.path.isdir(split_dir):
            candidate = os.path.join("splits", split_dir)
            if os.path.isdir(candidate):
                split_dir = candidate
        split_csv = os.path.join(split_dir, f"splits_{args.fold}.csv")
        if os.path.isfile(split_csv):
            return split_csv

    raise FileNotFoundError("Unable to resolve split CSV. Provide --split_csv or valid --split_dir/--fold")


def resolve_text_prompt_path(path):
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join("text_prompt", path),
        os.path.join(REPO_ROOT, "text_prompt", path),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"text_prompt_path not found: {path}")


def load_text_prompt_list(path):
    path = resolve_text_prompt_path(path)
    try:
        df = pd.read_csv(path)
        cols = [c.strip().lower() for c in df.columns]
        if "low_resolution_description" in cols and "high_resolution_description" in cols:
            low_idx = cols.index("low_resolution_description")
            high_idx = cols.index("high_resolution_description")
            low_prompts = df.iloc[:, low_idx].astype(str).fillna("").tolist()
            high_prompts = df.iloc[:, high_idx].astype(str).fillna("").tolist()
            return [str(x) for x in low_prompts] + [str(x) for x in high_prompts]
    except Exception:
        pass

    arr = pd.read_csv(path, header=None).values
    return [str(x) for x in arr.reshape(-1).tolist()]


def load_prompt_pool_csv(prompt_pool_path, class_names):
    resolved_path = prompt_pool_path
    if not os.path.isfile(resolved_path):
        candidate_paths = [
            os.path.join("text_prompt", prompt_pool_path),
            os.path.join(REPO_ROOT, "text_prompt", prompt_pool_path),
        ]
        for c in candidate_paths:
            if os.path.isfile(c):
                resolved_path = c
                break

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            f"Prompt pool CSV not found: {prompt_pool_path}. Try absolute path or text_prompt/<file>.csv"
        )

    df = pd.read_csv(resolved_path)
    norm_cols = {c.strip().lower(): c for c in df.columns}
    class_col = norm_cols.get("class_name") or norm_cols.get("class") or norm_cols.get("label")
    scale_col = norm_cols.get("scale") or norm_cols.get("resolution")
    text_col = norm_cols.get("prompt_text") or norm_cols.get("prompt") or norm_cols.get("sentence") or norm_cols.get("text")

    if class_col is None or scale_col is None or text_col is None:
        raise ValueError(
            "prompt pool CSV must contain class_name/class/label, scale/resolution, and prompt_text/prompt/sentence/text"
        )

    class_to_idx = {name.lower(): i for i, name in enumerate(class_names)}
    prompt_pool = {
        "low": [[] for _ in class_names],
        "high": [[] for _ in class_names],
    }

    for _, row in df.iterrows():
        c_name = str(row[class_col]).strip().lower()
        scale = str(row[scale_col]).strip().lower()
        text = str(row[text_col]).strip()
        if c_name not in class_to_idx or not text:
            continue

        c_idx = class_to_idx[c_name]
        if scale in ["low", "lowres", "low_resolution", "5x"]:
            prompt_pool["low"][c_idx].append(text)
        elif scale in ["high", "highres", "high_resolution", "20x"]:
            prompt_pool["high"][c_idx].append(text)

    return prompt_pool


def build_task_meta(task):
    if task == "task_adenocarcinoma":
        return {
            "n_classes": 2,
            "class_names": ["Adenocarcinoma", "NonAdenocarcinoma"],
            "csv_path": "dataset_csv/all_data.csv",
        }
    if task == "task_tcga_lung_subtyping":
        return {
            "n_classes": 2,
            "class_names": ["LUAD", "LUSC"],
            "csv_path": "dataset_csv/TCGA_Lung_subtyping.csv",
        }
    if task == "task_tcga_rcc_subtyping":
        return {
            "n_classes": 3,
            "class_names": ["CCRCC", "PRCC", "CRCC"],
            "csv_path": "dataset_csv/TCGA_RCC_subtyping.csv",
        }
    raise NotImplementedError(task)


def build_dataset(args, meta):
    from datasets.dataset_generic import Generic_MIL_Dataset

    return Generic_MIL_Dataset(
        csv_path=meta["csv_path"],
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(meta["class_names"])},
        patient_strat=False,
        ignore=[],
    )


def build_model(args, meta, text_prompt, prompt_pool, device):
    import ml_collections
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
    import torch

    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.text_prompt = text_prompt
    config.prototype_number = int(args.prototype_number)

    config.finetune_text_encoder = False

    config.enable_dynamic_prompt = bool(args.enable_dynamic_prompt)
    config.prompt_pool = prompt_pool
    config.class_names = meta["class_names"]
    config.retrieval_topk = int(args.retrieval_topk)
    config.retrieval_temp = float(args.retrieval_temp)
    config.dynamic_prompt_mix = float(args.dynamic_prompt_mix)

    config.enable_vcp = bool(args.enable_vcp)
    config.vcp_beta = float(args.vcp_beta)
    config.vcp_dropout = float(args.vcp_dropout)

    config.enable_rag_rewrite = bool(args.enable_rag_rewrite)
    config.rag_mode = str(args.rag_mode)
    config.rag_cache_path = str(args.rag_cache_path)
    config.rag_topk = int(args.rag_topk)
    config.rag_ollama_model = str(args.rag_ollama_model)
    config.rag_ollama_url = str(args.rag_ollama_url)
    config.rag_temperature = float(args.rag_temperature)
    config.rag_max_tokens = int(args.rag_max_tokens)
    config.rag_timeout_sec = int(args.rag_timeout_sec)
    config.rag_max_retries = int(args.rag_max_retries)
    config.rag_retry_delay_sec = float(args.rag_retry_delay_sec)
    config.rag_failure_log_path = args.rag_failure_log_path
    config.rag_fallback = str(args.rag_fallback)

    model = ViLa_MIL_BiomedCLIP(config=config, num_classes=meta["n_classes"])

    if args.checkpoint_path:
        try:
            ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        ckpt_clean = {k.replace(".module", ""): v for k, v in ckpt.items() if "instance_loss_fn" not in k}
        model.load_state_dict(ckpt_clean, strict=False)
        print(f"[Info] checkpoint loaded: {args.checkpoint_path}")

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)

    model.eval()
    model.set_dynamic_prompt_runtime(True)
    model.set_rag_runtime(True)
    model.set_vcp_runtime(bool(args.enable_vcp))

    return model


def iter_loader(loader, model, device, seen_cache, limit=0):
    import torch

    total = len(loader)
    success = 0
    failed = 0
    skipped = 0
    processed = 0

    for idx, batch in enumerate(loader, start=1):
        data_s, coord_s, data_l, coords_l, label, slide_ids = batch
        slide_id = slide_ids[0] if isinstance(slide_ids, (list, tuple)) and len(slide_ids) > 0 else None
        if slide_id is None:
            failed += 1
            continue

        sid = str(slide_id)
        if sid in seen_cache:
            skipped += 1
            continue

        if limit > 0 and processed >= limit:
            break

        processed += 1

        data_s = data_s.to(device)
        coord_s = coord_s.to(device)
        data_l = data_l.to(device)
        coords_l = coords_l.to(device)
        label = label.to(device)

        with torch.no_grad():
            _ = model(data_s, coord_s, data_l, coords_l, label, slide_id=sid)

        rag_debug = model.get_last_rag_debug()
        if sid in model.rag_rewriter.cache:
            seen_cache.add(sid)
            success += 1
        else:
            failed += 1
            if rag_debug is not None:
                reason = rag_debug.get("reason", "unknown")
                print(f"[WARN] cache miss after forward: slide={sid}, reason={reason}")

        if idx % 20 == 0 or idx == total:
            print(
                f"    progress {idx}/{total} | processed_missing={processed} | "
                f"success={success} failed={failed} skipped_existing={skipped}"
            )

    return {
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "processed": processed,
    }


def main():
    args = parse_args()

    import torch
    from utils.utils import get_split_loader

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_csv = resolve_split_csv(args)
    meta = build_task_meta(args.task)

    text_prompt = load_text_prompt_list(args.text_prompt_path)
    expected = meta["n_classes"] * 2
    if len(text_prompt) != expected:
        print(f"[WARN] text_prompt count={len(text_prompt)}; expected={expected}")

    prompt_pool = load_prompt_pool_csv(args.prompt_pool_path, meta["class_names"])
    dataset = build_dataset(args, meta)

    train_split, val_split, test_split = dataset.return_splits(from_id=False, csv_path=split_csv)
    split_map = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    selected = [args.which_splits] if args.which_splits in split_map else ["train", "val", "test"]

    model = build_model(args, meta, text_prompt, prompt_pool, device)
    if model.rag_rewriter is None:
        raise RuntimeError("RAG rewriter not initialized. Ensure --enable_rag_rewrite is enabled.")

    seen_cache = set(model.rag_rewriter.cache.keys())

    print("=== Strict Forward Cache Fill ===")
    print(f"split_csv: {split_csv}")
    print(f"selected_splits: {selected}")
    print(f"initial cache size: {len(seen_cache)}")
    print(f"failure log: {model.rag_rewriter.failure_log_path}")

    grand = {"success": 0, "failed": 0, "skipped": 0, "processed": 0}
    remaining_limit = int(args.limit)

    for name in selected:
        ds = split_map.get(name)
        if ds is None or len(ds) == 0:
            print(f"[Info] split {name}: empty, skip")
            continue

        loader = get_split_loader(ds, training=False, testing=False, weighted=False, mode=args.mode)

        limit_for_this = 0
        if remaining_limit > 0:
            limit_for_this = remaining_limit

        print(f"[Run] split={name}, size={len(ds)}, limit={limit_for_this if limit_for_this > 0 else 'all'}")
        stats = iter_loader(loader, model, device, seen_cache, limit=limit_for_this)

        for k in grand:
            grand[k] += stats[k]

        if remaining_limit > 0:
            remaining_limit = max(0, remaining_limit - stats["processed"])
            if remaining_limit == 0:
                print("[Info] global --limit reached, stop further splits")
                break

    final_cache_size = len(model.rag_rewriter.cache)
    print("=== Done ===")
    print(
        f"processed_missing={grand['processed']}, success={grand['success']}, failed={grand['failed']}, "
        f"skipped_existing={grand['skipped']}"
    )
    print(f"cache size: {final_cache_size}")


if __name__ == "__main__":
    main()
