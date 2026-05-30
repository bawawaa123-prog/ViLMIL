from __future__ import annotations

import argparse
import ast
import json
import pickle
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

DEFAULT_RESULTS_DIR = Path("results_stage9/rce_mil_v3_prior_calib_vr_a005_5fold_e20_s1")
DEFAULT_OUT_DIR = Path("results_stage9/stage13_rce_evidence_export")
DEFAULT_EXPERIMENT_FILE = "experiment_rce_mil_v3_prior_calib_vr_a005_5fold_e20.txt"
DEFAULT_MODEL_PATH = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASS_NAMES = ["Adenocarcinoma", "NonAdenocarcinoma"]
SLIDE_COLUMNS = [
    "slide_id",
    "fold",
    "split",
    "label",
    "pred",
    "correct",
    "prob_0",
    "prob_1",
    "final_logit_0",
    "final_logit_1",
    "visual_logit_0",
    "visual_logit_1",
    "visual_residual_alpha",
    "low_visual_logit_0",
    "low_visual_logit_1",
    "high_visual_logit_0",
    "high_visual_logit_1",
]
PROMPT_COLUMNS = [
    "slide_id",
    "fold",
    "split",
    "scale",
    "class_id",
    "concept_id",
    "prompt_id",
    "concept_text",
    "evidence_score",
    "prompt_weight",
    "rank",
]


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def build_rce_v3_a005_config() -> SimpleNamespace:
    return SimpleNamespace(
        task="task_adenocarcinoma",
        n_classes=2,
        class_names=list(CLASS_NAMES),
        model_type="RCE_MIL_BiomedCLIP",
        mode="transformer",
        drop_out=False,
        model_size=None,
        prototype_number=16,
        scale_mode="dual",
        use_concept_prompt_pool=True,
        prompt_ensemble_mode="embedding_mean",
        use_dynamic_prompt_gate=False,
        dynamic_gate_hidden_dim=256,
        dynamic_gate_residual_mean=False,
        prompt_dropout=0.0,
        peps_topk=3,
        peps_tau=0.1,
        save_peps_weights=False,
        save_sap_peps_weights=False,
        spatial_lambda=1.0,
        spatial_sigma=1.0,
        spatial_score_type="centroid_mean_dist",
        scale_fusion_mode="sum",
        scale_gate_hidden_dim=128,
        scale_gate_dropout=0.25,
        scale_residual_gamma=0.25,
        allow_legacy_scale_fusion_ckpt=False,
        finetune_text_encoder=False,
        text_finetune_mode="proj",
        text_unfreeze_last_n=2,
        rce_use_concept_prior=True,
        rce_use_logit_calibration=True,
        rce_use_visual_residual=True,
        rce_logit_scale_init=10.0,
        rce_concept_prior_strength=1.0,
        rce_visual_residual_init=0.05,
        seed=1,
        data_root_dir="/xiangmu/data/VILMIL",
        data_folder_s="features_biomedclip_5x",
        data_folder_l="features_biomedclip_20x",
        concept_prompt_path="dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json",
        split_dir="splits/adenocarcinoma/task_adenocarcinoma_strictcv_100",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage13 RCE evidence for RCE-v3-VR-a005.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to ViLa-MIL-main root.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index.")
    parser.add_argument("--split", type=str, default="test", help="Split name: train / val / test.")
    parser.add_argument("--max_slides", type=int, default=10, help="Maximum slides to export; 0 means all.")
    parser.add_argument("--ckpt_path", type=Path, default=None, help="Optional checkpoint path override.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory containing checkpoints and split CSVs.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for exported evidence files.",
    )
    parser.add_argument("--top_k_concepts", type=int, default=5, help="Top concepts per scale/class.")
    parser.add_argument("--top_k_regions", type=int, default=3, help="Top regions retained in pickle summaries.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path | str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def resolve_checkpoint_path(
    root: Path,
    results_dir: Path,
    fold: int,
    ckpt_path: Path | None,
    warning_log: list[str],
) -> Path | None:
    if ckpt_path is not None:
        resolved = resolve_path(root, ckpt_path)
    else:
        resolved = results_dir / f"s_{fold}_checkpoint.pt"
    if not resolved.is_file():
        warn_message(f"Checkpoint not found: {resolved}", warning_log)
        return None
    return resolved


def load_experiment_args(results_dir: Path, warning_log: list[str]) -> dict:
    candidates = sorted(results_dir.glob("experiment_*.txt"))
    if not candidates:
        warn_message(f"No experiment_*.txt found under {results_dir}", warning_log)
        return {}
    try:
        return ast.literal_eval(candidates[0].read_text(encoding="utf-8"))
    except Exception as exc:
        warn_message(f"Failed to parse experiment config {candidates[0]}: {exc}", warning_log)
        return {}


def apply_experiment_overrides(config: SimpleNamespace, root: Path, experiment_args: dict) -> SimpleNamespace:
    for key, value in experiment_args.items():
        setattr(config, key, value)

    config.task = "task_adenocarcinoma"
    config.model_type = "RCE_MIL_BiomedCLIP"
    config.mode = "transformer"
    config.n_classes = 2
    config.class_names = list(CLASS_NAMES)
    config.prototype_number = 16
    config.scale_mode = "dual"
    config.use_concept_prompt_pool = True
    config.prompt_ensemble_mode = "embedding_mean"
    config.rce_use_concept_prior = True
    config.rce_use_logit_calibration = True
    config.rce_use_visual_residual = True
    config.rce_logit_scale_init = 10.0
    config.rce_concept_prior_strength = 1.0
    config.rce_visual_residual_init = 0.05
    config.data_root_dir = str(getattr(config, "data_root_dir", "/xiangmu/data/VILMIL"))
    config.data_folder_s = str(getattr(config, "data_folder_s", "features_biomedclip_5x"))
    config.data_folder_l = str(getattr(config, "data_folder_l", "features_biomedclip_20x"))
    config.concept_prompt_path = str(
        getattr(config, "concept_prompt_path", "dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json")
    )
    config.split_dir = str(
        getattr(config, "split_dir", "splits/adenocarcinoma/task_adenocarcinoma_strictcv_100")
    )

    concept_prompt_path = Path(config.concept_prompt_path)
    if not concept_prompt_path.is_absolute():
        config.concept_prompt_path = str(resolve_path(root, concept_prompt_path))

    split_dir = Path(config.split_dir)
    if not split_dir.is_absolute():
        config.split_dir = str(resolve_path(root, split_dir))

    return config


def build_dataset(config: SimpleNamespace, root: Path, warning_log: list[str]):
    try:
        from datasets.dataset_generic import Generic_MIL_Dataset
    except Exception as exc:
        warn_message(f"Failed to import dataset loader: {exc}", warning_log)
        return None

    data_dir_s = Path(config.data_root_dir) / str(config.data_folder_s)
    data_dir_l = Path(config.data_root_dir) / str(config.data_folder_l)
    csv_path = root / "dataset_csv/all_data.csv"
    if not csv_path.is_file():
        warn_message(f"Dataset CSV not found: {csv_path}", warning_log)
        return None
    if not data_dir_s.is_dir():
        warn_message(f"Low-scale feature directory not found: {data_dir_s}", warning_log)
        return None
    if not data_dir_l.is_dir():
        warn_message(f"High-scale feature directory not found: {data_dir_l}", warning_log)
        return None

    try:
        dataset = Generic_MIL_Dataset(
            csv_path=str(csv_path),
            mode="transformer",
            data_dir_s=str(data_dir_s),
            data_dir_l=str(data_dir_l),
            shuffle=False,
            print_info=False,
            label_dict={name: idx for idx, name in enumerate(CLASS_NAMES)},
            patient_strat=False,
            ignore=[],
        )
    except Exception as exc:
        warn_message(f"Failed to build dataset: {exc}", warning_log)
        return None
    return dataset


def get_split_dataset(
    dataset: Generic_MIL_Dataset,
    results_dir: Path,
    fold: int,
    split_name: str,
    warning_log: list[str],
):
    split_csv = results_dir / f"splits_{fold}.csv"
    if not split_csv.is_file():
        warn_message(f"Split CSV not found: {split_csv}", warning_log)
        return None

    try:
        train_split, val_split, test_split = dataset.return_splits(from_id=False, csv_path=str(split_csv))
    except Exception as exc:
        warn_message(f"Failed to load split CSV {split_csv}: {exc}", warning_log)
        return None

    split_map = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }
    split_dataset = split_map.get(split_name)
    if split_dataset is None:
        warn_message(f"Requested split '{split_name}' is empty or invalid for {split_csv}", warning_log)
    return split_dataset


def load_prompt_metadata(prompt_json_path: str, warning_log: list[str]) -> dict[str, list[list[dict]]]:
    try:
        from utils.prompt_utils import _group_concept_prompt_items
    except Exception as exc:
        warn_message(f"Failed to import prompt utils: {exc}", warning_log)
        return {"low": {0: [], 1: []}, "high": {0: [], 1: []}}

    try:
        _, grouped_items = _group_concept_prompt_items(
            prompt_json_path=prompt_json_path,
            num_classes=2,
            class_names=CLASS_NAMES,
        )
        return grouped_items
    except Exception as exc:
        warn_message(f"Failed to parse concept prompt JSON {prompt_json_path}: {exc}", warning_log)
        return {"low": {0: [], 1: []}, "high": {0: [], 1: []}}


def load_model(
    config: SimpleNamespace,
    ckpt_path: Path,
    warning_log: list[str],
) -> object | None:
    try:
        import ml_collections
        from models.model_RCE_MIL_BiomedCLIP import RCE_MIL_BiomedCLIP

        model_config = ml_collections.ConfigDict()
        model_config.input_size = 512
        model_config.prototype_number = int(config.prototype_number)
        model_config.peps_tau = float(config.peps_tau)
        model_config.scale_mode = str(config.scale_mode)
        model_config.use_concept_prompt_pool = bool(config.use_concept_prompt_pool)
        model_config.concept_prompt_path = str(config.concept_prompt_path)
        model_config.prompt_ensemble_mode = str(config.prompt_ensemble_mode)
        model_config.rce_use_logit_calibration = bool(config.rce_use_logit_calibration)
        model_config.rce_use_concept_prior = bool(config.rce_use_concept_prior)
        model_config.rce_concept_prior_strength = float(config.rce_concept_prior_strength)
        model_config.rce_use_visual_residual = bool(config.rce_use_visual_residual)
        model_config.rce_visual_residual_init = float(config.rce_visual_residual_init)
        model_config.rce_logit_scale_init = float(config.rce_logit_scale_init)
        model_config.class_names = list(CLASS_NAMES)
        model_config.finetune_text_encoder = False
        model = RCE_MIL_BiomedCLIP(config=model_config, num_classes=2, model_path=DEFAULT_MODEL_PATH)
    except Exception as exc:
        warn_message(f"Failed to initialize RCE model: {exc}", warning_log)
        return None

    try:
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        clean_state_dict = {}
        for key, value in state_dict.items():
            if "instance_loss_fn" in key:
                continue
            clean_state_dict[key.replace(".module", "")] = value
        model.load_state_dict(clean_state_dict, strict=True)
    except Exception as exc:
        warn_message(f"Failed to load checkpoint {ckpt_path}: {exc}", warning_log)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def topk_concepts_from_prompt_evidence(
    slide_id: str,
    fold: int,
    split_name: str,
    prompt_evidence: np.ndarray,
    prompt_weights: np.ndarray,
    prompt_metadata: dict[str, list[list[dict]]],
    top_k_concepts: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scale_name in ["low", "high"]:
        evidence_by_class = prompt_evidence[scale_name]
        weights_by_class = prompt_weights[scale_name]
        metadata_by_class = prompt_metadata.get(scale_name, {})
        for class_id in range(evidence_by_class.shape[0]):
            order = np.argsort(-evidence_by_class[class_id])[:top_k_concepts]
            for rank, prompt_idx in enumerate(order, start=1):
                class_metadata = metadata_by_class.get(class_id, []) if isinstance(metadata_by_class, dict) else []
                meta = class_metadata[prompt_idx] if prompt_idx < len(class_metadata) else {}
                records.append(
                    {
                        "slide_id": slide_id,
                        "fold": fold,
                        "split": split_name,
                        "scale": scale_name,
                        "class_id": class_id,
                        "concept_id": meta.get("concept_id", "") or f"class{class_id}_prompt{prompt_idx}",
                        "prompt_id": int(prompt_idx),
                        "concept_text": meta.get("prompt", "") or meta.get("concept_en", ""),
                        "evidence_score": float(evidence_by_class[class_id, prompt_idx]),
                        "prompt_weight": float(weights_by_class[class_id, prompt_idx]),
                        "rank": rank,
                    }
                )
    return records


def extract_model_evidence(
    model,
    loader,
    fold: int,
    split_name: str,
    max_slides: int,
    top_k_concepts: int,
    top_k_regions: int,
    prompt_metadata: dict[str, list[list[dict]]],
    warning_log: list[str],
):
    slide_rows: list[dict[str, object]] = []
    concept_rows: list[dict[str, object]] = []
    evidence_payload: list[dict[str, object]] = []

    device = next(model.parameters()).device
    processed = 0
    with torch.no_grad():
        loader_iter = iter(loader)
        while True:
            try:
                data_s, coord_s, data_l, coords_l, label, batch_slide_ids = next(loader_iter)
            except StopIteration:
                break
            except FileNotFoundError as exc:
                warn_message(f"Missing feature file while iterating loader: {exc}", warning_log)
                break
            except Exception as exc:
                warn_message(f"Failed to iterate split loader: {exc}", warning_log)
                break

            slide_id = batch_slide_ids[0] if isinstance(batch_slide_ids, (list, tuple)) and batch_slide_ids else "unknown"
            try:
                data_s = data_s.to(device)
                coord_s = coord_s.to(device)
                data_l = data_l.to(device)
                coords_l = coords_l.to(device)
                label = label.to(device)
                y_prob, y_hat, _ = model(data_s, coord_s, data_l, coords_l, label, slide_id=slide_id)
            except FileNotFoundError as exc:
                warn_message(f"Missing feature file for slide {slide_id}: {exc}", warning_log)
                continue
            except Exception as exc:
                warn_message(f"Failed to run inference for slide {slide_id}: {exc}", warning_log)
                continue

            prob = y_prob.detach().cpu().numpy()[0]
            pred = int(y_hat.detach().cpu().numpy()[0][0])
            true_label = int(label.detach().cpu().numpy()[0])
            final_logits = model.last_final_logits.numpy()[0] if model.last_final_logits is not None else np.full(2, np.nan)
            visual_logits = model.last_visual_logits.numpy()[0] if model.last_visual_logits is not None else np.full(2, np.nan)
            low_visual_logits = (
                model.last_low_visual_logits.numpy()[0] if model.last_low_visual_logits is not None else np.full(2, np.nan)
            )
            high_visual_logits = (
                model.last_high_visual_logits.numpy()[0] if model.last_high_visual_logits is not None else np.full(2, np.nan)
            )
            alpha = (
                float(model.last_visual_residual_alpha.numpy().reshape(-1)[0])
                if model.last_visual_residual_alpha is not None
                else np.nan
            )

            slide_rows.append(
                {
                    "slide_id": slide_id,
                    "fold": fold,
                    "split": split_name,
                    "label": true_label,
                    "pred": pred,
                    "correct": int(pred == true_label),
                    "prob_0": float(prob[0]),
                    "prob_1": float(prob[1]),
                    "final_logit_0": float(final_logits[0]),
                    "final_logit_1": float(final_logits[1]),
                    "visual_logit_0": float(visual_logits[0]),
                    "visual_logit_1": float(visual_logits[1]),
                    "visual_residual_alpha": alpha,
                    "low_visual_logit_0": float(low_visual_logits[0]),
                    "low_visual_logit_1": float(low_visual_logits[1]),
                    "high_visual_logit_0": float(high_visual_logits[0]),
                    "high_visual_logit_1": float(high_visual_logits[1]),
                }
            )

            low_prompt_evidence = model.last_low_prompt_evidence.numpy()[0] if model.last_low_prompt_evidence is not None else None
            high_prompt_evidence = (
                model.last_high_prompt_evidence.numpy()[0] if model.last_high_prompt_evidence is not None else None
            )
            low_prompt_weights = model.last_low_prompt_weights.numpy()[0] if model.last_low_prompt_weights is not None else None
            high_prompt_weights = model.last_high_prompt_weights.numpy()[0] if model.last_high_prompt_weights is not None else None

            if (
                low_prompt_evidence is not None
                and high_prompt_evidence is not None
                and low_prompt_weights is not None
                and high_prompt_weights is not None
            ):
                concept_rows.extend(
                    topk_concepts_from_prompt_evidence(
                        slide_id=slide_id,
                        fold=fold,
                        split_name=split_name,
                        prompt_evidence={"low": low_prompt_evidence, "high": high_prompt_evidence},
                        prompt_weights={"low": low_prompt_weights, "high": high_prompt_weights},
                        prompt_metadata=prompt_metadata,
                        top_k_concepts=top_k_concepts,
                    )
                )

            low_sim = model.last_low_region_concept_sim.numpy()[0] if model.last_low_region_concept_sim is not None else None
            high_sim = model.last_high_region_concept_sim.numpy()[0] if model.last_high_region_concept_sim is not None else None
            low_region_features = (
                model.last_low_region_features.numpy()[0] if model.last_low_region_features is not None else None
            )
            high_region_features = (
                model.last_high_region_features.numpy()[0] if model.last_high_region_features is not None else None
            )

            payload = {
                "slide_id": slide_id,
                "fold": fold,
                "split": split_name,
                "label": true_label,
                "pred": pred,
                "low_region_concept_sim": low_sim,
                "high_region_concept_sim": high_sim,
                "low_prompt_evidence": low_prompt_evidence,
                "high_prompt_evidence": high_prompt_evidence,
                "low_prompt_weights": low_prompt_weights,
                "high_prompt_weights": high_prompt_weights,
                "low_region_features": low_region_features,
                "high_region_features": high_region_features,
                "final_logits": final_logits,
                "visual_logits": visual_logits,
                "visual_residual_alpha": alpha,
            }
            if low_sim is not None:
                payload["low_top_region_indices"] = np.argsort(-low_sim.max(axis=(0, 2)))[:top_k_regions]
            if high_sim is not None:
                payload["high_top_region_indices"] = np.argsort(-high_sim.max(axis=(0, 2)))[:top_k_regions]
            evidence_payload.append(payload)

            processed += 1
            if max_slides > 0 and processed >= max_slides:
                break

    return slide_rows, concept_rows, evidence_payload


def save_outputs(
    out_dir: Path,
    slide_rows: list[dict[str, object]],
    concept_rows: list[dict[str, object]],
    evidence_payload: list[dict[str, object]],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_csv = out_dir / "slide_prediction_evidence.csv"
    concept_csv = out_dir / "slide_top_concepts.csv"
    evidence_pkl = out_dir / "region_concept_evidence.pkl"

    pd.DataFrame(slide_rows, columns=SLIDE_COLUMNS).to_csv(slide_csv, index=False)
    pd.DataFrame(concept_rows, columns=PROMPT_COLUMNS).to_csv(concept_csv, index=False)
    with evidence_pkl.open("wb") as f:
        pickle.dump(evidence_payload, f)

    return [slide_csv, concept_csv, evidence_pkl]


def build_report(
    fold: int,
    split_name: str,
    max_slides: int,
    ckpt_path: Path | None,
    exported_slide_count: int,
    output_files: list[Path],
    warning_log: list[str],
) -> str:
    lines = [
        "# Stage13 RCE Evidence Export",
        "",
        f"- fold: `{fold}`",
        f"- split: `{split_name}`",
        f"- max_slides: `{max_slides}`",
        f"- checkpoint_path: `{ckpt_path}`" if ckpt_path is not None else "- checkpoint_path: `missing`",
        f"- exported_slides: `{exported_slide_count}`",
        "",
        "## Output Files",
        "",
    ]
    for path in output_files:
        lines.append(f"- `{path}`")

    lines.extend(["", "## Warnings", ""])
    if warning_log:
        for warning in warning_log:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Step14: concept-class graph or evidence visualization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_dir = resolve_path(root, args.results_dir)
    out_dir = resolve_path(root, args.out_dir)
    warning_log: list[str] = []

    config = build_rce_v3_a005_config()
    experiment_args = load_experiment_args(results_dir, warning_log)
    config = apply_experiment_overrides(config, root, experiment_args)

    ckpt_path = resolve_checkpoint_path(root, results_dir, args.fold, args.ckpt_path, warning_log)
    output_files: list[Path] = []

    if ckpt_path is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "stage13_rce_evidence_export_report.md"
        report_path.write_text(
            build_report(
                fold=args.fold,
                split_name=args.split,
                max_slides=args.max_slides,
                ckpt_path=None,
                exported_slide_count=0,
                output_files=[],
                warning_log=warning_log,
            ),
            encoding="utf-8",
        )
        print(f"Checkpoint missing. Wrote report to: {report_path}")
        return

    dataset = build_dataset(config, root, warning_log)
    if dataset is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "stage13_rce_evidence_export_report.md"
        report_path.write_text(
            build_report(
                fold=args.fold,
                split_name=args.split,
                max_slides=args.max_slides,
                ckpt_path=ckpt_path,
                exported_slide_count=0,
                output_files=[],
                warning_log=warning_log,
            ),
            encoding="utf-8",
        )
        print(f"Dataset unavailable. Wrote report to: {report_path}")
        return

    split_dataset = get_split_dataset(dataset, results_dir, args.fold, args.split, warning_log)
    if split_dataset is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "stage13_rce_evidence_export_report.md"
        report_path.write_text(
            build_report(
                fold=args.fold,
                split_name=args.split,
                max_slides=args.max_slides,
                ckpt_path=ckpt_path,
                exported_slide_count=0,
                output_files=[],
                warning_log=warning_log,
            ),
            encoding="utf-8",
        )
        print(f"Split unavailable. Wrote report to: {report_path}")
        return

    prompt_metadata = load_prompt_metadata(config.concept_prompt_path, warning_log)
    model = load_model(config, ckpt_path, warning_log)
    if model is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "stage13_rce_evidence_export_report.md"
        report_path.write_text(
            build_report(
                fold=args.fold,
                split_name=args.split,
                max_slides=args.max_slides,
                ckpt_path=ckpt_path,
                exported_slide_count=0,
                output_files=[],
                warning_log=warning_log,
            ),
            encoding="utf-8",
        )
        print(f"Model unavailable. Wrote report to: {report_path}")
        return

    try:
        from utils.utils import get_simple_loader
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        warn_message(f"Failed to import dataloader helper: {exc}", warning_log)
        report_path = out_dir / "stage13_rce_evidence_export_report.md"
        report_path.write_text(
            build_report(
                fold=args.fold,
                split_name=args.split,
                max_slides=args.max_slides,
                ckpt_path=ckpt_path,
                exported_slide_count=0,
                output_files=[],
                warning_log=warning_log,
            ),
            encoding="utf-8",
        )
        print(f"Dataloader unavailable. Wrote report to: {report_path}")
        return

    loader = get_simple_loader(split_dataset, batch_size=1, num_workers=0, mode="transformer")
    slide_rows, concept_rows, evidence_payload = extract_model_evidence(
        model=model,
        loader=loader,
        fold=args.fold,
        split_name=args.split,
        max_slides=args.max_slides,
        top_k_concepts=args.top_k_concepts,
        top_k_regions=args.top_k_regions,
        prompt_metadata=prompt_metadata,
        warning_log=warning_log,
    )

    output_files = save_outputs(out_dir, slide_rows, concept_rows, evidence_payload)
    report_path = out_dir / "stage13_rce_evidence_export_report.md"
    report_path.write_text(
        build_report(
            fold=args.fold,
            split_name=args.split,
            max_slides=args.max_slides,
            ckpt_path=ckpt_path,
            exported_slide_count=len(slide_rows),
            output_files=output_files + [report_path],
            warning_log=warning_log,
        ),
        encoding="utf-8",
    )

    print(f"Saved slide-level evidence to: {output_files[0]}")
    print(f"Saved top-concept evidence to: {output_files[1]}")
    print(f"Saved region-concept evidence pickle to: {output_files[2]}")
    print(f"Saved export report to: {report_path}")


if __name__ == "__main__":
    main()
