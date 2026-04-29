import numpy as np
import torch
from models.model_mil import MIL_fc, MIL_fc_mc
import pandas as pd
import os
import json
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from utils.metric_utils import compute_classification_metrics


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'ViLa_MIL_BiomedCLIP':
        import ml_collections
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.class_names = getattr(args, 'class_names', None)
        config.use_concept_prompt_pool = bool(getattr(args, 'use_concept_prompt_pool', False))
        config.concept_prompt_path = getattr(args, 'concept_prompt_path', None)
        config.prompt_ensemble_mode = str(getattr(args, 'prompt_ensemble_mode', 'embedding_mean'))
        config.use_dynamic_prompt_gate = bool(getattr(args, 'use_dynamic_prompt_gate', False))
        config.dynamic_gate_hidden_dim = int(getattr(args, 'dynamic_gate_hidden_dim', 256))
        config.dynamic_gate_residual_mean = bool(getattr(args, 'dynamic_gate_residual_mean', False))
        config.prompt_dropout = float(getattr(args, 'prompt_dropout', 0.0))
        config.peps_topk = int(getattr(args, 'peps_topk', 3))
        config.peps_tau = float(getattr(args, 'peps_tau', 0.1))
        config.save_peps_weights = bool(getattr(args, 'save_peps_weights', False))
        config.save_sap_peps_weights = bool(getattr(args, 'save_sap_peps_weights', False))
        config.spatial_lambda = float(getattr(args, 'spatial_lambda', 1.0))
        config.spatial_sigma = float(getattr(args, 'spatial_sigma', 1.0))
        config.spatial_score_type = str(getattr(args, 'spatial_score_type', 'centroid_mean_dist'))
        config.scale_mode = str(getattr(args, 'scale_mode', 'dual'))
        config.prototype_number = args.prototype_number
        config.finetune_text_encoder = bool(getattr(args, 'finetune_text_encoder', False))
        config.text_finetune_mode = str(getattr(args, 'text_finetune_mode', 'proj'))
        config.text_unfreeze_last_n = int(getattr(args, 'text_unfreeze_last_n', 2))
        model = ViLa_MIL_BiomedCLIP(config=config, num_classes=args.n_classes)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    try:
        # Prefer weights_only for security (PyTorch 2.2+ supports this kwarg)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        print('[Info] torch.load using weights_only=True')
    except TypeError:
        # Older PyTorch versions don't support weights_only
        ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
    model.eval()
    return model


def _extract_topk_prompt_strings(prompt_texts, prompt_weights, class_idx, topk=3):
    prompts = prompt_texts[class_idx]
    weights = np.asarray(prompt_weights[class_idx], dtype=float)
    order = np.argsort(-weights)[:topk]
    top_prompts = [prompts[idx] for idx in order]
    top_weights = [float(weights[idx]) for idx in order]
    return top_prompts, top_weights


def _build_prompt_export_record(
    slide_row,
    label,
    y_hat,
    y_prob,
    diagnostics,
    class_names,
    prompt_texts_low,
    prompt_texts_high,
):
    pred_class = int(y_hat.item())
    true_class = int(label.item())
    num_classes = len(class_names)
    opp_class = 1 - pred_class if num_classes == 2 else int(np.argsort(y_prob[0])[-2])

    prompt_weights_low = diagnostics["prompt_weights_low"].detach().cpu().numpy()[0]
    prompt_weights_high = diagnostics["prompt_weights_high"].detach().cpu().numpy()[0]
    logits_low = diagnostics["logits_low"].detach().cpu().numpy()[0]
    logits_high = diagnostics["logits_high"].detach().cpu().numpy()[0]
    final_logits = diagnostics["final_logits"].detach().cpu().numpy()[0]
    pred_probs = diagnostics["pred_probs"].detach().cpu().numpy()[0]

    pred_low_prompts, pred_low_weights = _extract_topk_prompt_strings(prompt_texts_low, prompt_weights_low, pred_class)
    pred_high_prompts, pred_high_weights = _extract_topk_prompt_strings(prompt_texts_high, prompt_weights_high, pred_class)
    opp_low_prompts, opp_low_weights = _extract_topk_prompt_strings(prompt_texts_low, prompt_weights_low, opp_class)
    opp_high_prompts, opp_high_weights = _extract_topk_prompt_strings(prompt_texts_high, prompt_weights_high, opp_class)

    record = {
        "case_id": slide_row["case_id"] if "case_id" in slide_row.index else slide_row["slide_id"],
        "slide_id": slide_row["slide_id"],
        "true_label": true_class,
        "pred_label": pred_class,
        "pred_class_name": class_names[pred_class],
        "opp_class_name": class_names[opp_class],
        "pred_prob": float(pred_probs[pred_class]),
        "top3_pred_low_prompts": json.dumps(pred_low_prompts, ensure_ascii=False),
        "top3_pred_low_weights": json.dumps(pred_low_weights),
        "top3_pred_high_prompts": json.dumps(pred_high_prompts, ensure_ascii=False),
        "top3_pred_high_weights": json.dumps(pred_high_weights),
        "top3_opp_low_prompts": json.dumps(opp_low_prompts, ensure_ascii=False),
        "top3_opp_low_weights": json.dumps(opp_low_weights),
        "top3_opp_high_prompts": json.dumps(opp_high_prompts, ensure_ascii=False),
        "top3_opp_high_weights": json.dumps(opp_high_weights),
    }

    for class_idx in range(num_classes):
        record[f"logits_low_class{class_idx}"] = float(logits_low[class_idx])
        record[f"logits_high_class{class_idx}"] = float(logits_high[class_idx])
        record[f"final_logit_class{class_idx}"] = float(final_logits[class_idx])

    return record


def _build_peps_export_records(
    slide_row,
    label,
    y_hat,
    diagnostics,
    class_names,
    prompt_metadata_low,
    prompt_metadata_high,
):
    true_class = int(label.item())
    pred_class = int(y_hat.item())
    pred_probs = diagnostics["pred_probs"].detach().cpu().numpy()[0]
    mode = str(diagnostics.get("mode", "peps"))

    scale_configs = [
        (
            "low",
            diagnostics["prompt_weights_low"].detach().cpu().numpy()[0],
            diagnostics["prompt_semantic_evidence_low"].detach().cpu().numpy()[0],
            diagnostics["prompt_spatial_score_low"].detach().cpu().numpy()[0],
            diagnostics["prompt_final_evidence_low"].detach().cpu().numpy()[0],
            diagnostics["supporting_prototype_index_low"].detach().cpu().numpy()[0],
            diagnostics["topk_prototype_indices_low"].detach().cpu().numpy()[0],
            diagnostics["topk_proto_mean_dist_low"].detach().cpu().numpy()[0],
            prompt_metadata_low,
        ),
        (
            "high",
            diagnostics["prompt_weights_high"].detach().cpu().numpy()[0],
            diagnostics["prompt_semantic_evidence_high"].detach().cpu().numpy()[0],
            diagnostics["prompt_spatial_score_high"].detach().cpu().numpy()[0],
            diagnostics["prompt_final_evidence_high"].detach().cpu().numpy()[0],
            diagnostics["supporting_prototype_index_high"].detach().cpu().numpy()[0],
            diagnostics["topk_prototype_indices_high"].detach().cpu().numpy()[0],
            diagnostics["topk_proto_mean_dist_high"].detach().cpu().numpy()[0],
            prompt_metadata_high,
        ),
    ]

    records = []
    for (
        scale_name,
        weights_by_class,
        semantic_by_class,
        spatial_by_class,
        final_by_class,
        support_by_class,
        topk_proto_indices_by_class,
        topk_mean_dist_by_class,
        metadata_by_class,
    ) in scale_configs:
        for class_idx, class_name in enumerate(class_names):
            order = np.argsort(-weights_by_class[class_idx])[:3]
            record = {
                "case_id": slide_row["case_id"] if "case_id" in slide_row.index else slide_row["slide_id"],
                "slide_id": slide_row["slide_id"],
                "true_label": true_class,
                "pred_label": pred_class,
                "pred_prob": float(pred_probs[pred_class]),
                "scale": scale_name,
                "class_id": class_idx,
                "class_name": class_name,
                "prompt_selection_mode": mode,
                "semantic_evidence_mean": float(np.mean(semantic_by_class[class_idx])),
                "semantic_evidence_std": float(np.std(semantic_by_class[class_idx])),
                "spatial_score_mean": float(np.mean(spatial_by_class[class_idx])),
                "spatial_score_std": float(np.std(spatial_by_class[class_idx])),
                "final_evidence_mean": float(np.mean(final_by_class[class_idx])),
                "final_evidence_std": float(np.std(final_by_class[class_idx])),
                "topk_proto_mean_dist_mean": float(np.mean(topk_mean_dist_by_class[class_idx])),
                "topk_proto_mean_dist_std": float(np.std(topk_mean_dist_by_class[class_idx])),
            }
            for rank_idx, prompt_idx in enumerate(order, start=1):
                meta = metadata_by_class[class_idx][prompt_idx]
                concept_name = meta.get("concept_en") or meta.get("concept_id") or ""
                record[f"top{rank_idx}_prompt_text"] = meta.get("prompt", "")
                record[f"top{rank_idx}_prompt_concept"] = concept_name
                record[f"top{rank_idx}_prompt_weight"] = float(weights_by_class[class_idx][prompt_idx])
                record[f"top{rank_idx}_prompt_semantic_evidence"] = float(semantic_by_class[class_idx][prompt_idx])
                record[f"top{rank_idx}_prompt_spatial_score"] = float(spatial_by_class[class_idx][prompt_idx])
                record[f"top{rank_idx}_prompt_final_evidence"] = float(final_by_class[class_idx][prompt_idx])
                record[f"top{rank_idx}_prompt_topk_proto_mean_dist"] = float(topk_mean_dist_by_class[class_idx][prompt_idx])
                record[f"top{rank_idx}_supporting_prototype_index"] = int(support_by_class[class_idx][prompt_idx])
                record[f"top{rank_idx}_supporting_prototype_indices"] = json.dumps(
                    [int(x) for x in topk_proto_indices_by_class[class_idx][prompt_idx].tolist()]
                )
            records.append(record)
    return records


def eval(mode, dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, mode=args.mode)
    patient_results, metrics, df, acc_logger, prompt_export_df = summary(mode, model, loader, args)
    print('test_error: ', 1 - metrics["acc"])
    print('auc: ', metrics["auc"])
    print('f1: ', metrics["f1"])
    print('balanced_acc: ', metrics["balanced_acc"])
    print('sensitivity: ', metrics["sensitivity"])
    print('specificity: ', metrics["specificity"])
    print('pr_auc: ', metrics["pr_auc"])

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)

    return model, patient_results, metrics, df, each_class_acc, prompt_export_df

def summary(mode, model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    all_pred = []
    all_label = []

    slide_ids = loader.dataset.slide_data['slide_id']
    slide_rows = loader.dataset.slide_data.reset_index(drop=True)
    patient_results = {}
    prompt_export_records = []
    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coord_l, label, batch_slide_ids) in enumerate(loader):
            data_s, coord_s, data_l, coord_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coord_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            if isinstance(batch_slide_ids, (list, tuple)) and len(batch_slide_ids) > 0:
                slide_id_for_model = batch_slide_ids[0]
            else:
                slide_id_for_model = slide_id
            with torch.no_grad():
                diagnostics = None
                prompt_mode = str(getattr(args, "prompt_ensemble_mode", ""))
                need_prompt_diagnostics = bool(getattr(args, "use_dynamic_prompt_gate", False)) or (
                    prompt_mode == "peps" and bool(getattr(args, "save_peps_weights", False))
                ) or (
                    prompt_mode == "sap_peps"
                    and (
                        bool(getattr(args, "save_sap_peps_weights", False))
                        or bool(getattr(args, "save_peps_weights", False))
                    )
                )
                if hasattr(model, "forward_with_prompt_diagnostics") and need_prompt_diagnostics:
                    Y_prob, Y_hat, loss, diagnostics = model.forward_with_prompt_diagnostics(
                        data_s, coord_s, data_l, coord_l, label, slide_id=slide_id_for_model
                    )
                else:
                    Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coord_l, label, slide_id=slide_id_for_model)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()
            all_pred.append(Y_hat.cpu().numpy())
            all_label.append(label.cpu().numpy())
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error
            if diagnostics is not None:
                if str(getattr(args, "prompt_ensemble_mode", "")) in {"peps", "sap_peps"}:
                    prompt_export_records.extend(
                        _build_peps_export_records(
                            slide_row=slide_rows.iloc[batch_idx],
                            label=label,
                            y_hat=Y_hat,
                            diagnostics=diagnostics,
                            class_names=getattr(args, "class_names", [str(i) for i in range(args.n_classes)]),
                            prompt_metadata_low=getattr(model, "concept_prompt_metadata_low", None),
                            prompt_metadata_high=getattr(model, "concept_prompt_metadata_high", None),
                        )
                    )
                else:
                    prompt_export_records.append(
                        _build_prompt_export_record(
                            slide_row=slide_rows.iloc[batch_idx],
                            label=label,
                            y_hat=Y_hat,
                            y_prob=probs,
                            diagnostics=diagnostics,
                            class_names=getattr(args, "class_names", [str(i) for i in range(args.n_classes)]),
                            prompt_texts_low=getattr(model, "concept_prompt_texts_low", None),
                            prompt_texts_high=getattr(model, "concept_prompt_texts_high", None),
                        )
                    )

        # 将列表转换为numpy数组并展平
        all_pred_np = np.concatenate(all_pred)
        all_label_np = np.concatenate(all_label)
        test_error /= len(loader)
        metrics = compute_classification_metrics(all_labels, all_probs, all_pred_np, args.n_classes)
        metrics["error"] = 1.0 - metrics["acc"]

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)
        prompt_export_df = pd.DataFrame(prompt_export_records) if prompt_export_records else None

        return patient_results, metrics, df, acc_logger, prompt_export_df 
