import numpy as np
import torch
from models.model_mil import MIL_fc, MIL_fc_mc
import pandas as pd
import os
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize


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
        config.prototype_number = args.prototype_number
        config.finetune_text_encoder = bool(getattr(args, 'finetune_text_encoder', False))
        config.enable_dynamic_prompt = bool(getattr(args, 'enable_dynamic_prompt', False))
        config.prompt_pool = getattr(args, 'prompt_pool', None)
        config.class_names = getattr(args, 'class_names', None)
        config.retrieval_topk = int(getattr(args, 'retrieval_topk', 3))
        config.retrieval_temp = float(getattr(args, 'retrieval_temp', 0.1))
        config.dynamic_prompt_mix = float(getattr(args, 'dynamic_prompt_mix', 1.0))
        config.enable_vcp = bool(getattr(args, 'enable_vcp', False))
        config.vcp_beta = float(getattr(args, 'vcp_beta', 0.1))
        config.vcp_dropout = float(getattr(args, 'vcp_dropout', 0.1))
        config.enable_rag_rewrite = bool(getattr(args, 'enable_rag_rewrite', False))
        config.rag_mode = str(getattr(args, 'rag_mode', 'offline'))
        config.rag_cache_path = str(getattr(args, 'rag_cache_path', 'results/rag_rewrite_cache.jsonl'))
        config.rag_topk = int(getattr(args, 'rag_topk', 3))
        config.rag_ollama_model = str(getattr(args, 'rag_ollama_model', 'qwen2.5:14b-instruct'))
        config.rag_ollama_url = str(getattr(args, 'rag_ollama_url', 'http://localhost:11434/api/generate'))
        config.rag_temperature = float(getattr(args, 'rag_temperature', 0.2))
        config.rag_max_tokens = int(getattr(args, 'rag_max_tokens', 256))
        config.rag_timeout_sec = int(getattr(args, 'rag_timeout_sec', 60))
        config.rag_max_retries = int(getattr(args, 'rag_max_retries', 2))
        config.rag_retry_delay_sec = float(getattr(args, 'rag_retry_delay_sec', 0.5))
        config.rag_failure_log_path = getattr(args, 'rag_failure_log_path', None)
        config.rag_fallback = str(getattr(args, 'rag_fallback', 'dynamic'))
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
        model = model.to(torch.device('cuda'))
        # pass
    model.eval()
    return model

def eval(mode, dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, mode=args.mode)
    patient_results, test_error, auc, test_f1, df, acc_logger = summary(mode, model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('f1: ', test_f1)

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)

    return model, patient_results, test_error, auc, test_f1, df, each_class_acc

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
    patient_results = {}
    retrieval_rows = []

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coord_l, label, batch_slide_ids) in enumerate(loader):
            data_s, coord_s, data_l, coord_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coord_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            if isinstance(batch_slide_ids, (list, tuple)) and len(batch_slide_ids) > 0:
                slide_id_for_model = batch_slide_ids[0]
            else:
                slide_id_for_model = slide_id
            with torch.no_grad():
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

            if bool(getattr(args, 'save_retrieval_log', False)) and hasattr(model, 'get_last_retrieval_debug'):
                dbg = model.get_last_retrieval_debug()
                if dbg is not None:
                    for scale_name in ['low', 'high']:
                        for item in dbg.get(scale_name, []):
                            retrieval_rows.append({
                                'slide_id': str(slide_id),
                                'true_label': int(label.item()),
                                'pred_label': int(Y_hat.item()),
                                'scale': scale_name,
                                'class_idx': int(item.get('class_idx', -1)),
                                'top_indices': '|'.join(map(str, item.get('top_indices', []))),
                                'top_scores': '|'.join([f"{float(x):.6f}" for x in item.get('top_scores', [])]),
                                'top_weights': '|'.join([f"{float(x):.6f}" for x in item.get('top_weights', [])]),
                                'top_texts': ' || '.join([str(x) for x in item.get('top_texts', [])]),
                            })

        # 将列表转换为numpy数组并展平
        all_pred_np = np.concatenate(all_pred)
        all_label_np = np.concatenate(all_label)
        test_f1 = f1_score(all_label_np, all_pred_np, average='macro')
        test_error /= len(loader)

        aucs = []
        if len(np.unique(all_labels)) == 1:
            auc_score = -1

        else:
            if args.n_classes == 2:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                for class_idx in range(args.n_classes):
                    if class_idx in all_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                        aucs.append(auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                if args.micro_average:
                    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                    auc_score = auc(fpr, tpr)
                else:
                    auc_score = np.nanmean(np.array(aucs))

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)

        if bool(getattr(args, 'save_retrieval_log', False)) and len(retrieval_rows) > 0:
            retrieval_df = pd.DataFrame(retrieval_rows)
            log_name = str(getattr(args, 'retrieval_log_name', 'retrieval_log.csv'))
            save_dir = getattr(args, 'save_dir', None)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                log_path = os.path.join(save_dir, log_name)
                retrieval_df.to_csv(log_path, index=False)
                print(f"[DynamicPrompt] retrieval logs saved: {log_path}")

        return patient_results, test_error, auc_score, test_f1, df, acc_logger 
