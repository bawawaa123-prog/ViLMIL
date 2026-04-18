#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def read_csv_auto(path: str) -> Tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception as ex:
            last_err = ex
    raise RuntimeError(f"Failed to read CSV with tried encodings: {path}. Last error: {last_err}")


def normalize_text(text: str) -> str:
    text = str(text).replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"[。；;！？!?\n]+", text)
    out = []
    for s in chunks:
        s = s.strip(" -•*：:,.，；;()（）[]【】\"' ")
        if s:
            out.append(s)
    return out


LEAKAGE_PATTERNS = [
    r"浸润性腺癌",
    r"肺腺癌",
    r"腺癌",
    r"非腺癌",
    r"鳞癌",
    r"小细胞癌",
    r"大细胞癌",
    r"转移癌",
    r"adenocarcinoma",
    r"non-adenocarcinoma",
    r"检查结论",
    r"病理分期",
    r"TNM",
    r"\bpT\d+[a-zA-Z]*\b",
    r"\bpN[0-3xX]+\b",
]


def mask_leakage(sentence: str) -> str:
    s = sentence
    for p in LEAKAGE_PATTERNS:
        s = re.sub(p, " ", s, flags=re.IGNORECASE)
    # remove explicit percentages and long numeric fragments
    s = re.sub(r"\d+\s*%", " ", s)
    s = re.sub(r"\b\d{2,}\b", " ", s)
    # remove staging-like shorthand
    s = re.sub(r"\bPL\d+\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSTAS\b", " ", s, flags=re.IGNORECASE)
    # cleanup separators
    s = re.sub(r"^[0-9]+[.)、．]\s*", "", s)
    s = re.sub(r"[()（）\[\]【】]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


LOW_KEYWORDS = [
    "腺体", "结构", "排列", "浸润", "间质", "纤维", "坏死", "乳头", "贴壁", "腺泡",
    "实性", "边界", "架构", "簇", "筛状", "复杂", "desmoplas", "acinar", "papillary",
    "lepidic", "solid", "strom", "architect",
]

HIGH_KEYWORDS = [
    "核", "核异型", "核分裂", "染色质", "核仁", "胞浆", "细胞", "多形", "有丝分裂",
    "核浆比", "异型", "pleomorph", "mitosis", "chromatin", "nucle", "cytolog", "atypia",
]

IMAGING_KEYWORDS = [
    "结节", "实性", "磨玻璃", "毛刺", "空泡", "胸膜", "淋巴结", "CT", "PET", "SUV",
    "影像", "强化", "密度", "结论",
]


def infer_scale_and_tags(sentence: str, source_field: str) -> Tuple[str, List[str]]:
    low_hits = [k for k in LOW_KEYWORDS if k.lower() in sentence.lower()]
    high_hits = [k for k in HIGH_KEYWORDS if k.lower() in sentence.lower()]
    img_hits = [k for k in IMAGING_KEYWORDS if k.lower() in sentence.lower()]

    if source_field == "影像数据":
        # Imaging text is treated as low-scale auxiliary evidence by default.
        scale = "low" if len(high_hits) <= len(low_hits) + 1 else "high"
    else:
        scale = "low" if len(low_hits) >= len(high_hits) else "high"

    tags = sorted(set(low_hits + high_hits + img_hits))
    return scale, tags


def is_useful_sentence(sentence: str) -> bool:
    if len(sentence) < 8:
        return False
    # Skip pure administrative lines
    if re.search(r"^(影像所见|主要病变一览表|检查项目|检查报告)$", sentence):
        return False
    # At least one morphology/imaging clue
    key_hit = any(k.lower() in sentence.lower() for k in LOW_KEYWORDS + HIGH_KEYWORDS + IMAGING_KEYWORDS)
    # Skip numeric-heavy fragments that are mostly measurements/table rows
    digit_ratio = sum(ch.isdigit() for ch in sentence) / max(1, len(sentence))
    if digit_ratio > 0.22:
        return False
    return key_hit


def score_sentence(sentence: str, tags: List[str], source_field: str) -> float:
    length = len(sentence)
    len_score = 1.0
    if length < 20:
        len_score = 0.6
    elif length > 240:
        len_score = 0.7

    tag_score = min(1.0, len(tags) / 5.0)

    src_boost = 0.0
    if source_field == "石蜡报告":
        src_boost = 0.15
    elif source_field == "冰冻报告":
        src_boost = 0.1
    elif source_field == "影像数据":
        src_boost = 0.05

    score = 0.55 * len_score + 0.45 * tag_score + src_boost
    return round(min(1.0, score), 4)


def norm_dedupe_key(text: str) -> str:
    x = text.lower()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[，,。.;；:：()（）\[\]【】'\"-]", "", x)
    return x


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_fold_kb(
    df_hospital: pd.DataFrame,
    split_csv: str,
    fold_id: int,
    out_dir: str,
    include_fields: List[str],
    max_per_class_scale: int,
) -> Dict[str, int]:
    split_df = pd.read_csv(split_csv)
    if "train" not in split_df.columns:
        raise ValueError(f"split file missing train column: {split_csv}")

    train_ids = set(split_df["train"].dropna().astype(str).str.strip())
    sub = df_hospital[df_hospital["slide_id"].astype(str).str.strip().isin(train_ids)].copy()

    facts = []
    dedupe = set()
    class_scale_counter = Counter()
    src_counter = Counter()

    for _, row in sub.iterrows():
        slide_id = str(row.get("slide_id", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        class_name = str(row.get("label", "")).strip()
        if not slide_id or not class_name:
            continue

        for field in include_fields:
            raw = normalize_text(row.get(field, ""))
            if not raw or raw.lower() in {"nan", "none"}:
                continue

            for sent in split_sentences(raw):
                sent = mask_leakage(sent)
                if not is_useful_sentence(sent):
                    continue

                scale, tags = infer_scale_and_tags(sent, field)
                key = (class_name, scale, norm_dedupe_key(sent))
                if key in dedupe:
                    continue

                if max_per_class_scale > 0 and class_scale_counter[(class_name, scale)] >= max_per_class_scale:
                    continue

                dedupe.add(key)
                class_scale_counter[(class_name, scale)] += 1
                src_counter[field] += 1

                rec = {
                    "kb_id": f"f{fold_id}_{len(facts):06d}",
                    "fold": fold_id,
                    "source_slide_id": slide_id,
                    "source_case_id": case_id,
                    "source_field": field,
                    "class_name": class_name,
                    "scale": scale,
                    "text": sent,
                    "tags": tags,
                    "quality_score": score_sentence(sent, tags, field),
                }
                facts.append(rec)

    facts.sort(key=lambda x: x["quality_score"], reverse=True)

    fold_dir = os.path.join(out_dir, f"fold_{fold_id}")
    ensure_dir(fold_dir)

    kb_jsonl = os.path.join(fold_dir, "kb_facts.jsonl")
    with open(kb_jsonl, "w", encoding="utf-8") as f:
        for r in facts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Export a prompt pool compatible with existing p1 retrieval interface
    pool_csv = os.path.join(fold_dir, "prompt_pool.csv")
    with open(pool_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "scale", "prompt_text"])
        for r in facts:
            w.writerow([r["class_name"], r["scale"], r["text"]])

    stats = {
        "fold": fold_id,
        "train_slide_count": len(train_ids),
        "kb_fact_count": len(facts),
        "kb_unique_class_scale": len(class_scale_counter),
        "from_ice": int(src_counter.get("冰冻报告", 0)),
        "from_paraffin": int(src_counter.get("石蜡报告", 0)),
        "from_imaging": int(src_counter.get("影像数据", 0)),
    }

    with open(os.path.join(fold_dir, "kb_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def parse_folds(text: str) -> List[int]:
    out = []
    for t in str(text).split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(description="Build fold-wise structured pathology text KB from hospital CSV")
    parser.add_argument("--hospital_csv", type=str, default="dataset_csv/嘉豪GDPH本中心队列文本数据补充20260414.csv")
    parser.add_argument("--split_dir", type=str, default="splits/Yifuyuan_strict")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--out_dir", type=str, default="knowledge_base/gdph_structured")
    parser.add_argument("--include_imaging", action="store_true", default=False,
                        help="Include 影像数据 field in KB building")
    parser.add_argument("--max_per_class_scale", type=int, default=0,
                        help="Cap items per class+scale (0 means no cap)")
    args = parser.parse_args()

    df_hospital, enc = read_csv_auto(args.hospital_csv)
    need_cols = ["slide_id", "label", "冰冻报告", "石蜡报告", "影像数据"]
    miss = [c for c in need_cols if c not in df_hospital.columns]
    if miss:
        raise ValueError(f"hospital csv missing columns: {miss}")

    include_fields = ["冰冻报告", "石蜡报告"]
    if args.include_imaging:
        include_fields.append("影像数据")

    ensure_dir(args.out_dir)

    folds = parse_folds(args.folds)
    all_stats = []

    print(f"Loaded hospital csv with encoding={enc}, rows={len(df_hospital)}")
    print(f"Building KB for folds={folds}, fields={include_fields}")

    for fold_id in folds:
        split_csv = os.path.join(args.split_dir, f"splits_{fold_id}.csv")
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f"split file not found: {split_csv}")

        stats = build_fold_kb(
            df_hospital=df_hospital,
            split_csv=split_csv,
            fold_id=fold_id,
            out_dir=args.out_dir,
            include_fields=include_fields,
            max_per_class_scale=int(args.max_per_class_scale),
        )
        all_stats.append(stats)
        print(f"fold={fold_id} kb_fact_count={stats['kb_fact_count']} train_slides={stats['train_slide_count']}")

    pd.DataFrame(all_stats).to_csv(os.path.join(args.out_dir, "kb_build_summary.csv"), index=False)
    print(f"Done. Summary saved to {os.path.join(args.out_dir, 'kb_build_summary.csv')}")


if __name__ == "__main__":
    main()
