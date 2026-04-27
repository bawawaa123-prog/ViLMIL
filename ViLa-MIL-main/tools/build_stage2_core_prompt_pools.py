import copy
import json
from pathlib import Path


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
INPUT_PATH = ROOT / "dataset_csv" / "private_lung_concept_prompt_pool_stage2_core10.json"
OUTPUT_12 = ROOT / "dataset_csv" / "private_lung_concept_prompt_pool_stage2_core12.json"
OUTPUT_14 = ROOT / "dataset_csv" / "private_lung_concept_prompt_pool_stage2_core14.json"


ADDITIONS_12 = {
    ("Adenocarcinoma", 0, "low"): [
        {
            "concept_id": "adeno_cribriform_low",
            "concept_en": "cribriform glandular pattern",
            "prompt": "A low magnification histopathology image showing lung adenocarcinoma with cribriform glandular architecture.",
        },
        {
            "concept_id": "adeno_alveolar_replacement_low",
            "concept_en": "alveolar replacement by atypical glands",
            "prompt": "A low magnification histopathology image showing atypical adenocarcinoma glands replacing pre-existing alveolar structures.",
        },
    ],
    ("Adenocarcinoma", 0, "high"): [
        {
            "concept_id": "adeno_fused_glands_high",
            "concept_en": "fused back-to-back glands",
            "prompt": "A high magnification histopathology image showing fused back-to-back adenocarcinoma glands with little intervening stroma.",
        },
        {
            "concept_id": "adeno_prominent_nucleoli_high",
            "concept_en": "prominent nucleoli in gland-forming tumor cells",
            "prompt": "A high magnification histopathology image showing gland-forming adenocarcinoma cells with vesicular chromatin and prominent nucleoli.",
        },
    ],
    ("NonAdenocarcinoma", 1, "low"): [
        {
            "concept_id": "nonadeno_solid_nests_low",
            "concept_en": "solid non-gland-forming tumor nests",
            "prompt": "A low magnification histopathology image showing solid non-gland-forming tumor nests without adenocarcinoma-type acinar structures.",
        },
        {
            "concept_id": "nonadeno_organoid_neuroendocrine_low",
            "concept_en": "organoid neuroendocrine nesting pattern",
            "prompt": "A low magnification histopathology image showing organoid nests and trabeculae consistent with neuroendocrine morphology.",
        },
    ],
    ("NonAdenocarcinoma", 1, "high"): [
        {
            "concept_id": "nonadeno_spindle_atypia_high",
            "concept_en": "spindle cell atypia without gland formation",
            "prompt": "A high magnification histopathology image showing atypical spindle tumor cells without gland formation or mucin production.",
        },
        {
            "concept_id": "nonadeno_chondroid_matrix_high",
            "concept_en": "chondroid or myxochondroid matrix",
            "prompt": "A high magnification histopathology image showing chondroid or myxochondroid matrix characteristic of hamartomatous morphology.",
        },
    ],
}


ADDITIONS_14 = {
    ("Adenocarcinoma", 0, "low"): [
        {
            "concept_id": "adeno_angulated_invasive_glands_low",
            "concept_en": "angulated invasive glands in fibrotic stroma",
            "prompt": "A low magnification histopathology image showing angulated invasive adenocarcinoma glands infiltrating fibrotic stroma.",
        },
        {
            "concept_id": "adeno_complex_glandular_mixture_low",
            "concept_en": "complex mixed glandular architecture",
            "prompt": "A low magnification histopathology image showing complex mixed glandular, papillary, and focal solid adenocarcinoma architecture.",
        },
    ],
    ("Adenocarcinoma", 0, "high"): [
        {
            "concept_id": "adeno_mitotic_glandular_atypia_high",
            "concept_en": "mitotically active glandular atypia",
            "prompt": "A high magnification histopathology image showing mitotically active atypical adenocarcinoma cells lining glandular spaces.",
        },
        {
            "concept_id": "adeno_apical_snouts_high",
            "concept_en": "apical snouts and luminal secretions",
            "prompt": "A high magnification histopathology image showing adenocarcinoma cells with apical snouts and luminal secretions within neoplastic glands.",
        },
    ],
    ("NonAdenocarcinoma", 1, "low"): [
        {
            "concept_id": "nonadeno_chondromyxoid_hamartoma_low",
            "concept_en": "chondromyxoid hamartomatous lesion",
            "prompt": "A low magnification histopathology image showing a chondromyxoid hamartomatous lesion with cleft-like entrapped epithelium.",
        },
        {
            "concept_id": "nonadeno_fibrosing_granulomatous_low",
            "concept_en": "fibrosing granulomatous or reactive lesion",
            "prompt": "A low magnification histopathology image showing fibrosing granulomatous or reactive lung lesion without malignant glandular morphology.",
        },
    ],
    ("NonAdenocarcinoma", 1, "high"): [
        {
            "concept_id": "nonadeno_keratin_pearl_whorls_high",
            "concept_en": "concentric keratin pearl whorls",
            "prompt": "A high magnification histopathology image showing concentric keratin pearl whorls typical of squamous differentiation.",
        },
        {
            "concept_id": "nonadeno_crush_artifact_neuroendocrine_high",
            "concept_en": "crush artifact and finely granular chromatin",
            "prompt": "A high magnification histopathology image showing neuroendocrine-type cells with crush artifact, scant cytoplasm, and finely granular chromatin.",
        },
    ],
}


def _append_prompts(base_obj, additions):
    obj = copy.deepcopy(base_obj)
    prompts = obj["prompts"]
    for (class_name, class_id, scale), items in additions.items():
        for item in items:
            prompts.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "scale": scale,
                    "concept_id": item["concept_id"],
                    "concept_en": item["concept_en"],
                    "prompt": item["prompt"],
                    "source": "expert_added_core_pool_extension",
                    "use_in_stage2": True,
                }
            )
    return obj


def _count_prompts(obj):
    counts = {}
    for item in obj["prompts"]:
        if not item.get("use_in_stage2", True):
            continue
        key = (item["class_name"], item["class_id"], item["scale"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _finalize_metadata(obj, size):
    obj["metadata"]["created_at"] = "2026-04-27T00:00:00+08:00"
    obj["metadata"]["description"] = (
        f"Core-{size} concept prompt pool for Stage-2 size-sweep ablation. "
        f"Each class_id and scale keeps {size} pathology morphology prompts."
    )
    obj["metadata"]["prompt_counts"] = {
        "Adenocarcinoma_low": size,
        "Adenocarcinoma_high": size,
        "NonAdenocarcinoma_low": size,
        "NonAdenocarcinoma_high": size,
    }
    obj["metadata"]["notes"] = [
        "Only pathology morphology concepts are included.",
        "Non-core concepts such as TNM, margin status, pleural invasion, and vascular invasion are excluded.",
        "This file is intended for embedding_mean concept-pool size sweeps.",
    ]
    return obj


def main():
    base_obj = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    obj12 = _append_prompts(base_obj, ADDITIONS_12)
    obj12 = _finalize_metadata(obj12, 12)
    counts12 = _count_prompts(obj12)

    obj14 = _append_prompts(obj12, ADDITIONS_14)
    obj14 = _finalize_metadata(obj14, 14)
    counts14 = _count_prompts(obj14)

    expected12 = {
        ("Adenocarcinoma", 0, "low"): 12,
        ("Adenocarcinoma", 0, "high"): 12,
        ("NonAdenocarcinoma", 1, "low"): 12,
        ("NonAdenocarcinoma", 1, "high"): 12,
    }
    expected14 = {
        ("Adenocarcinoma", 0, "low"): 14,
        ("Adenocarcinoma", 0, "high"): 14,
        ("NonAdenocarcinoma", 1, "low"): 14,
        ("NonAdenocarcinoma", 1, "high"): 14,
    }
    if counts12 != expected12:
        raise ValueError(f"Concept-12 counts mismatch: {counts12}")
    if counts14 != expected14:
        raise ValueError(f"Concept-14 counts mismatch: {counts14}")

    OUTPUT_12.write_text(json.dumps(obj12, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_14.write_text(json.dumps(obj14, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved: {OUTPUT_12}")
    print(f"Saved: {OUTPUT_14}")


if __name__ == "__main__":
    main()
