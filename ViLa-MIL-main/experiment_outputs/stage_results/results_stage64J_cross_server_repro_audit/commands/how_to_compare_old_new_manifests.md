# Stage64J Manifest Comparison

## 1. Collect manifests on each server

New server example:

```bash
bash run_manifest_on_new_server.sh \
  /private/ljh-data/shared/Linux_school/ViLMIL \
  /private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/yiyuan \
  /tmp/stage64J_new_manifest
```

Old server example:

```bash
bash run_manifest_on_old_server.sh \
  /path/to/ViLMIL_repo_root \
  /path/to/data_root \
  /tmp/stage64J_old_manifest
```

## 2. Compare manifests

```bash
python /private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/tools/compare_stage64J_manifests.py \
  --old-dir /tmp/stage64J_old_manifest \
  --new-dir /tmp/stage64J_new_manifest \
  --output /tmp/stage64J_manifest_compare.json
```

## 3. Read the comparison

- `environment_version_differences`: package, Python, CUDA, Git differences.
- `asset_compare`: ordinary file SHA and metadata differences.
- `h5_compare_5x` and `h5_compare_20x`: H5 SHA, shape, dtype, feature-hash, coords-hash differences.

If `sha256_different` is empty for all asset sections, data contents match across servers for the captured manifests.
