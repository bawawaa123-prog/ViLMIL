# Stage13 RCE Evidence Export

- fold: `0`
- split: `test`
- max_slides: `10`
- checkpoint_path: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/rce_mil_v3_prior_calib_vr_a005_5fold_e20_s1/s_0_checkpoint.pt`
- exported_slides: `0`

## Output Files


## Warnings

- Failed to initialize RCE model: Failed initial config/weights load from HF Hub microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224: Failed to download file (open_clip_config.json) for microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224. Last error: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json (Caused by ProxyError(\'Unable to connect to proxy\', NewConnectionError("HTTPSConnection(host=\'127.0.0.1\', port=7897): Failed to establish a new connection: [Errno 1] Operation not permitted")))'), '(Request ID: ddbbc9c5-ce1a-455a-baed-8e932d62a6e8)')

## Next Step

Step14: concept-class graph or evidence visualization.
