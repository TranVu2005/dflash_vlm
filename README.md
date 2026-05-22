# dflash_vlm

DFlash implementation for vision-language models (VLM), focused on Qwen3-VL.

## Goals

- Fine-tune a DFlash draft model for `Qwen3-VL` targets.
- Run speculative decoding in a VLM pipeline.
- Benchmark generation throughput.

## Repository layout

- `dflash/model.py`: draft model architecture and VLM speculative decoding.
- `dflash/train_dflash_vlm.py`: training script using Arrow data.
- `dflash/eval_dflash_vlm.py`: acceptance-length evaluation script.
- `dflash/benchmark.py`: lightweight throughput benchmark.
- `pyproject.toml`: package metadata and dependencies.

## Installation

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

## Training data format

Expected Arrow columns:

- `input_ids`
- `attention_mask`
- `hidden_states` (shape: `seq_len x (len(target_layer_ids) * hidden_size)`)
- `image_hidden` (`None` for text-only samples)
- `logits_topk_indices` and `logits_topk_values` (for KL loss)

## Train

```bash
python -m dflash.train_dflash_vlm \
  --draft_model z-lab/Qwen3-4B-DFlash-b16 \
  --target_model Qwen/Qwen3-VL-4B-Instruct \
  --data_path ./data/train.arrow \
  --output_dir ./checkpoints \
  --loss_type kl \
  --batch_size 4 \
  --epochs 3
```

## Evaluate

Text-only:

```bash
python -m dflash.eval_dflash_vlm \
  --draft_model ./checkpoints/final \
  --target_model Qwen/Qwen3-VL-4B-Instruct \
  --prompt "Describe this scene in detail." \
  --block_size 16 \
  --max_new_tokens 256
```

With image:

```bash
python -m dflash.eval_dflash_vlm \
  --draft_model ./checkpoints/final \
  --target_model Qwen/Qwen3-VL-4B-Instruct \
  --image_path ./test.jpg \
  --prompt "What is in this image?" \
  --block_size 16 \
  --max_new_tokens 256
```

## Benchmark

```bash
python -m dflash.benchmark \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --draft-model ./checkpoints/final \
  --dataset gsm8k \
  --max-samples 32 \
  --max-new-tokens 128
```
