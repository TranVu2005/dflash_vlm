"""Simple benchmark for DFlash VLM draft with Qwen3-VL target."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from .model import DFlashDraftModel, load_and_process_dataset


def build_text_input(processor: AutoProcessor, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], return_tensors="pt").to(device)["input_ids"]


@torch.inference_mode()
def run_one(
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[int, float]:
    input_ids = build_text_input(processor, prompt, target.device)

    t0 = time.perf_counter()
    output_ids = draft.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=[processor.tokenizer.eos_token_id],
        temperature=temperature,
    )
    elapsed = time.perf_counter() - t0

    generated_tokens = max(0, output_ids.shape[1] - input_ids.shape[1])
    return generated_tokens, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash VLM benchmark (text prompts)")
    parser.add_argument("--model", type=str, required=True, help="Target model, e.g. Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--draft-model", type=str, required=True, help="Draft checkpoint path or HF repo")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"])
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device(args.device)

    print(f"Loading target model: {args.model}")
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()

    print(f"Loading draft model: {args.draft_model}")
    from transformers import AutoConfig

    draft_cfg = AutoConfig.from_pretrained(args.draft_model, trust_remote_code=True)
    target_cfg = AutoConfig.from_pretrained(args.model)

    draft_text_cfg = draft_cfg.text_config if hasattr(draft_cfg, "text_config") else draft_cfg
    target_text_cfg = target_cfg.text_config if hasattr(target_cfg, "text_config") else target_cfg

    draft_text_cfg.hidden_size = target_text_cfg.hidden_size
    draft_text_cfg.num_attention_heads = target_text_cfg.num_attention_heads
    draft_text_cfg.num_key_value_heads = target_text_cfg.num_key_value_heads
    draft_text_cfg.intermediate_size = target_text_cfg.intermediate_size
    draft_text_cfg.head_dim = getattr(
        target_text_cfg,
        "head_dim",
        target_text_cfg.hidden_size // target_text_cfg.num_attention_heads,
    )

    target_cfg.text_config = draft_text_cfg
    target_cfg.dflash_config = getattr(draft_cfg, "dflash_config", {})
    target_cfg.block_size = getattr(draft_cfg, "block_size", 16)
    target_cfg.num_target_layers = getattr(draft_cfg, "num_target_layers", target_text_cfg.num_hidden_layers)

    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=target_cfg,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model)

    dataset = load_and_process_dataset(args.dataset)
    if len(dataset) > args.max_samples:
        random.shuffle(dataset)
        dataset = dataset[: args.max_samples]

    total_generated = 0
    total_time = 0.0

    for item in dataset:
        prompt = item["turns"][0] if isinstance(item["turns"], list) else item["turns"]
        n_gen, elapsed = run_one(
            draft=draft,
            target=target,
            processor=processor,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        total_generated += n_gen
        total_time += elapsed

    tps = total_generated / max(total_time, 1e-6)
    print("=" * 60)
    print(f"Samples: {len(dataset)}")
    print(f"Generated tokens: {total_generated}")
    print(f"Total decode time: {total_time:.2f}s")
    print(f"Throughput: {tps:.2f} tokens/s")


if __name__ == "__main__":
    main()
