"""Evaluate DFlash VLM draft acceptance behavior against a Qwen3-VL target."""

import argparse
import time

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

try:
    from .model import DFlashDraftModel
except ImportError:
    from model import DFlashDraftModel


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def compute_acceptance_length(draft_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
    match = draft_tokens == target_tokens
    chain = match.cumprod(dim=1)
    return chain.sum(dim=1).float().mean().item()


@torch.inference_mode()
def spec_generate_eval(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 0.0,
    stop_token_ids: list[int] | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
):
    from transformers import DynamicCache

    raw_cfg = draft.config.dflash_config if hasattr(draft.config, "dflash_config") else {}
    mask_token_id = raw_cfg.get("mask_token_id") if isinstance(raw_cfg, dict) else getattr(raw_cfg, "mask_token_id", None)
    target_layer_ids = draft.target_layer_ids

    device = next(target.parameters()).device

    def extract_context_feature(hidden_states, layer_ids):
        offset = 1
        return torch.cat([hidden_states[lid + offset] for lid in layer_ids], dim=-1)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    prefill_kwargs = dict(
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_kv_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    if pixel_values is not None:
        prefill_kwargs["pixel_values"] = pixel_values
        prefill_kwargs["image_grid_thw"] = image_grid_thw

    output = target(input_ids, **prefill_kwargs)
    num_cached_tokens = past_kv_target.get_seq_length()

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, target_layer_ids)

    acceptance_lengths = []
    start = num_input_tokens
    t0 = time.perf_counter()

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        noise_embedding = target.model.language_model.embed_tokens(block_output_ids)

        draft_text_start = start - num_input_tokens
        draft_cache_len = past_kv_draft.get_seq_length()

        draft_out = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, draft_cache_len : draft_text_start + block_size],
            past_key_values=past_kv_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target.lm_head(draft_out[:, -block_size + 1 :, :])
        past_kv_draft.crop(draft_text_start)

        draft_proposed = sample(draft_logits, temperature)
        block_output_ids[:, 1:] = draft_proposed

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits, temperature)

        acc_len = compute_acceptance_length(draft_proposed, posterior[:, :-1])
        acceptance_lengths.append(acc_len)

        n_accept = int(round(acc_len))
        output_ids[:, start : start + n_accept + 1] = block_output_ids[:, : n_accept + 1]
        output_ids[:, start + n_accept + 1] = posterior[:, n_accept]
        start += n_accept + 1

        past_kv_target.crop(num_cached_tokens + start - num_input_tokens)
        target_hidden = extract_context_feature(output.hidden_states, target_layer_ids)[:, : n_accept + 1, :]

        if stop_token_ids:
            stop_t = torch.tensor(stop_token_ids, device=device)
            if torch.isin(output_ids[0, num_input_tokens:], stop_t).any():
                break

    elapsed = time.perf_counter() - t0

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=device)
        indices = torch.isin(output_ids[0][num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + indices[0] + 1]

    n_generated = output_ids.shape[1] - num_input_tokens
    stats = {
        "acceptance_lengths": acceptance_lengths,
        "mean_acceptance_length": sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0,
        "n_draft_steps": len(acceptance_lengths),
        "n_generated_tokens": n_generated,
        "tokens_per_second": n_generated / elapsed if elapsed > 0 else 0.0,
        "elapsed_sec": elapsed,
        "avg_tokens_per_target_call": (n_accept + 1) if acceptance_lengths else 1.0,
    }
    return output_ids, stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate DFlash VLM acceptance length")
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading target model...")
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        args.target_model,
        dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()

    print("Loading draft model...")
    draft_config = AutoConfig.from_pretrained(args.draft_model)
    target_config = AutoConfig.from_pretrained(args.target_model)

    def get_text_cfg(cfg):
        return cfg.text_config if hasattr(cfg, "text_config") else cfg

    draft_text_cfg = get_text_cfg(draft_config)
    target_text_cfg = get_text_cfg(target_config)

    draft_text_cfg.hidden_size = target_text_cfg.hidden_size
    draft_text_cfg.num_attention_heads = target_text_cfg.num_attention_heads
    draft_text_cfg.num_key_value_heads = target_text_cfg.num_key_value_heads
    draft_text_cfg.intermediate_size = target_text_cfg.intermediate_size
    draft_text_cfg.head_dim = getattr(
        target_text_cfg,
        "head_dim",
        target_text_cfg.hidden_size // target_text_cfg.num_attention_heads,
    )

    target_config.text_config = draft_text_cfg
    target_config.dflash_config = getattr(draft_config, "dflash_config", {})
    target_config.block_size = getattr(draft_config, "block_size", 16)
    target_config.num_target_layers = getattr(draft_config, "num_target_layers", target_text_cfg.num_hidden_layers)

    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=target_config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(args.target_model)

    pixel_values = None
    image_grid_thw = None

    if args.image_path:
        from PIL import Image

        image = Image.open(args.image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

    stop_token_ids = [processor.tokenizer.eos_token_id]

    print(f"\nPrompt: {args.prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Block size: {args.block_size} | Max new tokens: {args.max_new_tokens}")
    print("-" * 60)

    output_ids, stats = spec_generate_eval(
        draft=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=args.temperature,
        stop_token_ids=stop_token_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    generated = processor.tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
    print(f"\nOutput:\n{generated}")
    print("\n" + "=" * 60)
    print("EVAL STATS")
    print("=" * 60)
    print(f"  Mean acceptance length : {stats['mean_acceptance_length']:.3f} / {args.block_size - 1}")
    print(f"  Draft steps            : {stats['n_draft_steps']}")
    print(f"  Tokens generated       : {stats['n_generated_tokens']}")
    print(f"  Tokens / second        : {stats['tokens_per_second']:.1f}")
    print(f"  Elapsed                : {stats['elapsed_sec']:.2f}s")
    print(f"  Avg tokens/target call : {stats['avg_tokens_per_target_call']:.2f}")


if __name__ == "__main__":
    main()
