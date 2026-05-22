"""
Fine-tune DFlash draft model for Qwen3-VL targets.

Expected Arrow row format:
{
    "input_ids": [int, ...],
    "attention_mask": [int, ...],
    "hidden_states": [[float, ...]],
    "image_hidden": [[float, ...]] or None,
    "logits_topk_indices": [[int, ...]],
    "logits_topk_values": [[float, ...]],
}
"""

import argparse
import math
import random
from pathlib import Path

import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_cosine_schedule_with_warmup
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class DFlashVLMDataset(Dataset):
    """Memory-mapped Arrow dataset for DFlash VLM training."""

    def __init__(self, data_path: str, max_seq_len: int = 2048, use_kl: bool = True):
        self.max_seq_len = max_seq_len
        self.use_kl = use_kl
        reader = ipc.open_file(data_path)
        self.table = reader.read_all()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1)

        input_ids = torch.tensor(row.column("input_ids")[0].as_py(), dtype=torch.long)
        attention_mask = torch.tensor(row.column("attention_mask")[0].as_py(), dtype=torch.long)
        hidden_states = torch.tensor(row.column("hidden_states")[0].as_py(), dtype=torch.float32)

        seq_len = min(len(input_ids), self.max_seq_len)
        input_ids = input_ids[:seq_len]
        attention_mask = attention_mask[:seq_len]
        hidden_states = hidden_states[:seq_len]
        position_ids = torch.arange(seq_len, dtype=torch.long)

        try:
            img_raw = row.column("image_hidden")[0].as_py()
            if img_raw is not None and len(img_raw) > 0:
                image_hidden = torch.tensor(img_raw, dtype=torch.float32)
            else:
                image_hidden = None
        except Exception:
            image_hidden = None

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "image_hidden": image_hidden,
        }

        if self.use_kl:
            try:
                item["topk_indices"] = torch.tensor(
                    row.column("logits_topk_indices")[0].as_py(), dtype=torch.long
                )[:seq_len]
                item["topk_values"] = torch.tensor(
                    row.column("logits_topk_values")[0].as_py(), dtype=torch.float32
                )[:seq_len]
            except Exception:
                pass

        return item


def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    has_kl = "topk_indices" in batch[0]

    input_ids_list = []
    attention_mask_list = []
    position_ids_list = []
    hidden_states_list = []
    image_hidden_list = []
    topk_indices_list = []
    topk_values_list = []

    for x in batch:
        seq_len = x["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(F.pad(x["input_ids"], (0, pad_len), value=0))
        attention_mask_list.append(F.pad(x["attention_mask"], (0, pad_len), value=0))
        position_ids_list.append(F.pad(x["position_ids"], (0, pad_len), value=0))
        hidden_states_list.append(F.pad(x["hidden_states"], (0, 0, 0, pad_len), value=0.0))
        image_hidden_list.append(x["image_hidden"])

        if has_kl:
            topk_indices_list.append(F.pad(x["topk_indices"], (0, 0, 0, pad_len), value=0))
            topk_values_list.append(F.pad(x["topk_values"], (0, 0, 0, pad_len), value=0.0))

    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "position_ids": torch.stack(position_ids_list),
        "hidden_states": torch.stack(hidden_states_list),
        "image_hidden": image_hidden_list,
    }

    if has_kl:
        result["topk_indices"] = torch.stack(topk_indices_list)
        result["topk_values"] = torch.stack(topk_values_list)

    return result


def make_block_noise(
    input_ids: torch.Tensor,
    mask_token_id: int,
    block_size: int,
    randomize_size: bool = True,
) -> tuple[torch.Tensor, int, int]:
    bsz, seq_len = input_ids.shape
    actual_block = random.randint(2, block_size) if randomize_size else block_size

    max_start = seq_len - actual_block - 1
    if max_start < 1:
        actual_block = max(2, min(block_size, seq_len - 1))
        start = 1
    else:
        start = random.randint(1, max_start)

    noise_ids = input_ids.clone()
    noise_ids[:, start + 1 : start + actual_block] = mask_token_id

    predict_len = actual_block - 1
    return noise_ids, start, predict_len


def make_decay_weights(predict_len: int, gamma: float, device: torch.device) -> torch.Tensor:
    weights = torch.tensor([gamma**i for i in range(predict_len)], dtype=torch.float32, device=device)
    return weights / weights.sum()


def ce_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 0.7) -> torch.Tensor:
    bsz, predict_len, vocab_size = logits.shape
    weights = make_decay_weights(predict_len, gamma, logits.device)

    loss_per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape(bsz, predict_len)

    return (loss_per_token * weights.unsqueeze(0)).sum(dim=1).mean()


def kl_loss(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_values: torch.Tensor,
    vocab_size: int,
    temperature: float = 1.0,
    gamma: float = 0.7,
) -> torch.Tensor:
    bsz, predict_len, vocab = logits.shape
    weights = make_decay_weights(predict_len, gamma, logits.device)

    draft_log_probs = F.log_softmax(logits / temperature, dim=-1)

    target_sparse = torch.full_like(logits, fill_value=float("-inf"))
    target_sparse.scatter_(dim=-1, index=topk_indices, src=topk_values.to(logits.dtype) / temperature)
    target_probs = F.softmax(target_sparse, dim=-1)

    kl_per_token = F.kl_div(
        draft_log_probs.reshape(-1, vocab),
        target_probs.reshape(-1, vocab),
        reduction="none",
        log_target=False,
    ).sum(dim=-1).reshape(bsz, predict_len)

    return (kl_per_token * weights.unsqueeze(0)).sum(dim=1).mean()


def train_step(
    draft: nn.Module,
    target_lm_head: nn.Module,
    target_embed_tokens: nn.Module,
    batch: dict,
    mask_token_id: int,
    block_size: int,
    vocab_size: int,
    loss_type: str,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    hidden_states = batch["hidden_states"].to(device, dtype=torch.bfloat16)

    bsz = input_ids.shape[0]
    text_cfg = getattr(draft, "text_config", None) or (
        draft.config.text_config if hasattr(draft.config, "text_config") else draft.config
    )
    hidden_size = text_cfg.hidden_size

    img_list = [x.to(device, dtype=torch.bfloat16) if x is not None else None for x in batch["image_hidden"]]
    max_patch = max((x.shape[0] for x in img_list if x is not None), default=0)

    if max_patch > 0:
        padded_imgs = []
        for img in img_list:
            if img is None:
                padded_imgs.append(torch.zeros(max_patch, hidden_size, device=device, dtype=torch.bfloat16))
            else:
                pad_len = max_patch - img.shape[0]
                if pad_len > 0:
                    img = F.pad(img, (0, 0, 0, pad_len), value=0.0)
                padded_imgs.append(img)
        image_hidden = torch.stack(padded_imgs)
    else:
        image_hidden = torch.zeros(bsz, 0, hidden_size, device=device, dtype=torch.bfloat16)

    noise_ids, anchor_pos, predict_len = make_block_noise(
        input_ids,
        mask_token_id,
        block_size,
        randomize_size=True,
    )
    noise_ids = noise_ids.to(device)

    total_block = predict_len + 1
    block_noise_ids = noise_ids[:, anchor_pos : anchor_pos + total_block]
    with torch.no_grad():
        noise_embedding = target_embed_tokens(block_noise_ids)

    ctx_hidden = hidden_states[:, :anchor_pos, :]

    block_position_ids = torch.arange(anchor_pos + total_block, device=device).unsqueeze(0).expand(bsz, -1)

    draft_hidden = draft(
        noise_embedding=noise_embedding,
        target_hidden=ctx_hidden,
        image_hidden=image_hidden,
        position_ids=block_position_ids,
    )

    draft_hidden_pred = draft_hidden[:, 1:, :]
    logits = target_lm_head(draft_hidden_pred)
    labels = input_ids[:, anchor_pos + 1 : anchor_pos + 1 + predict_len]

    if loss_type == "kl" and "topk_indices" in batch:
        topk_indices = batch["topk_indices"][:, anchor_pos + 1 : anchor_pos + 1 + predict_len, :].to(device)
        topk_values = batch["topk_values"][:, anchor_pos + 1 : anchor_pos + 1 + predict_len, :].to(device)
        loss = kl_loss(logits, topk_indices, topk_values, vocab_size, gamma=gamma)
    else:
        loss = ce_loss(logits, labels, gamma=gamma)

    return loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DFlash draft model for Qwen3-VL")
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--loss_type", type=str, default="kl", choices=["ce", "kl"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading draft model...")
    draft = AutoModel.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).train()

    dflash_cfg = getattr(draft.config, "dflash_config", {})
    mask_token_id = dflash_cfg.get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("Missing mask_token_id in draft config dflash_config")

    num_target_layers_used = len(draft.target_layer_ids)
    text_cfg = getattr(draft, "text_config", None) or (
        draft.config.text_config if hasattr(draft.config, "text_config") else draft.config
    )
    expected_hidden_dim = num_target_layers_used * text_cfg.hidden_size
    print(
        f"Draft fc input dim: {expected_hidden_dim} "
        f"({num_target_layers_used} layers x {text_cfg.hidden_size})"
    )
    print(f"Dataset hidden_states last dim must be: {expected_hidden_dim}")

    print("Loading target lm_head + embed_tokens...")
    target_full = Qwen3VLForConditionalGeneration.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    target_lm_head = target_full.lm_head.to(device).eval()
    target_embed_tokens = target_full.model.language_model.embed_tokens.to(device).eval()
    vocab_size = target_full.config.text_config.vocab_size

    for p in target_lm_head.parameters():
        p.requires_grad_(False)
    for p in target_embed_tokens.parameters():
        p.requires_grad_(False)

    del target_full
    torch.cuda.empty_cache()

    trainable_params = [p for p in draft.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_params / 1e6:.1f}M")

    dataset = DFlashVLMDataset(args.data_path, max_seq_len=args.max_seq_len, use_kl=args.loss_type == "kl")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    print(f"Dataset: {len(dataset)} samples | {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Total steps: {total_steps} | Warmup: {warmup_steps} | Loss: {args.loss_type.upper()}")

    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            loss = train_step(
                draft=draft,
                target_lm_head=target_lm_head,
                target_embed_tokens=target_embed_tokens,
                batch=batch,
                mask_token_id=mask_token_id,
                block_size=args.block_size,
                vocab_size=vocab_size,
                loss_type=args.loss_type,
                gamma=args.gamma,
                device=device,
            )

            (loss / args.grad_accum).backward()
            accum_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                print(
                    f"[{epoch + 1}/{args.epochs}] step {global_step}/{total_steps} | "
                    f"loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )

                if global_step % args.save_steps == 0:
                    ckpt = Path(args.output_dir) / f"step-{global_step}"
                    draft.save_pretrained(ckpt)
                    print(f"Saved {ckpt}")

        epoch_ckpt = Path(args.output_dir) / f"epoch-{epoch + 1}"
        draft.save_pretrained(epoch_ckpt)
        print(f"Epoch {epoch + 1} done -> {epoch_ckpt}")

    final_ckpt = Path(args.output_dir) / "final"
    draft.save_pretrained(final_ckpt)
    print(f"Training complete -> {final_ckpt}")


if __name__ == "__main__":
    main()
