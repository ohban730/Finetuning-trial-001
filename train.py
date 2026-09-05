"""main.pyでコメントアウトされていた学習手順を、データセット（finance/transport）を指定して
再実行できるようにした独立スクリプト。ハイパーパラメータはmain.py内のコメントアウトされた
元のコードと同じ値を使用している。
"""
import argparse
import sys
import time
from functools import partial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from model_arch import (
    GPTModel,
    download_and_load_gpt2,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    train_model_simple,
)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_lenght=None, device="cuda"):
    max_length = max(len(item) + 1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_lenght is not None:
            inputs = inputs[:allowed_max_lenght]
            targets = targets[:allowed_max_lenght]

        inputs_list.append(inputs)
        targets_list.append(targets)

    input_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return input_tensor, targets_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True, help="学習済み重みの保存先(.pth)")
    parser.add_argument("--loss_plot_path", required=True, help="損失曲線の保存先(.png)")
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 元のmain.pyはbatch_size=8, allowed_max_lenght=1024だったが、model_arch.pyの
    # Attention実装はメモリ効率の良いFlash Attention等を使っていない素朴な実装のため、
    # 長めの応答を含むバッチでGPUメモリを使い切りスラッシングする事例が確認された。
    # 安全マージンを確保するため小さめの値に変更している。
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_lenght=512)
    batch_size = 4
    torch.manual_seed(123)

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
    }

    settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 2

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer,
    )
    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epoch_tensor = torch.linspace(0, num_epochs, len(train_losses))

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epoch_tensor, train_losses, label="Training loss")
    ax1.plot(epoch_tensor, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(args.loss_plot_path)

    torch.save(model.state_dict(), args.output_path)
    print(f"Saved model to {args.output_path}, loss plot to {args.loss_plot_path}")


if __name__ == "__main__":
    main()
