from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import tiktoken
import time
from model_arch import GPTModel, download_and_load_gpt2, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text, download_file, plot_losses, train_model_simple
import re

### 学習データの読み込み
with open("dataset_finance.json", "r", encoding="utf-8") as f:
    data = json.load(f)

### プロンプトフォーマット関数を作成
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    return instruction_text + input_text

### データセットの分割
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")
print(f"Validation data size: {len(val_data)}")

### データセットの作成
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

tokenizer = tiktoken.get_encoding("gpt2")

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_lenght=None, device="cuda"):
    max_length = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []


    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] 

        padded = (
            new_item + [pad_token_id] * (max_length - len(new_item))
        )
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

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_config = {
    "gpt2-small (124M)": {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12},
    "gpt2-medium (355M)": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16},
    "gpt2-large (774M)": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_lenght=1024)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

CHOOSEN_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_config[CHOOSEN_MODEL])

model_size = CHOOSEN_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

setteings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

# model = GPTModel(BASE_CONFIG)
# load_weights_into_gpt(model, params)
# model.to(device)
# model.eval()


#######################比較検証########################

### 学習データの読み込み
with open("comparison_prompts.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# finance_model_path = "gpt2-medium355M-finance-cc.pth"
model_f = GPTModel(BASE_CONFIG)
model_f.load_state_dict(torch.load("gpt2-medium355M-finance-cc.pth"))
model_f.to(device)
model_f.eval()

# transport_model_path = "gpt2-medium355M-transport-cc.pth
model_t = GPTModel(BASE_CONFIG)
model_t.load_state_dict(torch.load("gpt2-medium355M-transport-cc.pth"))
model_t.to(device)
model_t.eval()

for entry in eval_data:
    input_text = format_input(entry)
    ids = generate(model=model_f, idx=text_to_token_ids(input_text, tokenizer).to(device), max_new_tokens=150, context_size=BASE_CONFIG["context_length"], eos_id=50256)
    entry["bank_model_response"] = token_ids_to_text(ids, tokenizer)[len(input_text):].replace("### Response:", "").strip()
    ids = generate(model=model_t, idx=text_to_token_ids(input_text, tokenizer).to(device), max_new_tokens=150, context_size=BASE_CONFIG["context_length"], eos_id=50256)
    entry["transport_model_response"] = token_ids_to_text(ids, tokenizer)[len(input_text):].replace("### Response:", "").strip()

with open("comparison_prompts.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)

# input_text = format_input(val_data[0])
# print("Input text:", input_text)

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids(input_text, tokenizer).to(device),
#     max_new_tokens=150,
#     context_size=BASE_CONFIG["context_length"],
#     eos_id=50256
# )
# generated_text = token_ids_to_text(token_ids, tokenizer)

# response_text = generated_text[len(input_text):].strip()
# print("Generated response:", response_text)


# start_time = time.time()

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# num_epochs = 2

# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device, num_epochs = num_epochs, eval_freq=5, eval_iter=5,
#     start_context = format_input(val_data[0]), tokenizer=tokenizer
# )

# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# epoch_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epoch_tensor, tokens_seen, train_losses, val_losses)


# file_name = f"{re.sub(r'[ ()]', '', CHOOSEN_MODEL)}-finance-cc.pth"
# torch.save(model.state_dict(), file_name)