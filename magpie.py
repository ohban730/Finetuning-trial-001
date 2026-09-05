import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import random
import json

hf_logging.set_verbosity_error()
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = "あなたは日本語で応答する運送会社のAIアシスタントです。{label}に関するユーザーからの日本語の質問に、正確かつ簡潔に答えてください。"
# SYSTEM_PROMPT = "あなたは日本語で応答する銀行のAIアシスタントです。{label}に関するユーザーからの日本語の質問に、正確かつ簡潔に答えてください。"
PROMPT = (
    "<|im_start|>system\n"
    f"{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n"
)
MAX_NEW_TOKENS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, prompt: str, **generate_kwargs) -> str:
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=None
        ).input_ids.to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text_gen_only = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return full_text, text_gen_only

def main() -> None:
    tokenizer, model = load_model()
    dataset_list = []
    for _ in range(1000):
        label = random.choice(["配達状況", "配送料金", "配送時間", "配達日時の指定・変更"])
        # label = random.choice(["口座開設の手続き", "口座情報の変更", "通帳・キャッシュカードの再発行", "NISA口座", "投資信託の購入・解約", "外貨預金の取引", "住宅ローンの申し込み", "カードローンの申し込み", "振込・送金の手続き", "ATMの利用方法"])
        sys_usr, instruction = generate(tokenizer, model, PROMPT.format(label=label))
        eot_str = "<|im_end|>"
        if not sys_usr.endswith(eot_str):
            sys_usr += eot_str
        Response_gen_input = sys_usr + "\n<|im_start|>assistant\n"
        sys_usr_response, output = generate(tokenizer, model, Response_gen_input)
        dataset_list.append({
            "instruction": instruction.replace(eot_str, ""),
            "input": "",
            "output": output.replace(eot_str, "")
        })
        if len(dataset_list) % 10 == 0:
            print(f"Generated {len(dataset_list)} samples")

    json.dump(dataset_list, open("dataset_trsprt.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()