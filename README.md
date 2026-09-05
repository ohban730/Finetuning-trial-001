# Finetuning-trial-001

GPT-2 (355M) をベースに、銀行・運送会社それぞれのカスタマーセンター向け問い合わせ応答をファインチューニングする実験プロジェクトです。

## 構成

- `model_arch.py` — GPTモデル定義・GPT-2重みのダウンロード/ロード・学習ループ
- `magpie.py` — MAGPIE手法によるデータ生成（`SYSTEM_PROMPT`/labelリストと出力ファイル名をコメントアウトで切り替えて銀行/運送それぞれ実行する）
- `train.py` — `dataset_finance.json` / `dataset_trsprt.json` を指定してのファインチューニング実行スクリプト
- `main.py` — データ読み込み、2モデル比較検証（`comparison_prompts.json`への応答生成）の実行スクリプト
- `dataset_trsprt.json` / `dataset_finance.json` — 学習データ（運送/金融ドメイン）
- `comparison_prompts.json` — 両ドメインにまたがる比較検証用プロンプトと、各モデルの応答

## セットアップ

```bash
pip install -r requirements.txt
python main.py
```

学習済みの重み（`*.pth`）は容量の都合上このリポジトリには含めていません。Hugging Face Hub で別途配布予定です。

## ライセンス

本プロジェクトのコード（`model_arch.py` を中心に）は [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（Copyright (c) 2023-2026 Sebastian Raschka, Apache License 2.0）のコードを組み合わせ・改変して使用しています。ライセンス全文は [LICENSE](LICENSE)、帰属表示の詳細は [NOTICE](NOTICE) を参照してください。

本プロジェクトが読み込む事前学習済みGPT-2の重みはOpenAIが独自のModified MIT Licenseで配布しているものであり、上記Apache Licenseの対象外です。

### 学習データについて

学習データ（`dataset_finance.json` / `dataset_trsprt.json`）は`Qwen/Qwen2.5-7B-Instruct`（Apache License 2.0、ゲートなし）の出力をMAGPIE手法で合成したものです。

当初は`meta-llama/Meta-Llama-3-8B-Instruct`を使用していましたが、同モデルが従うMeta Llama 3 Community Licenseの第1.b.v項に「Llama 3の出力を使って他の大規模言語モデルを改善（ファインチューニング）してはならない」という制限があることに後から気づき、本プロジェクトの用途がこの制限に文面上抵触する可能性があったため、Qwen2.5-7B-Instructにデータ生成モデルを切り替え、データ・学習済みモデルを作り直しました。他のモデルの出力を学習データに使う際は、ベースモデルのライセンス条項（特に生成物の再利用・再配布・派生モデルの学習に関する制限）を事前に確認することをお勧めします。
