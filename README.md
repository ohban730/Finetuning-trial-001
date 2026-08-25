# Finetuning-trial-001

GPT-2 (355M) をベースに、銀行・運送会社それぞれのカスタマーセンター向け問い合わせ応答をファインチューニングする実験プロジェクトです。

## 構成

- `model_arch.py` — GPTモデル定義・GPT-2重みのダウンロード/ロード・学習ループ
- `main.py` — データ読み込み、ファインチューニング、2モデル比較検証の実行スクリプト
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
