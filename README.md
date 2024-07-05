# Magpie-based Synthetic Dialogue Dataset Generator

[Magpie](https://arxiv.org/abs/2406.08464)という手法と、[DeepInfra](https://deepinfra.com/)上の[nvidia/Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)を使用して、合成対話データセットを生成するためのコードです。

## 概要

Magpieは、既存の大規模言語モデル（LLM）を使用して、高品質な指示データを大量に合成する手法です。

このコードは以下の論文に基づいています：

[Lin, B. Y., et al. (2024). "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing". arXiv preprint arXiv:2406.08464.](https://arxiv.org/abs/2406.08464)

## 使用方法

1. 必要なライブラリをインストールします。

    ```text
    pip install -r requirements.txt
    ```

2. [DeepInfra](https://deepinfra.com/)のサービスよりAPIキーを取得します。

3. 以下のようなコマンドを実行してデータセットを生成します。

    ```text
    python generate_dataset.py --api_key YOUR_API_KEY_HERE --target_count 1000 --num_turns 3 --max_workers 50 --user_max_tokens 256 --assistant_max_tokens 1024 --temperature 1.0 --output_file generated_conversations.jsonl
    ```

    パラメータの説明：
    - `--api_key`: DeepInfra APIキー（必須）
    - `--target_count`: 生成するデータの数（デフォルト: 1000）
    - `--num_turns`: 各データのターン数（デフォルト: 3）
    - `--max_workers`: 並行処理で使用するワーカー数（デフォルト: 50）
    - `--user_max_tokens`: 指示の最大トークン数（デフォルト: 256）
    - `--assistant_max_tokens`: 応答の最大トークン数（デフォルト: 1024）
    - `--temperature`: 生成の温度（デフォルト: 1.0）
    - `--output_file`: 出力ファイル名（デフォルト: generated_conversations.jsonl）

4. 生成されたデータセットは指定した出力ファイル（デフォルトでは `generated_conversations.jsonl`）に保存されます。

## 注意事項

- このコードはNemotron-4-340B-Instructの使用を前提としています。他のモデルを使用する場合は、コードの一部を変更する必要があります。
- DeepInfra以外で推論させる場合、コードの一部を変更する必要があります。
- 大量のデータを生成する場合は、APIの使用制限と料金に注意してください。
- 生成されたデータセットの品質と内容を必ず確認し、必要に応じてフィルタリングを行ってください。

## ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。
