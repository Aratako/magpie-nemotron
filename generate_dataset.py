import argparse
import concurrent.futures
import json

import requests
from tqdm import tqdm

API_URL = "https://api.deepinfra.com/v1/inference/nvidia/Nemotron-4-340B-Instruct"


def generate_message(prompt, is_user_turn, api_key, max_tokens, temperature):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    stop = [
        "<extra_id_0>",
        "<extra_id_1>",
        "\u0011",
        "<|endoftext|>",
    ]
    if is_user_turn:
        stop += ["\n\n"]

    input_data = {
        "input": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "stop": stop,
    }

    response = requests.post(API_URL, headers=headers, json=input_data)
    response_data = response.json()

    generated_text = response_data["results"][0]["generated_text"].strip()
    cost = response_data["inference_status"]["cost"]

    return generated_text, cost


def format_prompt(messages):
    if not messages:
        return "<extra_id_0>System\n以下の難易度の高い質問に日本語で答えてください。\n<extra_id_1>User\n"
    prompt = "<extra_id_0>System\n以下の難易度の高い質問に日本語で答えてください。\n"
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        prompt += f"<extra_id_1>{role}\n{message['content']}\n"
    next_role = "Assistant" if messages[-1]["role"] == "user" else "User"
    prompt += f"<extra_id_1>{next_role}\n"
    return prompt


def generate_conversation(
    conversation_id,
    num_turns,
    api_key,
    user_max_tokens,
    assistant_max_tokens,
    temperature,
):
    messages = []
    total_cost = 0

    for turn in range(num_turns * 2):
        is_user_turn = turn % 2 == 0
        role = "user" if is_user_turn else "assistant"
        max_tokens = user_max_tokens if is_user_turn else assistant_max_tokens

        prompt = format_prompt(messages)
        content, cost = generate_message(
            prompt, is_user_turn, api_key, max_tokens, temperature
        )

        messages.append({"role": role, "content": content})
        total_cost += cost

    return {"id": conversation_id, "messages": messages}, total_cost


def main(args):
    api_key = args.api_key
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

    all_conversations = []
    total_cost = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        future_to_id = {
            executor.submit(
                generate_conversation,
                i,
                args.num_turns,
                api_key,
                args.user_max_tokens,
                args.assistant_max_tokens,
                args.temperature,
            ): i
            for i in range(args.target_count)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_id),
            total=args.target_count,
        ):
            conversation_id = future_to_id[future]
            try:
                result, cost = future.result()
                all_conversations.append(result)
                total_cost += cost
                print(
                    f"データ {conversation_id + 1}/{args.target_count} 完了, 累計コスト: ${total_cost:.4f}"
                )
            except Exception as e:
                print(f"データ {conversation_id} の生成中にエラーが発生しました: {e}")

    print(f"総コスト: ${total_cost:.4f}")

    # 結果をIDでソートして出力
    all_conversations.sort(key=lambda x: x["id"])

    with open(args.output_file, "w", encoding="utf-8") as f:
        for conversation in all_conversations:
            json.dump(conversation, f, ensure_ascii=False)
            f.write("\n")

    print(
        f"{len(all_conversations)}件のデータをJSONL形式で {args.output_file} に出力しました。"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Magpie dataset creation using Nemotron-4-340B-Instruct"
    )
    parser.add_argument("--api_key", type=str, required=True, help="DeepInfra API Key")
    parser.add_argument(
        "--target_count", type=int, default=1000, help="生成するデータの数"
    )
    parser.add_argument("--num_turns", type=int, default=3, help="各データのターン数")
    parser.add_argument(
        "--max_workers", type=int, default=50, help="並行処理で使用するワーカー数"
    )
    parser.add_argument(
        "--user_max_tokens", type=int, default=256, help="指示の最大トークン数"
    )
    parser.add_argument(
        "--assistant_max_tokens",
        type=int,
        default=1024,
        help="応答の最大トークン数",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="生成の温度")
    parser.add_argument(
        "--output_file",
        type=str,
        default="generated_conversations.jsonl",
        help="出力ファイル名",
    )

    args = parser.parse_args()
    main(args)
