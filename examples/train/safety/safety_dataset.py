"""
Preprocess a prompt-pairs JSONL to parquet format for safety classification
GRPO training. Uses walledai/XSTest from HuggingFace as the validation set.
Follows the same structure as examples/train/gsm8k/gsm8k_dataset.py.

Usage:
    uv run --no-project --python 3.12 --with datasets \
        examples/train/safety/safety_dataset.py \
        --data_path ~/aq_worktrial/sdg_output/prompt_pairs.jsonl \
        --output_dir ~/data/safety
"""

import argparse
import json
import os
import random
from datasets import Dataset, load_dataset


def normalize_categories(categories) -> list[str]:
    if categories is None:
        return []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",")]
    return [str(c).strip() for c in categories if str(c).strip()]


def make_example(
    prompt_text: str,
    label: str,
    split: str,
    idx: int,
    data_source: str,
    categories=None,
) -> dict:
    assert label in ("safe", "unsafe"), f"Unexpected label: {label!r}"
    categories = normalize_categories(categories)
    reward_spec = {
        "method": "rule",
        "ground_truth": label,
    }
    if categories:
        reward_spec["categories"] = categories
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ],
        "env_class": "safety",
        "reward_spec": reward_spec,
        "extra_info": {
            "split": split,
            "index": idx,
            "original_prompt": prompt_text,
            "label": label,
            "categories": categories,
        },
    }


def load_from_jsonl(path: str) -> list[dict]:
    """
    Read prompt_pairs.jsonl. Each row has a 'prompt' (safe) and an
    'unsafe_prompt' field. Expand each pair into two labelled examples.
    """
    examples = []
    with open(path) as f:
        for row in f:
            row = json.loads(row)
            examples.append({
                "prompt": row["prompt"],
                "label": "safe",
                "categories": row.get("categories") or row.get("prompt_categories"),
            })
            if "unsafe_prompt" in row:
                examples.append({
                    "prompt": row["unsafe_prompt"],
                    "label": "unsafe",
                    "categories": row.get("unsafe_categories") or row.get("unsafe_prompt_categories"),
                })
    return examples


def load_from_hf() -> list[dict]:
    raw = load_dataset("walledai/XSTest")
    split_name = "test" if "test" in raw else list(raw.keys())[0]
    dataset = raw[split_name]

    label_col = next(
        (c for c in ("label", "type", "safety_label") if c in dataset.column_names),
        None,
    )
    if label_col is None:
        raise ValueError(f"Cannot find label column in XSTest. Columns: {dataset.column_names}")
    return [{"prompt": row["prompt"], "label": dataset[label_col][i].lower(), "categories": None}
            for i, row in enumerate(dataset)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to local prompt_pairs.jsonl (used as training set).",
    )
    parser.add_argument("--output_dir", default="~/data/safety")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    data_source = os.path.basename(args.data_path)
    all_rows = load_from_jsonl(os.path.expanduser(args.data_path))

    safe_rows = [r for r in all_rows if r["label"] == "safe"]
    unsafe_rows = [r for r in all_rows if r["label"] == "unsafe"]
    # 80% safe, 20% unsafe → keep all safe, take safe_count // 4 unsafe
    unsafe_rows = unsafe_rows[: len(safe_rows) // 4]
    train_rows = safe_rows + unsafe_rows
    random.seed(42)
    random.shuffle(train_rows)
    train_rows = train_rows[:5120]

    val_rows = load_from_hf()

    train_dataset = Dataset.from_list([
        make_example(r["prompt"], r["label"], "train", i, data_source, r.get("categories"))
        for i, r in enumerate(train_rows)
    ])
    val_dataset = Dataset.from_list([
        make_example(r["prompt"], r["label"], "val", i, "walledai/XSTest")
        for i, r in enumerate(val_rows)
    ])

    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.output_dir, "validation.parquet"))

    label_counts: dict = {}
    for row in train_dataset:
        l = row["reward_spec"]["ground_truth"]
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"Saved {len(train_dataset)} train + {len(val_dataset)} val rows → {args.output_dir}/")
    print(f"Train label distribution: {label_counts}")
