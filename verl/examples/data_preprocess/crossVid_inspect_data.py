"""
Quick inspection of CrossVid preprocessed parquet: columns, sample content, answer leakage risk.
"""
import json
import numpy as np
import pandas as pd

# Default parquet path; change as needed
parquet_file = "./format_data/crossvid_train_mini.parquet"

try:
    df = pd.read_parquet(parquet_file)
    print(f"Read OK: {parquet_file}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")

    samples_to_check = df.head(3)

    for idx, row in samples_to_check.iterrows():
        print(f"\n{'='*40} Sample {idx} {'='*40}")

        reward_model_data = row.get("reward_model", {})
        if hasattr(reward_model_data, "item"):
            reward_model_data = reward_model_data.item()
        ground_truth = reward_model_data.get("ground_truth", "N/A")
        print(f"Ground truth: {ground_truth}")

        prompts = row.get("prompt", [])
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()

        print("\nPrompt content:")
        user_content = ""
        for msg in prompts:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                print(f"  [System]: (length {len(content)})")
            elif role == "user":
                user_content = content
                print(f"  [User]:\n{'-'*20}\n{content}\n{'-'*20}")

        # Check if answer string appears in user text (short answers like 'A' need manual check)
        is_suspicious = str(ground_truth).strip() in user_content
        print(f"\nLeakage risk: {'WARNING: answer appears in User Prompt, verify manually' if is_suspicious else 'OK: answer string not detected'}")

        extra_info = row.get("extra_info", {})
        if hasattr(extra_info, "keys"):
            print(f"\nextra_info keys: {list(extra_info.keys())}")

except FileNotFoundError:
    print(f"File not found: {parquet_file}. Check path.")
except Exception as e:
    print(f"Error: {e}")
