# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess CrossVid raw data into parquet for Agent multi-turn GRPO training.
This script is a mini version: takes only the first N samples per task.
"""

import argparse
import json
import os
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs


# Task config: folder, prompt file, and file pattern for each of the nine task types
TASK_CONFIG = {
    "assembly": {
        "folder": "assembly_synthesis",
        "prompt_file": "build_assembly.prompt",
        "file_pattern": "task_*.json",
        "has_options": True,
    },
    "behavior": {
        "folder": "behavior_synthesis",
        "prompt_file": "build_behavior.prompt",
        "file_pattern": "task_*.json",
        "has_options": True,
    },
    "cooking": {
        "folder": "cooking_synthesis",
        "prompt_file": "build_cooking.prompt",
        "file_pattern": "task_*.json",
        "has_options": True,
    },
    "grounding": {
        "folder": "grounding_synthesis",
        "prompt_file": "build_grounding.prompt",
        "file_pattern": "task_*.json",
        "has_options": False,
    },
    "movie": {
        "folder": "movie_synthesis",
        "prompt_file": "build_movie.prompt",
        "file_pattern": "task_*.json",
        "has_options": True,
    },
    "plot": {
        "folder": "plot_synthesis",
        "prompt_file": "build_plot.prompt",
        "file_pattern": "task_*.json",
        "has_options": True,
    },
    "sort": {
        "folder": "sort_synthesis",
        "prompt_file": "build_sort.prompt",
        "file_pattern": "task_*.json",
        "has_options": False,
    },
    "uav_count": {
        "folder": "uav_count_synthesis",
        "prompt_file": "build_uavcount.prompt",
        "file_pattern": "moc_task_*.json",
        "has_options": True,
        "use_question_text": True,
    },
    "uav_position": {
        "folder": "uav_position_synthesis",
        "prompt_file": "build_uavpostion.prompt",
        "file_pattern": "msr_task_*.json",
        "has_options": True,
        "use_question_text": True,
    },
}


def load_system_prompt(prompt_file_path):
    """Load system prompt from prompt file."""
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def construct_user_content(task_data, task_config):
    """Build user-side content (question + optional options) from task config."""
    if task_config.get("use_question_text", False):
        if isinstance(task_data.get("question"), dict):
            question = task_data["question"].get("text", "")
        else:
            question = task_data.get("question", "")
    else:
        question = task_data.get("question", "")

    if task_config["has_options"]:
        options = task_data.get("options", [])
        if not options and isinstance(task_data.get("question"), dict):
            options = task_data["question"].get("options", [])
        if options:
            if isinstance(options, list):
                options_str = "\n".join(options)
            else:
                options_str = str(options)
            user_content = f"{question}\n\n{options_str}"
        else:
            user_content = question
    else:
        user_content = question

    return user_content


def get_correct_answer(task_data, task_config):
    """Parse correct answer from task JSON (supports multiple field names and types)."""
    if "correct_answer" in task_data:
        return task_data["correct_answer"]
    if isinstance(task_data.get("question"), dict) and "correct_answer" in task_data["question"]:
        return task_data["question"]["correct_answer"]
    elif "answer" in task_data:
        answer = task_data["answer"]
        if isinstance(answer, list):
            if len(answer) > 0:
                if isinstance(answer[0], (int, float)):
                    return str(answer)
                if isinstance(answer[0], str):
                    return answer[0] if len(answer) == 1 else "".join(sorted(answer))
            return str(answer)
        return answer
    else:
        return ""


def process_task_data(task_name, task_config, data_dir, prompts_dir, num_samples=None):
    """
    Process all JSON files under a task directory into a list of training samples.
    If num_samples is None, process all; otherwise take the first num_samples files.
    """
    task_folder = os.path.join(data_dir, "finished task", task_config["folder"])
    prompt_file = os.path.join(prompts_dir, task_config["prompt_file"])

    system_prompt = load_system_prompt(prompt_file)
    task_files = sorted(Path(task_folder).glob(task_config["file_pattern"]))

    if num_samples is not None and num_samples > 0:
        task_files = task_files[:num_samples]
        print(f"  - Taking first {num_samples} samples only")

    processed_data = []

    for idx, task_file in enumerate(task_files):
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)

            user_content = construct_user_content(task_data, task_config)
            solution = get_correct_answer(task_data, task_config)

            original_question = task_data.get("question", "")
            if isinstance(original_question, dict):
                original_question = json.dumps(original_question, ensure_ascii=False)

            # Video scripts etc. written to extra_info as simulator ground truth
            raw_videos_data = task_data.get("videos", {})
            if not isinstance(raw_videos_data, str):
                raw_videos_data = json.dumps(raw_videos_data, ensure_ascii=False)

            data_item = {
                "data_source": f"crossvid_{task_name}",
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "ability": "planning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "task_name": task_name,
                    "task_id": task_data.get("id", idx),
                    "index": idx,
                    "videos": raw_videos_data,
                    "original_question": original_question,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "answer": {"create_kwargs": {"ground_truth": solution}},
                        "get_caption": {"create_kwargs": {"dummy": ""}},
                        "observe": {"create_kwargs": {"dummy": ""}},
                    },
                },
            }

            processed_data.append(data_item)

        except Exception as e:
            print(f"Failed to process {task_file}: {e}")
            continue

    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./build_data", help="CrossVid raw data directory")
    parser.add_argument("--local_dir", default=None, help="Deprecated, use --local_save_dir")
    parser.add_argument("--local_save_dir", default="./format_data", help="Output parquet directory")
    parser.add_argument("--hdfs_dir", default=None, help="Optional: copy output to this HDFS path")
    parser.add_argument("--tasks", nargs="+", default=None, help="Task list to process (default: all)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per task")

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    prompts_dir = os.path.join(data_dir, "prompts")

    local_save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir
    if args.local_dir is not None:
        print("Warning: --local_dir is deprecated, use --local_save_dir.")
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    tasks_to_process = args.tasks if args.tasks else list(TASK_CONFIG.keys())
    print(f"Mini dataset: {args.num_samples} samples per task, {len(tasks_to_process)} tasks: {tasks_to_process}")

    all_data = []
    for task_name in tasks_to_process:
        if task_name not in TASK_CONFIG:
            print(f"Unknown task '{task_name}', skipping")
            continue
        print(f"Processing task: {task_name}")
        task_config = TASK_CONFIG[task_name]
        task_data = process_task_data(
            task_name,
            task_config,
            data_dir,
            prompts_dir,
            num_samples=args.num_samples
        )
        all_data.extend(task_data)
        print(f"  - Processed {len(task_data)} samples")

    print(f"\nTotal: {len(all_data)} samples")

    dataset = datasets.Dataset.from_list(all_data)
    output_file = os.path.join(local_save_dir, "crossvid_train_mini.parquet")
    dataset.to_parquet(output_file)
    print(f"Saved: {output_file}")

    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print("HDFS copy done")
