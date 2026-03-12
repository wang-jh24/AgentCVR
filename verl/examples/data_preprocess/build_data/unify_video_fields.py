#!/usr/bin/env python3
"""
Unify video script field names to 'videos' across task types (multi-threaded).

Features:
1. Multi-threaded (default 32 workers) for NAS/network storage IO.
2. No backup; overwrites in place.
3. Progress bar via tqdm.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Mapping from old field names to unified 'videos'
FIELD_MAPPING = {
    'grounding_synthesis': 'synthesized_videos',
    'plot_synthesis': 'scripts',
    'sort_synthesis': 'segments',
    'uav_count_synthesis': 'video_scripts',
    'uav_position_synthesis': 'video_scripts',
}

# Tasks that already use the correct field name
CORRECT_TASKS = [
    'assembly_synthesis',
    'behavior_synthesis',
    'cooking_synthesis',
    'movie_synthesis',
]

def process_json_file(file_path: Path, task_type: str, dry_run: bool) -> str:
    """Process a single JSON file; returns 'modified' / 'skipped' / 'error'."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        old_field = FIELD_MAPPING.get(task_type)

        if not old_field:
            return 'skipped'

        if old_field not in data:
            return 'skipped'

        if 'videos' in data:
            return 'skipped'

        if dry_run:
            return 'modified'

        data['videos'] = data.pop(old_field)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return 'modified'

    except Exception as e:
        print(f"\n[Error] {file_path.name}: {e}")
        return 'error'


def process_task_directory(task_dir: Path, dry_run: bool, max_workers: int) -> Dict[str, int]:
    """Process all JSON files under a task directory with multi-threading."""
    task_type = task_dir.name
    stats = {'total': 0, 'modified': 0, 'skipped': 0, 'error': 0}

    print(f"Processing task: {task_type}")

    if task_type in CORRECT_TASKS:
        print(f"  -> Skip (no change needed)")
        return stats

    if task_type not in FIELD_MAPPING:
        print(f"  -> Skip (unknown task type)")
        return stats

    json_files = list(task_dir.glob('*.json'))
    stats['total'] = len(json_files)

    if stats['total'] == 0:
        print("  -> No JSON files in directory")
        return stats

    print(f"  -> Running {max_workers} workers on {stats['total']} files...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_json_file, f, task_type, dry_run): f
            for f in json_files
        }
        for future in tqdm(as_completed(future_to_file), total=stats['total'], unit="file", ncols=100):
            result_status = future.result()
            stats[result_status] += 1

    print(f"  Stats: modified={stats['modified']}, skipped={stats['skipped']}, error={stats['error']}")
    print("-" * 60)

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Unify video script field name to videos (multi-threaded)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate run')
    parser.add_argument('--task', type=str, help='Specific task type to process')
    parser.add_argument('--workers', type=int, default=32, help='Number of threads (default 32)')
    parser.add_argument(
        '--base-dir',
        type=str,
        default='examples/data_preprocess/build_data/finished_task',
        help='Task data root (subdirs are task types)',
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: directory does not exist: {base_dir}")
        return 1

    print("=" * 60)
    print(f"Workers: {args.workers} | {'[dry run]' if args.dry_run else '[execute]'}")
    print("=" * 60)

    total_stats = {'total': 0, 'modified': 0, 'skipped': 0, 'error': 0}

    if args.task:
        task_dirs = [base_dir / args.task]
    else:
        task_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    for task_dir in task_dirs:
        stats = process_task_directory(task_dir, args.dry_run, args.workers)
        for k in total_stats:
            total_stats[k] += stats[k]

    print("\n" + "=" * 60)
    print(f"Total: {total_stats['total']} files")
    print(f"Modified: {total_stats['modified']} | Skipped: {total_stats['skipped']} | Error: {total_stats['error']}")
    print("=" * 60)
    return 0

if __name__ == '__main__':
    main()
