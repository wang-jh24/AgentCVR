# FSA task entry. Run from agent_system: python run_tasks/run_FSA_agent.py

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import re
import logging
from datetime import datetime
import traceback
from typing import List, Dict, Any, Optional, Tuple

from qwen_agent import QwenModel
from agent_executor import AgentExecutor
from utils import video_processor
from utils.config import get_local_video_root, get_remote_video_base_url
from utils.log_utils import setup_logger
from utils.answer_parse import interval_iou, parse_json_interval_string

QUESTION_FILE = os.path.join(_ROOT, "question", "FSA.json")
MAX_FRAME_LENGTH = 360
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "FSA_task"
GENERIC_GROUNDING_QUERY = "Provide you two cooking videos, which step in Video 2 is functionally equivalent to the step shown between {BEGIN}s and {END}s in Video 1?"
PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_FSA.prompt")}

if __name__ == "__main__":
    if not os.path.exists(QUESTION_FILE):
        print(f"FATAL: Question file not found: {QUESTION_FILE}")
        raise SystemExit(1)
    with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    start_question_num = START_FROM_QUESTION_NUM
    if start_question_num < 1: start_question_num = 1
    start_index = start_question_num - 1
    end_question_num = END_AT_QUESTION_NUM
    original_total_questions = len(all_questions)
    if end_question_num is None: end_index = original_total_questions
    else: end_index = min(end_question_num, original_total_questions)
    questions_to_run = all_questions[start_index:end_index]
    num_questions_to_run = len(questions_to_run)

    logger = setup_logger(TASK_NAME.replace("_task", ""), start_question_num, end_index)
    logger.info(f"▶️  Run Config: Task={TASK_NAME}, Q{start_question_num}-Q{end_index}")

    qwen_agent_instance = QwenModel(prompt_config=PROMPT_CONFIG)
    executor = AgentExecutor(agent_instance=qwen_agent_instance, prompt_config=PROMPT_CONFIG)
    all_ious = []

    for i, question_data in enumerate(questions_to_run):
        current_question_abs_num = i + start_question_num
        question_id = question_data.get('id', str(current_question_abs_num))
        log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(question_id))
        os.makedirs(log_dir, exist_ok=True)

        logger.info("==================================")
        logger.info(f"   Testing Question {current_question_abs_num}/{original_total_questions} (ID: {question_id})   ")
        video_name_A = question_data.get('video A')
        video_name_B = question_data.get('video B')
        ref_segment = question_data.get('ref_segment')
        correct_answer_interval = question_data.get('answer')
        if not all([video_name_A, video_name_B, ref_segment, correct_answer_interval]):
            logger.error(f"❌ Skipping question {question_id} due to missing data.")
            continue

        generic_query_text = GENERIC_GROUNDING_QUERY.format(BEGIN=ref_segment[0], END=ref_segment[1])
        logger.info("📹 Preparing video metadata...")
        video_contexts = []
        video_names = [video_name_A, video_name_B]
        for v_idx, v_name in enumerate(video_names):
            local_path = os.path.join(get_local_video_root(), "CC", f"{v_name}.mp4")
            remote_url = f"{get_remote_video_base_url().rstrip('/')}/CC/{v_name}.mp4"
            video_path = local_path if os.path.exists(local_path) else remote_url
            try:
                _, _, _, original_fps, total_f, total_d = video_processor.process_video(
                    input_path=video_path, n_frames=1, intervals=[(0, 1)], max_length=MAX_FRAME_LENGTH, encode=False
                )
                video_contexts.append({
                    "path": video_path, "fps": original_fps, "total_frames": total_f,
                    "duration_seconds": total_d, "name_for_log": v_name
                })
                logger.info(f"    ✅ Video {v_idx+1} ({v_name}): Dur={total_d:.1f}s, FPS={original_fps:.2f}")
            except Exception as e:
                logger.error(f"    ❌ Error getting metadata for {v_name}: {e}")
                video_contexts = []
                break
        if not video_contexts:
            logger.error("❌ Skipping due to video failure.")
            continue

        query_for_agent = (
            f"{generic_query_text}\n"
            f"Reference Segment (Video 1): {ref_segment}s\n"
            f"Video 1 Name: {video_name_A}\n"
            f"Video 2 Name: {video_name_B}"
        )
        try:
            executor.init_agent_state(query=query_for_agent, possible_answers=[], video_contexts=video_contexts)
            final_answer_text, final_history = executor.run_agent_loop(max_turns=MAX_TURNS)
            logger.info("📊 Evaluating result...")
            predicted_interval = parse_json_interval_string(final_answer_text)
            if not predicted_interval:
                nums = re.findall(r"([\d]+\.?[\d]*)", final_answer_text)
                if len(nums) >= 2:
                    predicted_interval = (float(nums[0]), float(nums[1]))
                    logger.warning(f"  ⚠️ Parsed interval using fallback regex: {predicted_interval}")
            gt_interval = (float(correct_answer_interval[0]), float(correct_answer_interval[1]))
            current_iou = 0.0
            if predicted_interval:
                if predicted_interval[0] > predicted_interval[1]:
                    predicted_interval = (predicted_interval[1], predicted_interval[0])
                current_iou = interval_iou(predicted_interval, gt_interval)
                logger.info(f"  - Predicted: {predicted_interval}")
                logger.info(f"  - Ground Truth: {gt_interval}")
                logger.info(f"  - Result: 📈 IOU = {current_iou:.4f}")
            else:
                logger.warning(f"  - 🔴 WRONG. Failed to parse interval from: {final_answer_text[:100]}...")
            all_ious.append(current_iou)
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"❌ Loop Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()
            logger.info(f"--- End of Q{question_id} ---")

    logger.info("*******************")
    logger.info("   Final Summary   ")
    logger.info("*******************")
    mean_iou = (sum(all_ious) / len(all_ious)) if all_ious else 0
    logger.info(f"✅ Tested: {num_questions_to_run}")
    logger.info(f"📈 Mean IOU: {mean_iou:.4f}")
