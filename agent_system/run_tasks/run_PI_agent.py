# PI task entry. Run from agent_system: python run_tasks/run_PI_agent.py

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
from typing import List, Dict, Any, Optional

from qwen_agent import QwenModel
from agent_executor import AgentExecutor
from utils import video_processor
from utils.config import get_local_video_root, get_remote_video_base_url
from utils.log_utils import setup_logger
from utils.answer_parse import extract_correct_list, parse_agent_output_multi

QUESTION_FILE = os.path.join(_ROOT, "question", "PI.json")
MAX_FRAME_LENGTH = 360
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "PI_task"
PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_PI.prompt")}

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
    correct_answers_count = 0

    for i, question_data in enumerate(questions_to_run):
        current_question_abs_num = i + start_question_num
        question_id = question_data.get('id', str(current_question_abs_num))
        log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(question_id))
        os.makedirs(log_dir, exist_ok=True)

        logger.info("==================================")
        logger.info(f"   Testing Question {current_question_abs_num}/{original_total_questions} (ID: {question_id})   ")
        query = "Given the Beginning and Ending clips of a movie scene, infer what most likely happened in the middle."
        possible_answers = question_data['options']
        logger.info(f"🎯 Query: {query}")
        logger.info(f"📋 Options: {possible_answers}")

        logger.info("📹 Preparing video metadata...")
        video_contexts = []
        v_name = question_data.get('video')
        begin_intervals = question_data.get('beginning', [])
        end_intervals = question_data.get('ending', [])
        if not v_name or not begin_intervals or not end_intervals:
            logger.error("❌ Missing 'video', 'beginning' or 'ending' data.")
            continue

        local_path = os.path.join(get_local_video_root(), "NC", f"{v_name}.mp4")
        remote_url = f"{get_remote_video_base_url().rstrip('/')}/NC/{v_name}.mp4"
        video_path = local_path if os.path.exists(local_path) else remote_url
        try:
            _, _, _, original_fps, total_f, total_d = video_processor.process_video(
                input_path=video_path, n_frames=1, intervals=[(0, 1)], max_length=MAX_FRAME_LENGTH, encode=False
            )
            b_start, b_end = float(begin_intervals[0]), float(begin_intervals[1])
            video_contexts.append({
                "path": video_path, "fps": original_fps, "total_frames": total_f,
                "duration_seconds": max(0.0, b_end - b_start), "clip_begin": b_start, "clip_end": b_end,
                "name_for_log": f"{v_name}_beginning"
            })
            logger.info(f"    ✅ Clip 1 (Beginning): [{b_start}s - {b_end}s]")
            e_start, e_end = float(end_intervals[0]), float(end_intervals[1])
            video_contexts.append({
                "path": video_path, "fps": original_fps, "total_frames": total_f,
                "duration_seconds": max(0.0, e_end - e_start), "clip_begin": e_start, "clip_end": e_end,
                "name_for_log": f"{v_name}_ending"
            })
            logger.info(f"    ✅ Clip 2 (Ending): [{e_start}s - {e_end}s]")
        except Exception as e:
            logger.error(f"    ❌ Error preparing metadata: {e}")
            video_contexts = []
        if not video_contexts:
            logger.error("❌ Skipping due to video failure.")
            continue

        try:
            executor.init_agent_state(query=query, possible_answers=possible_answers, video_contexts=video_contexts)
            final_answer_text, final_history = executor.run_agent_loop(max_turns=MAX_TURNS)
            logger.info("📊 Evaluating result...")
            predicted_list = parse_agent_output_multi(final_answer_text, len(possible_answers))
            correct_list = extract_correct_list(question_data.get('answer'), len(possible_answers))
            is_correct = (len(correct_list) > 0 and set(predicted_list) == set(correct_list))
            logger.info(f"  - Predicted: {predicted_list}")
            logger.info(f"  - Ground Truth: {correct_list}")
            if is_correct:
                logger.info("  - Result: 🎉 CORRECT! 🎉")
                correct_answers_count += 1
            else:
                logger.info("  - Result: 🔴 WRONG. 🔴")
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"❌ Loop Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()
            logger.info(f"--- End of Q{question_id} ---")

    logger.info("*******************")
    logger.info("   Final Summary   ")
    logger.info("*******************")
    accuracy = (correct_answers_count / num_questions_to_run) * 100 if num_questions_to_run > 0 else 0
    logger.info(f"✅ Questions Run:    {num_questions_to_run}")
    logger.info(f"🏆 Correct Answers:  {correct_answers_count}")
    logger.info(f"📊 Accuracy:         {accuracy:.2f}%")
