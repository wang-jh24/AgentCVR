# PEA task entry. Run from agent_system: python run_tasks/run_PEA_agent.py

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
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

from qwen_agent import QwenModel
from agent_executor import AgentExecutor
from utils import video_processor
from utils.config import get_local_video_root, get_remote_video_base_url
from utils.log_utils import setup_logger
from utils.answer_parse import extract_correct_list, parse_agent_output_multi

QUESTION_FILE = os.path.join(_ROOT, "question", "PEA.json")
MAX_FRAME_LENGTH = 360
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "PEA_task"
PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_PEA.prompt")}

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="PrepTask") as prep_executor:
        for i, question_data in enumerate(questions_to_run):
            current_question_abs_num = i + start_question_num
            question_id = question_data.get('id', str(current_question_abs_num))
            log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(question_id))
            os.makedirs(log_dir, exist_ok=True)

            logger.info("==================================")
            logger.info(f"   Testing Question {current_question_abs_num}/{original_total_questions} (ID: {question_id})   ")
            query = question_data['question']
            possible_answers = question_data['options']

            logger.info("📹 Preparing video metadata for the current question...")
            video_contexts = []
            videos = question_data.get('videos', [])
            begins = question_data.get('begin', [None] * len(videos))
            ends = question_data.get('end', [None] * len(videos))

            for idx, v_name in enumerate(videos):
                begin_sec = begins[idx] if idx < len(begins) else 0.0
                end_sec = ends[idx] if idx < len(ends) else None
                local_path = os.path.join(get_local_video_root(), "PEA", f"{v_name.replace('/', '_')}.mp4")
                remote_url = f"{get_remote_video_base_url().rstrip('/')}/PEA/{v_name}.mp4"
                video_path = local_path if os.path.exists(local_path) else remote_url
                try:
                    _, _, _, original_fps, total_f, total_d = video_processor.process_video(
                        input_path=video_path, n_frames=1, intervals=[(0, 1)], max_length=MAX_FRAME_LENGTH, encode=False
                    )
                    actual_end = end_sec if end_sec is not None else total_d
                    actual_begin = begin_sec if begin_sec is not None else 0.0
                    effective_duration = max(0.0, actual_end - actual_begin)
                    video_contexts.append({
                        "path": video_path, "fps": original_fps, "total_frames": total_f,
                        "duration_seconds": effective_duration, "original_duration": total_d,
                        "clip_begin": actual_begin, "clip_end": actual_end
                    })
                    logger.info(f"    ✅ Video {idx+1}: Clip [{actual_begin:.1f}-{actual_end:.1f}], Eff Dur={effective_duration:.1f}s")
                except Exception as e:
                    logger.error(f"    ❌ Error getting metadata for Video {idx+1}: {e}")
                    video_contexts = []
                    break

            if not video_contexts:
                logger.error("❌ Skipping due to video failure.")
                continue

            try:
                executor.init_agent_state(query=query, possible_answers=possible_answers, video_contexts=video_contexts)
            except Exception as e:
                logger.error(f"❌ Init failed: {e}")
                continue

            try:
                final_answer_text, final_history = executor.run_agent_loop(max_turns=MAX_TURNS)
                predicted_list = parse_agent_output_multi(final_answer_text, len(possible_answers))
                correct_list = extract_correct_list(question_data.get('answer'), len(possible_answers))
                is_correct = (len(correct_list) > 0 and set(predicted_list) == set(correct_list))
                logger.info(f"  - Pred: {predicted_list} | True: {correct_list} | {'🎉 CORRECT' if is_correct else '🔴 WRONG'}")
                if is_correct: correct_answers_count += 1
                executor.save_logs(log_dir)
            except Exception as e:
                logger.error(f"❌ Loop Error: {e}", exc_info=True)
            finally:
                executor.cleanup_agent_state()
                logger.info(f"--- End of Q{question_id} ---")

    logger.info("*******************")
    accuracy = (correct_answers_count / num_questions_to_run) * 100 if num_questions_to_run > 0 else 0
    logger.info(f"Tested: {num_questions_to_run}, Correct: {correct_answers_count}, Acc: {accuracy:.2f}%")
