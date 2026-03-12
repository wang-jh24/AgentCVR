# CCQA task entry. Run from agent_system: python run_tasks/run_CCQA_agent.py

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import re
import logging
import time
import requests
from typing import List, Dict, Any, Optional

from qwen_agent import QwenModel
from agent_executor import AgentExecutor
from utils import video_processor
from utils.config import get_local_video_root, get_remote_video_base_url
from utils.log_utils import setup_logger
from utils.answer_parse import extract_answer_text

QUESTION_FILE = os.path.join(_ROOT, "question", "CCQA.json")
MAX_FRAME_LENGTH = 360
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "CCQA_task"
SCORING_PROMPT_FILE = os.path.join(_ROOT, "prompts", "scoring.prompt")
PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_CCQA.prompt")}

SCORING_API_CONFIG = {
    "base_url": os.environ.get("SCORING_API_BASE_URL", "https://your-scoring-api.com/v1:generateContent"),
    "api_key": os.environ.get("SCORING_API_KEY", "your-scoring-api-key"),
    "model_name": os.environ.get("SCORING_MODEL_NAME", "your-scoring-model"),
    "max_tokens": int(os.environ.get("SCORING_MAX_TOKENS", "16384")),
    "temperature": 0.0,
}

def call_scoring_model(logger: logging.Logger, prompt_text: str) -> str:
    headers = {"api-key": SCORING_API_CONFIG['api_key'], "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": SCORING_API_CONFIG['temperature'],
            "maxOutputTokens": SCORING_API_CONFIG['max_tokens'],
            "topP": 1, "seed": 0
        }
    }
    try:
        response = requests.post(SCORING_API_CONFIG['base_url'], headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return content if content else "[Scoring API returned empty content]"
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"  [_call_scoring_api] Failed to parse Gemini response: {e}")
        return f"Error: Failed to parse Gemini response structure."
    except requests.exceptions.Timeout:
        logger.error("  [_call_scoring_api] Scoring API request timed out after 120 seconds.")
        return "Error: Scoring API request timed out."
    except requests.exceptions.RequestException as e:
        logger.error(f"  [_call_scoring_api] Scoring API request failed: {e}")
        error_details = "[Could not get response body]"
        if e.response is not None:
            try: error_details = e.response.text
            except Exception: pass
        return f"Error: Scoring API request failed: {e}. Response body: {error_details}"
    except Exception as e:
        logger.error(f"  [_call_scoring_api] An unexpected error occurred during scoring: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during scoring: {e}"

def extract_json_from_score(text: str) -> Optional[Dict]:
    match = re.search(r'<score>\s*(\{.*?\})\s*</score>', text, re.DOTALL)
    if not match:
        match = re.search(r'(\{.*?\})', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

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

    logger.info(f"📄 Loading scoring prompt from '{SCORING_PROMPT_FILE}'...")
    try:
        with open(SCORING_PROMPT_FILE, 'r', encoding='utf-8') as f:
            scoring_prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"FATAL: Scoring prompt file not found: {SCORING_PROMPT_FILE}")
        raise SystemExit(1)

    total_score = 0
    total_max_score = 0
    scored_items_count = 0

    for i, question_data in enumerate(questions_to_run):
        current_question_abs_num = i + start_question_num
        question_id = question_data.get('id', str(current_question_abs_num))
        log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(question_id))
        os.makedirs(log_dir, exist_ok=True)

        logger.info("==================================")
        logger.info(f"   Testing Question {current_question_abs_num}/{original_total_questions} (ID: {question_id})   ")
        query = question_data['question']

        logger.info("📹 Preparing video metadata for the current question...")
        video_contexts = []
        video_keys_to_process = ["video A", "video B"]
        for v_idx, v_key in enumerate(video_keys_to_process):
            if v_key not in question_data:
                logger.error(f"❌ Missing key '{v_key}' in question.")
                continue
            v_name = question_data[v_key]
            local_path = os.path.join(get_local_video_root(), "CC", f"{v_name}.mp4")
            remote_url = f"{get_remote_video_base_url().rstrip('/')}/CC/{v_name}.mp4"
            video_path = local_path if os.path.exists(local_path) else remote_url
            try:
                _, _, _, original_fps, total_f, total_d = video_processor.process_video(
                    input_path=video_path, n_frames=1, intervals=[(0, 1)], max_length=MAX_FRAME_LENGTH, encode=False
                )
                video_contexts.append({
                    "path": video_path, "fps": original_fps, "total_frames": total_f, "duration_seconds": total_d,
                })
                logger.info(f"    ✅ {v_key}: Duration={total_d:.1f}s, FPS={original_fps:.2f}")
            except Exception as e:
                logger.error(f"    ❌ Error getting metadata for {v_key}: {e}")
                video_contexts = []
                break
        if not video_contexts:
            logger.error("❌ Skipping due to video failure.")
            continue

        try:
            executor.init_agent_state(query=query, possible_answers=[], video_contexts=video_contexts)
        except Exception as e:
            logger.error(f"❌ Init failed: {e}")
            continue

        try:
            final_answer_text, final_history = executor.run_agent_loop(max_turns=MAX_TURNS)
            logger.info("📊 Evaluating result...")
            standard_answer = question_data.get('answer', 'N/A')
            scoring_points = question_data.get('scoring_points', [])
            answer_to_score = extract_answer_text(final_answer_text)
            logger.info(f"  - Answer for scoring: {answer_to_score[:100]}...")

            max_scoring_retries = 3
            score_json = None
            for scoring_attempt in range(max_scoring_retries):
                logger.info(f"⚖️ Calling Scoring Model (Attempt {scoring_attempt + 1}/{max_scoring_retries})...")
                score_prompt = scoring_prompt_template.format(
                    QUESTION=query, ANSWER=standard_answer, POINTS=scoring_points, OUTPUT=answer_to_score
                )
                score_response = call_scoring_model(logger, score_prompt)

                if not score_response.startswith("Error:"):
                    score_json = extract_json_from_score(score_response)
                    if score_json and isinstance(score_json.get("coverage"), list) and isinstance(score_json.get("correctness"), list):
                        break
                    logger.warning(f"  - ⚠️ Attempt {scoring_attempt + 1}: Could not parse score JSON.")
                else:
                    logger.warning(f"  - ⚠️ Attempt {scoring_attempt + 1}: Scoring API error: {score_response[:200]}...")
                    if "content_filter" in score_response or "ResponsibleAIPolicyViolation" in score_response:
                        logger.info("  - 🔄 Content filter detected. Asking master to rephrase answer...")
                        rephrased_answer = executor.request_answer_rephrase(
                            current_answer=answer_to_score,
                            error_reason="Content filter: The answer triggered Azure OpenAI's content policy. Please rephrase your answer to avoid sensitive words (e.g., replace 'breast' with 'chicken piece', 'moist' with 'tender')."
                        )
                        if rephrased_answer and rephrased_answer != answer_to_score:
                            answer_to_score = rephrased_answer
                            logger.info(f"  - 📝 Master rephrased answer: {answer_to_score[:100]}...")
                            continue
                        logger.warning("  - ⚠️ Master failed to provide rephrased answer.")
                    if scoring_attempt < max_scoring_retries - 1:
                        time.sleep(2)
                        continue

            if score_json and isinstance(score_json.get("coverage"), list) and isinstance(score_json.get("correctness"), list):
                coverage = score_json["coverage"]
                correctness = score_json["correctness"]
                question_score = sum(c for c in coverage if c is True) + sum(c for c in correctness if c is True)
                question_max_score = len(coverage) + len(correctness)
                logger.info(f"  - ✅ Score: {question_score}/{question_max_score} | Coverage: {coverage} | Correctness: {correctness}")
                total_score += question_score
                total_max_score += question_max_score
                scored_items_count += 1
            else:
                logger.error(f"  - ❌ Failed to score after {max_scoring_retries} attempts.")
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"❌ Loop Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()
            logger.info(f"--- End of Q{question_id} ---")

    logger.info("*******************")
    logger.info("📊 Open任务评分统计结果")
    logger.info("*******************")
    accuracy = (total_score / total_max_score) if total_max_score > 0 else 0
    logger.info(f"✅ 完成的题目数量:  {scored_items_count}/{num_questions_to_run}")
    logger.info(f"📈 总得分:          {total_score}")
    logger.info(f"🎯 总得分点(满分):  {total_max_score}")
    logger.info(f"📊 准确率:          {accuracy:.4f} ({accuracy*100:.2f}%)")
