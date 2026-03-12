import json
import logging
import os

logger = logging.getLogger(__file__)


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, **kwargs):
    """
    Evaluate if the model generated a valid JSON tool call for 'answer'
    and if the content matches the ground truth.

    Expected JSON format from agent:
    {"action": "answer", "thought": "...", "params": {"final_answer": "A"}}
    """
    # Optional debug log: set CROSSVID_DEBUG_LOG to a file path to enable
    log_path = os.environ.get("CROSSVID_DEBUG_LOG")
    if log_path:
        try:
            import time
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "model_response": solution_str,
                "ground_truth": ground_truth,
                "extra_info": str(kwargs),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to write debug log: %s", e)

    # Preprocess: extract JSON objects from text
    clean_str = solution_str.strip()
    
    # =======================================================
    # Use brace-balanced extraction for all JSON objects (no greedy regex)
    # =======================================================
    def extract_json_objects(text):
        json_objects = []
        brace_stack = []
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    start_idx = i
                brace_stack.append(char)
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack:
                        # Found a balanced top-level JSON string
                        json_str = text[start_idx:i+1]
                        json_objects.append(json_str)
        return json_objects

    # Extract all potential JSON objects from the string
    candidates = extract_json_objects(clean_str)
    
    if not candidates:
        return 0.0

    # =======================================================
    # Iterate over extracted JSON objects to find the final answer
    # =======================================================
    final_submission = None
    
    # Iterate in reverse; answer tool call is usually last
    for json_str in reversed(candidates):
        try:
            data = json.loads(json_str)
            
            # Must contain action field
            if "action" not in data:
                continue

            # Only care about 'answer' action
            if data["action"] == "answer":
                # Extract final_answer: 1) prefer params
                params = data.get("params", {})
                if isinstance(params, dict):
                    final_submission = params.get("final_answer")

                # 2) fallback to root
                if final_submission is None:
                    final_submission = data.get("final_answer")

                if final_submission is not None:
                    break
        except json.JSONDecodeError:
            continue

    if final_submission is None:
        return 0.0  # No valid answer tool call found

    # =======================================================
    # Compare with ground truth
    # =======================================================
    pred = str(final_submission).strip().upper()
    truth = str(ground_truth).strip().upper()
    
    if pred == truth:
        return score
    else:
        return 0.0