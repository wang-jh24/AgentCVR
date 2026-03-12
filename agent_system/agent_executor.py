# agent_executor.py
# Orchestrates the agent loop: master dialogue, active_perception (video/image), get_caption (Whisper).
# Supports standard video and UAV image-folder contexts.

import re
import logging
import time
import json
import base64
import os
import copy
from typing import Dict, Callable, List, Any, Optional, Tuple, Union
from datetime import datetime

import cv2
import numpy as np

from utils import video_processor
from qwen_agent import QwenModel
from utils.caption_generator import generate_caption_for_segment
from utils.sanitize import safe_truncate, sanitize_for_log
from utils.text_utils import clean_thought_content

logger = logging.getLogger("AgentWorkflowLogger")

MAX_FRAME_LENGTH = 360


class AgentExecutor:
    """Runs ReAct loop: master LLM + tools (active_perception, get_caption)."""

    def __init__(self, agent_instance: QwenModel, prompt_config: Optional[Dict[str, str]] = None, tool_frames_per_clip: int = 128):
        self.agent = agent_instance
        self.tool_frames_per_clip = tool_frames_per_clip
        self.video_contexts: List[Dict[str, Any]] = []
        self.tool_mapping = self._create_tool_mapping()

        default_prompts = {"master": "prompts/master_autonomous.prompt"}
        self.prompt_config = default_prompts
        if prompt_config: self.prompt_config.update(prompt_config)
        
        master_prompt_path = self.prompt_config.get("master")
        if not master_prompt_path: raise ValueError("Prompt path for 'master' not configured.")
        with open(master_prompt_path, 'r', encoding='utf-8') as f:
            self.master_prompt_template = f.read()

        self._possible_answers: List[str] = []
        self._initial_query: str = ""
        self._master_history: List[Dict[str, Any]] = []
        self._tool_history: List[Dict[str, Any]] = [] 
        self.current_turn = 0

    def _load_prompt_template(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"FATAL: Master Prompt not found at {path}.")
            raise

    def _create_tool_mapping(self) -> Dict[str, Callable]:
        """Maps tool names to handlers: active_perception, get_caption."""
        return {
            "active_perception": self._proxy_active_perception,
            "get_caption": self._proxy_get_caption,
        }

    def save_logs(self, folder_path: str):
        """Persist master dialogue and tool execution to JSON and readable TXT (Base64 stripped)."""
        try:
            sanitized_master_history = sanitize_for_log(self._master_history)
            sanitized_tool_history = sanitize_for_log(self._tool_history)

            master_log_path = os.path.join(folder_path, "master_dialogue.json")
            with open(master_log_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_master_history, f, ensure_ascii=False, indent=2)
            
            tool_log_path = os.path.join(folder_path, "tools_execution.json")
            with open(tool_log_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_tool_history, f, ensure_ascii=False, indent=2)

            readable_master_path = os.path.join(folder_path, "master_dialogue_readable.txt")
            with open(readable_master_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Master Dialogue Log (Readable) ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, msg in enumerate(sanitized_master_history):
                    role = msg.get("role", "UNKNOWN").upper()
                    content = msg.get("content", "")
                    
                    f.write(f"--- [Step {i}] {role} ---\n")
                    
                    if isinstance(content, (dict, list)):
                        f.write(json.dumps(content, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(content))
                    
                    f.write("\n\n" + "="*60 + "\n\n")

            readable_tool_path = os.path.join(folder_path, "tools_execution_readable.txt")
            with open(readable_tool_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Tools Execution Log (Readable) ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for entry in sanitized_tool_history:
                    turn = entry.get("turn", "?")
                    tool = entry.get("tool_name", "unknown")
                    call_id = entry.get("call_id", "unknown")
                    
                    f.write(f"### [Turn {turn}] Tool: {tool} (ID: {call_id}) ###\n")
                    f.write(f"Time: {entry.get('timestamp_start')} -> {entry.get('timestamp_end')}\n")
                    
                    f.write("\n[Inputs]:\n")
                    inputs = entry.get("inputs")
                    if isinstance(inputs, (dict, list)):
                        f.write(json.dumps(inputs, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(inputs))
                        
                    f.write("\n\n[Output]:\n")
                    output = entry.get("output_raw")
                    if isinstance(output, (dict, list)):
                        f.write(json.dumps(output, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(output))
                        
                    f.write("\n\n" + "-"*60 + "\n\n")

            logger.info(f"    💾 Logs saved to: {folder_path} (JSON + Readable TXT)")
            
        except Exception as e:
            logger.error(f"    ❌ Failed to save logs: {e}", exc_info=True)


    def _proxy_active_perception(self, observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        """Routes to UAV (image_folder) or standard (video) perception handler."""
        if not observation_targets: return {"result": "Error: No observation targets provided."}

        first_idx = observation_targets[0].get("video_index")
        if first_idx is None: return {"result": "Error: Missing video_index."}
        
        real_idx = first_idx - 1
        if real_idx < 0 or real_idx >= len(self.video_contexts):
            return {"result": f"Error: Video index {first_idx} out of range."}
        
        context = self.video_contexts[real_idx]
        ctx_type = context.get("type", "video_file")

        if ctx_type == "image_folder":
            logger.info("    [EXECUTOR] 🔀 Dispatching to UAV Handler (will use Tool API)")
            return self._active_perception_uav(observation_targets, focus_prompt)
        else:
            logger.info("    [EXECUTOR] 🔀 Dispatching to Standard Handler (will use Tool API)")
            return self._active_perception_standard(observation_targets, focus_prompt)

    def _active_perception_uav(self, observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        """UAV path: frame-index based, draws bboxes, uses image_files list for lookup."""
        logger.info(f"    [TOOL-UAV] 🚁 Processing {len(observation_targets)} targets (Frame-based)")
        
        all_frames = []
        frame_mapping_text_parts = []
        
        try:
            for target in observation_targets:
                v_idx = target.get("video_index")
                if v_idx is None: return {"result": "Error: Missing video_index"}

                real_idx = v_idx - 1
                if real_idx < 0 or real_idx >= len(self.video_contexts):
                     return {"result": f"Error: Video index {v_idx} out of range."}

                context = self.video_contexts[real_idx]
                
                folder_path = context.get("path")
                bbox_lookup = context.get("bbox_data", {})
                obj_map = context.get("obj_map", {})
                total_f = context.get("total_frames", 0)
                image_files_list = context.get("image_files", [])

                start_f = target.get("start_frame")
                end_f = target.get("end_frame")
                n_frames = target.get("num_frames", 16)
                
                if start_f is None:
                    return {"result": f"Error: For UAV tasks, please use 'start_frame' and 'end_frame' parameters."}
                start_f = max(0, int(start_f))
                end_f = min(total_f - 1, int(end_f))
                
                if start_f > end_f:
                    return {"result": f"Error: start_frame {start_f} is greater than end_frame {end_f}"}

                indices = np.linspace(start_f, end_f, n_frames).astype(int)
                indices = sorted(list(set(indices)))
                
                extracted_count = 0
                for idx in indices:
                    if idx >= len(image_files_list):
                        logger.warning(f"Frame index {idx} out of bounds (Total: {len(image_files_list)})")
                        continue

                    file_name = image_files_list[idx] 
                    full_path = os.path.join(folder_path, file_name)
                    
                    if not os.path.exists(full_path):
                        continue
                    img = cv2.imread(full_path)
                    if img is None: continue
                    
                    if idx in bbox_lookup:
                        objs_in_frame = bbox_lookup[idx]
                        for tag, box in objs_in_frame.items():
                            display_name = obj_map.get(tag, tag)
                            xtl, ytl, xbr, ybr = map(int, box)
                            cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                            (w, h), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(img, (xtl, ytl - 20), (xtl + w, ytl), (0, 255, 0), -1)
                            cv2.putText(img, display_name, (xtl, ytl - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    _, buffer = cv2.imencode('.jpg', img)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    all_frames.append(b64_str)
                    extracted_count += 1
                
                mapping_desc = f"Frames {start_f}-{end_f} from Video {v_idx} (sampled {extracted_count} images)"
                frame_mapping_text_parts.append(mapping_desc)
                
            return self._call_vlm_for_perception(all_frames, frame_mapping_text_parts, focus_prompt)

        except Exception as e:
            logger.error(f"    [TOOL-UAV] ❌ Failed: {e}", exc_info=True)
            return {"result": f"Error during UAV observation: {e}"}

    def _active_perception_standard(self, observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        """Standard video path: time-based segments via video_processor, no bbox."""
        logger.info(f"    [TOOL-STD] 🎬 Processing {len(observation_targets)} targets (Time-based)")
        
        all_frames = []
        frame_mapping_text_parts = []
        current_frame_index = 0
        
        try:
            for target in observation_targets:
                v_idx = target.get("video_index")
                if v_idx is None: return {"result": "Error: Missing video_index"}
                
                real_idx = v_idx - 1
                if real_idx < 0 or real_idx >= len(self.video_contexts):
                    return {"result": f"Error: Video index {v_idx} out of range."}
                
                context = self.video_contexts[real_idx]
                video_path = context.get("path")
                clip_begin_offset = context.get("clip_begin", 0.0)
                duration = context.get("duration_seconds", 0)

                start_t = target.get("start_time", 0)
                end_t = target.get("end_time", duration)
                num_f = target.get("num_frames", 32)
                
                abs_start = start_t + clip_begin_offset
                abs_end = end_t + clip_begin_offset
                frame_groups, _, _, _, _, _ = video_processor.process_video(
                    input_path=video_path,
                    n_frames=num_f,
                    intervals=[(abs_start, abs_end)],
                    max_length=MAX_FRAME_LENGTH,
                    encode=True
                )
                
                if frame_groups and frame_groups[0]:
                    frames = frame_groups[0]
                    all_frames.extend(frames)
                    
                    s_idx = current_frame_index + 1
                    e_idx = current_frame_index + len(frames)
                    mapping_line = f"Frames {s_idx}-{e_idx} are from Video {v_idx} ({start_t:.1f}s to {end_t:.1f}s)"
                    frame_mapping_text_parts.append(mapping_line)
                    current_frame_index += len(frames)
            
            return self._call_vlm_for_perception(all_frames, frame_mapping_text_parts, focus_prompt)

        except Exception as e:
            logger.error(f"    [TOOL-STD] ❌ Failed: {e}", exc_info=True)
            return {"result": f"Error during video observation: {e}"}

    def _call_vlm_for_perception(self, frames: List[str], mapping_texts: List[str], prompt: str) -> Dict[str, Any]:
        """Calls tool VLM with frames and mapping info; returns observation text."""
        if not frames:
            return {"result": "Error: No frames extracted."}

        frame_mapping_text = "\n".join(mapping_texts)
        full_visual_prompt = f"""[Frame Mapping Information]
{frame_mapping_text}

[Visual Question]
{prompt}

Please carefully observe the frames and answer the question."""

        logger.info(f"    [TOOL] 👁️ Calling VLM with {len(frames)} frames...")
        
        vlm_response = self.agent.forward(
            task='active_perception',
            frames=frames,
            prompt=full_visual_prompt,
            return_prompt=True
        )
        
        obs_text = vlm_response.get("result", "")
        obs_text = self._strip_think_from_result(obs_text)
        
        return {"result": obs_text, "prompt": vlm_response.get("prompt")}

    def _proxy_get_caption(self, video_index: int, start_time: float = None, end_time: float = None) -> Dict[str, Any]:
        """Generates captions for the given video segment via Whisper (on-demand)."""
        logger.info(f"    [TOOL] 📝 Get Caption (on-demand Whisper): Video {video_index}, Time: {start_time}-{end_time}")
        
        try:
            if video_index is None:
                return {"result": "Error: video_index is required."}
            
            real_video_index = video_index - 1
            if real_video_index < 0 or real_video_index >= len(self.video_contexts):
                return {"result": f"Error: Video index {video_index} out of range."}
            
            context = self.video_contexts[real_video_index]
            ctx_type = context.get("type", "video_file")
            
            if ctx_type == "image_folder":
                return {"result": "Error: Caption generation is not available for image folder input (no video file)."}
            
            video_path = context.get("path")
            clip_begin = context.get("clip_begin", 0.0)
            duration_seconds = context.get("duration_seconds", 0.0)
            
            if not video_path:
                return {"result": f"Error: No video path for Video {video_index}."}
            
            abs_start = clip_begin + (start_time if start_time is not None else 0.0)
            abs_end = clip_begin + (end_time if end_time is not None else duration_seconds)
            abs_end = min(abs_end, clip_begin + duration_seconds)
            
            if abs_start >= abs_end:
                return {"result": f"Error: Invalid time range for Video {video_index}."}
            
            caption_data = generate_caption_for_segment(video_path, abs_start, abs_end)
            
            if not caption_data:
                return {"result": f"No speech detected in Video {video_index} between {abs_start:.1f}s and {abs_end:.1f}s (or generation failed)."}
            
            result_text = f"Caption for Video {video_index} ({abs_start:.1f}s - {abs_end:.1f}s):\n"
            for seg in caption_data:
                result_text += f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"
            
            return {"result": result_text}
            
        except Exception as e:
            logger.error(f"    [TOOL] ❌ Get caption failed: {e}", exc_info=True)
            return {"result": f"Error: {e}"}
    
    @staticmethod
    def _strip_think_from_result(result_str: str) -> str:
        if not isinstance(result_str, str): return result_str
        if '</think>' in result_str:
            parts = result_str.rsplit('</think>', 1)
            if len(parts) == 2: return parts[1].strip()
        return result_str

    def _execute_tool_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Runs one tool call and appends result to tool history."""
        tool_name = call.get("tool_name")
        tool_args = call.get("arguments", {}) or {}
        call_id = call.get("id", f"c_{datetime.now().microsecond}")

        tool_log_entry = {
            "turn": self.current_turn,
            "call_id": call_id,
            "tool_name": tool_name,
            "inputs": tool_args,
            "timestamp_start": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if tool_name not in self.tool_mapping:
            error_msg = f"Tool '{tool_name}' does not exist."
            tool_log_entry["error"] = error_msg
            self._tool_history.append(tool_log_entry)
            return {"call_id": call_id, "tool_name": tool_name, "error": error_msg}

        try:
            tool_function = self.tool_mapping[tool_name]
            tool_result_obj = tool_function(**tool_args)

            if isinstance(tool_result_obj, dict) and "result" in tool_result_obj:
                result_payload = tool_result_obj.get("result")
                call_prompt = tool_result_obj.get("prompt")
            else:
                result_payload = tool_result_obj
                call_prompt = getattr(self.agent, "last_tool_prompt", None)

            tool_log_entry["actual_prompt_sent"] = call_prompt
            tool_log_entry["output_raw"] = result_payload
            tool_log_entry["timestamp_end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._tool_history.append(tool_log_entry)

            if isinstance(result_payload, str):
                result_str = result_payload
            else:
                result_str = json.dumps(result_payload, ensure_ascii=False)

            return {
                "call_id": call_id,
                "tool_name": tool_name,
                "result": result_str
            }

        except Exception as e:
            logger.error(f"    [EXECUTOR] Tool execution failed ({tool_name}): {e}", exc_info=True)
            tool_log_entry["error"] = str(e)
            self._tool_history.append(tool_log_entry)
            return {
                "call_id": call_id,
                "tool_name": tool_name,
                "error": f"{e}"
            }

    def _parse_llm_response(self, response_str: str) -> Dict[str, Any]:
        """Parses master LLM response into JSON (handles markdown/think tags)."""
        response_str = response_str.strip()
        try:
            data = json.loads(response_str)
            if 'action' in data: return data
        except json.JSONDecodeError: pass

        match_md = re.search(r'```(?:json)?\s*(.*?)\s*```', response_str, re.DOTALL | re.IGNORECASE)
        if match_md:
            try:
                data = json.loads(match_md.group(1).strip())
                if 'action' in data: return data
            except: pass

        try:
            parts = response_str.rsplit('</think>', 1)
            if len(parts) == 2:
                json_part = parts[1].strip()
                match_md_2 = re.search(r'```(?:json)?\s*(.*?)\s*```', json_part, re.DOTALL | re.IGNORECASE)
                if match_md_2: json_part = match_md_2.group(1).strip()
                idx = json_part.find('{')
                if idx != -1:
                    data = json.loads(json_part[idx:])
                    if 'action' in data: return data
        except: pass
        
        try:
            p1 = response_str.find('{')
            p2 = response_str.rfind('}')
            if p1 != -1 and p2 != -1:
                data = json.loads(response_str[p1:p2+1])
                if 'action' in data: return data
        except: pass

        raise ValueError(f"LLM response is not valid JSON.")

    def init_agent_state(self, query: str, possible_answers: List[str], video_contexts: List[Dict[str, Any]], **kwargs):
        """Reset state for a new question; kwargs e.g. formatted_query, bbox_info for UAV."""
        self.video_contexts = video_contexts
        self._possible_answers = possible_answers
        self._initial_query = query
        self._master_history = []
        self._tool_history = []
        self.current_turn = 0

    def cleanup_agent_state(self):
        self.video_contexts = []
        self._master_history = []
        self._tool_history = []
        self._possible_answers = []
        self._initial_query = ""
        self.current_turn = 0

    def run_agent_loop(self, max_turns=100) -> Tuple[str, List[Dict[str, Any]]]:
        """Runs ReAct loop until answer or max_turns."""
        logger.info("🚀 Starting Master Agent Loop...")
        final_answer = "Error: Loop ended unexpectedly."
        
        try:
            video_info_list = []
            for i, ctx in enumerate(self.video_contexts):
                info = {"video_index": i + 1}
                if ctx.get("type") == "image_folder":
                    info["total_frames"] = ctx.get("total_frames")
                else:
                    info["duration_seconds"] = ctx.get("duration_seconds", 0)
                    
                video_info_list.append(info)
            
            system_content = self.master_prompt_template
            messages = [{"role": "system", "content": system_content}]
            
            user_payload = {
                "query": self._initial_query,
                "videos": video_info_list
            }
            if self._possible_answers:
                user_payload["possible_answers"] = self._possible_answers
            
            user_content_items = [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]
            messages.append({"role": "user", "content": user_content_items})
            
            self._master_history = list(messages)
            
            for turn in range(1, max_turns + 1):
                self.current_turn = turn
                logger.info(f"🔄 [TURN {turn}] Master Thinking...")
                
                max_retries = 3
                valid_response_dict = None
                raw_response_str_for_history = None
                last_failed_response = None

                for attempt in range(max_retries):
                    messages_to_send = list(self._master_history)
                    if attempt > 0:
                        logger.info(f"  Attempt {attempt+1}: Sending format reminder...")
                        gentle_reminder = {
                            "role": "user",
                            "content": "Please ensure your reply is valid JSON and output only one JSON object per turn.\n"
                        }
                        messages_to_send.append(gentle_reminder)

                    try:
                        llm_response_str = self.agent.forward(task='master', messages=messages_to_send)
                        last_failed_response = llm_response_str
                    except Exception as e:
                        logger.error(f"  ❌ API Network/Server Error on attempt {attempt+1}: {e}")
                        time.sleep(2)
                        continue
                    
                    try:
                        parsed_response = self._parse_llm_response(llm_response_str)
                        if "action" not in parsed_response:
                            raise ValueError("JSON missing 'action' field")

                        valid_response_dict = parsed_response
                        raw_response_str_for_history = llm_response_str
                        break 
                    except Exception as e:
                        logger.warning(f"  ⚠️ Format issue on attempt {attempt+1}. Retrying cleanly... ({e})")
                        continue

                if not valid_response_dict:
                    logger.error(f"❌ Max retries reached on Turn {turn}. Moving to termination.")
                    if last_failed_response:
                        self._master_history.append({
                            "role": "assistant", 
                            "content": f"[FAILED RESPONSE - Invalid JSON]\n{last_failed_response}"
                        })
                    break 

                cleaned_response = clean_thought_content(raw_response_str_for_history)
                self._master_history.append({"role": "assistant", "content": cleaned_response})
                
                action = valid_response_dict.get("action")
                thought = valid_response_dict.get("thought", "N/A")
                logger.info(f"  🤖 Thought: {thought[:100]}...")
                logger.info(f"  🤖 Action:  [{action.upper()}]")

                if action == "answer":
                    final_answer = valid_response_dict.get("final_answer", "No answer provided")
                    logger.info(f"🎯 Final Answer: {final_answer}")
                    break
                
                elif action == "observe":
                    params = valid_response_dict.get("params", {})
                    targets = params.get("observation_targets", [])
                    prompt = params.get("focus_prompt", "")
                    
                    if not targets:
                        err = "Missing observation_targets"
                        self._master_history.append({"role": "tool", "content": [{"error": err}]})
                        continue 
                        
                    tool_call = {
                        "id": f"obs_{turn}",
                        "tool_name": "active_perception",
                        "arguments": {"observation_targets": targets, "focus_prompt": prompt}
                    }
                    obs_result = self._execute_tool_call(tool_call)
                    self._master_history.append({
                        "role": "tool", 
                        "content": [{"type": "text", "text": json.dumps(obs_result, ensure_ascii=False)}]
                    })
                
                elif action == "get_caption":
                    params = valid_response_dict.get("params", {})
                    tool_call = {
                        "id": f"cap_{turn}",
                        "tool_name": "get_caption",
                        "arguments": params
                    }
                    cap_result = self._execute_tool_call(tool_call)
                    self._master_history.append({
                        "role": "tool", 
                        "content": [{"type": "text", "text": json.dumps(cap_result, ensure_ascii=False)}]
                    })
                
                else:
                    logger.warning(f"Unknown action: {action}")
                    error_obs = {"call_id": f"err_{turn}", "tool_name": "master", "error": f"Unknown action: {action}"}
                    self._master_history.append({
                        "role": "tool", 
                        "content": [{"type": "text", "text": json.dumps(error_obs, ensure_ascii=False)}]
                    })

            return final_answer, self._master_history

        except Exception as e:
            logger.error(f"❌ Critical Error in Loop: {e}", exc_info=True)
            return f"Error: {e}", self._master_history