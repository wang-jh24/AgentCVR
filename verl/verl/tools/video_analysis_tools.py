# Copyright 2025 ModelBest Inc. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
import asyncio
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import aiohttp
from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VideoAnalysisTool(BaseTool):
    # Global semaphore to limit concurrency to 10
    _sem = asyncio.Semaphore(10)

    # Global singleton Session (class variable) to avoid leak on instance destroy
    _global_session: Optional[aiohttp.ClientSession] = None

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.api_url = config.get("api_url")
        self.api_key = config.get("api_key") or os.getenv("API_KEY")

        if not self.api_key:
            logger.warning("Config/Env API Key missing! Set api_key in config or API_KEY in env.")
            self.api_key = ""

        logger.info("Tool initialized with API key from config/env.")

        self.model_name = config.get("model_name", "Qwen3-4B")
        self.timeout = config.get("timeout", 120)

        self.simulator_system_prompt = """You are a video analysis simulator. Your task is to play the role of an advanced computer vision tool.
You will receive a "video script (Ground Truth)" and the user's "query".
Generate outputs that simulate the vision tool based on the script content.
Notes:
1. Do not reveal that you are reading a script; respond as if you actually see the video.
2. Strictly follow the timestamps and action descriptions in the script.
3. If the user asks about a time range with no actions in the script, reply "No significant action observed."
"""

    @classmethod
    async def get_global_session(cls, timeout_val=120):
        if cls._global_session is None or cls._global_session.closed:
            # limit=20: pool slightly larger than semaphore; ttl_dns_cache=300: cache DNS
            connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
            cls._global_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout_val)
            )
        return cls._global_session

    def set_env_data(self, instance_id: str, env_data: Any):
        if instance_id not in self._instance_dict:
             self._instance_dict[instance_id] = {"history": [], "reward": 0.0}
        
        if isinstance(env_data, str):
            try:
                env_data = json.loads(env_data.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse env_data JSON: {e}")
                env_data = {} 

        self._instance_dict[instance_id]["video_data"] = env_data

    async def create(self, create_kwargs: dict[str, Any]) -> tuple[str, dict]:
        instance_id = uuid4().hex
        ground_truth = create_kwargs.get("ground_truth")
        self._instance_dict[instance_id] = {
            "history": [], "reward": 0.0, "final_answer": None,
            "ground_truth": ground_truth, "video_data": None
        }
        return instance_id, {}

    async def _call_simulator_api(self, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "api-key": self.api_key, 
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.simulator_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1, "stream": False, "max_tokens": 8192
        }
        
        max_retries = 3
        base_delay = 2
        
        async with VideoAnalysisTool._sem:
            session = await VideoAnalysisTool.get_global_session(self.timeout)
            
            for attempt in range(max_retries):
                try:
                    async with session.post(self.api_url, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            res = await resp.json()
                            try:
                                return res['choices'][0]['message']['content']
                            except KeyError as e:
                                return f"API response format error: missing key {e}."
                        elif resp.status >= 500:
                            error_text = await resp.text()
                            raise Exception(f"Server Error {resp.status}: {error_text}")
                        else:
                            error_text = await resp.text()
                            return f"Simulator call failed (Status {resp.status}): {error_text}"

                except (aiohttp.ClientError, Exception) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final Connection Error: {e}")
                        # Recreate session only on severe errors (e.g. connection reset)
                        if isinstance(e, (aiohttp.ServerDisconnectedError, ConnectionResetError)):
                            if VideoAnalysisTool._global_session:
                                await VideoAnalysisTool._global_session.close()
                                VideoAnalysisTool._global_session = None
                        return f"Simulator connection error (retried {max_retries} times): {e}"
                    
                    wait_time = base_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
        
        return "Simulator unknown error"

    async def _execute_get_caption(self, instance_id: str, parameters: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        video_index = parameters.get("video_index")
        instance_data = self._instance_dict.get(instance_id, {})
        raw_videos = instance_data.get("video_data", {})

        if not raw_videos:
            return ToolResponse(text="Error: No video data available."), 0.0, {}

        simulation_prompt = f"""
[Video script]:
{json.dumps(raw_videos, ensure_ascii=False, indent=2)}

[User request]:
The user is requesting captions/subtitles summary for **video {video_index}**.
Look up the corresponding video in the script (match keys flexibly, e.g. 'video_{video_index}' or '{video_index}').
If found, extract and summarize the plot; if not found, reply "No data found for video {video_index}."

[Output]:
Produce a natural video caption/subtitle summary.
"""
        try:
            simulated_response = await self._call_simulator_api(simulation_prompt)
            self._instance_dict[instance_id]["history"].append({
                "action": "get_caption", "params": parameters, "response": simulated_response
            })
            return ToolResponse(text=simulated_response), 0.0, {"action": "get_caption"}
        except Exception as e:
            return ToolResponse(text=f"Error executing get_caption: {e}"), 0.0, {}

    async def _execute_observe(self, instance_id: str, parameters: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        targets = parameters.get("observation_targets", [])
        focus_prompt = parameters.get("focus_prompt", "")
        instance_data = self._instance_dict.get(instance_id, {})
        raw_videos = instance_data.get("video_data", {}) 
        
        if not raw_videos:
            return ToolResponse(text="Error: No video data available."), 0.0, {}
            
        target_desc = json.dumps(targets, ensure_ascii=False)
        simulation_prompt = f"""
[Full script library (Ground Truth) for all videos]:
{json.dumps(raw_videos, ensure_ascii=False, indent=2)}

[User observation request]:
Observation targets: {target_desc}
Focus: "{focus_prompt}"

[Instructions]:
Locate the content in the script by observation targets (match keys flexibly) and answer.
Produce a detailed visual observation report.
"""
        try:
            simulated_response = await self._call_simulator_api(simulation_prompt)
            self._instance_dict[instance_id]["history"].append({
                "action": "observe", "params": parameters, "response": simulated_response
            })
            return ToolResponse(text=simulated_response), 0.0, {"action": "observe"}
        except Exception as e:
            return ToolResponse(text=f"Error executing observe: {e}"), 0.0, {}

    async def _execute_answer(self, instance_id: str, parameters: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        final_answer = parameters.get("final_answer")
        instance_data = self._instance_dict.get(instance_id, {})
        ground_truth = instance_data.get("ground_truth")
        score = 1.0 if ground_truth and str(final_answer).strip().upper() == str(ground_truth).strip().upper() else 0.0
        self._instance_dict[instance_id]["final_answer"] = final_answer
        return ToolResponse(text=f"Answer submitted: {final_answer}"), score, {"action": "answer", "final_answer": final_answer}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        instance_data = self._instance_dict.get(instance_id)
        return instance_data.get("reward", 0.0) if instance_data else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

# Wrapper classes
class GetCaptionTool(VideoAnalysisTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await self._execute_get_caption(instance_id, parameters)

class ObserveTool(VideoAnalysisTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await self._execute_observe(instance_id, parameters)

class AnswerTool(VideoAnalysisTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await self._execute_answer(instance_id, parameters)