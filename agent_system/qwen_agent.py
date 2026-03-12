import abc
import json
import os
import time
import requests
from typing import List, Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

device = "cpu"


class BaseModel(abc.ABC):
    requires_gpu = False
    def __init__(self, gpu_number: int = 0): self.dev = device
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str: ...
    @classmethod
    def list_processes(cls) -> List[str]: return [cls.name()]

class QwenModel(BaseModel):
    _name = 'qwen_layered_agent'
    requires_gpu = False

    def __init__(self, gpu_number=0, prompt_config: Optional[Dict[str, str]] = None):
        super().__init__(gpu_number)
        # API config from env (no secrets in repo)
        self.master_config = {
            "base_url": os.environ.get("MASTER_API_BASE_URL", "http://example-master-api.example.com/v1/chat/completions"),
            "api_key": os.environ.get("MASTER_API_KEY", "your-api-key"),
            "model_name": os.environ.get("MASTER_MODEL_NAME", "your-master-model"),
            "max_tokens": int(os.environ.get("MASTER_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("MASTER_TEMPERATURE", "0.5")),
        }

        self.tool_config = {
            "base_url": os.environ.get("TOOL_API_BASE_URL", "http://example-tool-api.example.com/v1/chat/completions"),
            "api_key": os.environ.get("TOOL_API_KEY", "your-api-key"),
            "model_name": os.environ.get("TOOL_MODEL_NAME", "your-tool-model"),
            "max_tokens": int(os.environ.get("TOOL_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("TOOL_TEMPERATURE", "0.5")),
        }
        self._master_seq = 0

        self.prompt_config = dict(prompt_config) if prompt_config else {}
        self.last_tool_prompt: Optional[str] = None
        self.last_tool_name: Optional[str] = None
        self.last_tool_images: int = 0

    @classmethod
    def name(cls): return cls._name

    def forward(self, task: str, **kwargs) -> Any:
        """Dispatches to master (chat) or active_perception (VLM); matches executor tools."""
        router = {
            'master': self._master,
            'active_perception': self._active_perception,
        }
        handler = router.get(task)
        if not handler:
            raise ValueError(f"Unknown task '{task}'. Supported: master, active_perception.")
        return handler(**kwargs)

    def _call_api(self, prompt_text: str, base64_images: Optional[List[str]] = None, expect_json: bool = False, config: Dict = None) -> str:
        api_cfg = config or self.tool_config
        
        headers = {"Authorization": f"Bearer {api_cfg['api_key']}", "Content-Type": "application/json"}
        logger.debug(f"  [_call_api] Using Model: {api_cfg['model_name']} @ {api_cfg['base_url']}")

        content_list = []
        if base64_images and "<frame>" in prompt_text:
            logger.info(f"  [_call_api] Found <frame> tags. Using interleaved logic with {len(base64_images)} images.")
            prompt_splits = prompt_text.split('<frame>')
            expected_placeholders = len(prompt_splits) - 1
            if expected_placeholders != len(base64_images):
                logger.warning(
                    f"  [_call_api] Mismatch! Prompt has {expected_placeholders} <frame> tags, "
                    f"but {len(base64_images)} images were provided. "
                    f"Will iterate up to {min(expected_placeholders, len(base64_images))} images."
                )

            for idx, split in enumerate(prompt_splits):
                if split:
                    content_list.append({"type": "text", "text": split})
                if idx < len(base64_images):
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_images[idx]}"}
                    })

        else:
            if "<frame>" in prompt_text:
                logger.warning("  [_call_api] <frame> tags in prompt but no images provided.")
            if base64_images:
                for img_b64 in base64_images:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
            content_list.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content_list}]
        payload: Dict[str, Any] = {
            "model": api_cfg['model_name'],
            "messages": messages,
            "max_tokens": api_cfg['max_tokens'],
            "temperature": api_cfg['temperature']
        }
        if expect_json:
            payload['result_format'] = 'message'

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"  [_call_api] Retry {attempt + 1}/{max_retries}...")
                response = requests.post(api_cfg['base_url'], headers=headers, json=payload, timeout=1200)
                response.raise_for_status()
                
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if content:
                    return content
                else:
                    logger.warning(f"  [_call_api] API returned empty content (Attempt {attempt + 1}).")
                    if attempt < max_retries - 1:
                        raise ValueError("Empty content received")
                    return "[API returned empty content]"

            except Exception as e:
                logger.error(f"  [_call_api] Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"Error: {e}"


    def _master(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        logger.info("  [QwenModel] 🧠 Routing to MASTER Agent API")
        if messages is not None:
            return self._call_api_chat(messages=messages, expect_json=True, config=self.master_config)
        return self._call_api(prompt_text=prompt, expect_json=True, config=self.master_config)

    def _active_perception(self, frames: List[str], prompt: str, return_prompt: bool = False) -> Union[str, Dict[str, Any]]:
        logger.info("  [QwenModel] 👁️ Routing to TOOL Agent API (Active Perception)")
        
        frame_markers = "".join(f"<frame>" for _ in range(len(frames or [])))
        prompt_with_markers = prompt + "\n" + frame_markers
        
        self.last_tool_prompt = prompt_with_markers
        self.last_tool_name = "active_perception"
        self.last_tool_images = len(frames or [])
        result = self._call_api(prompt_text=prompt_with_markers, base64_images=frames, config=self.tool_config)
        return {"result": result, "prompt": prompt_with_markers} if return_prompt else result

    def _call_api_chat(self, messages: list, expect_json: bool = False, config: Dict = None) -> str:
        api_cfg = config or self.master_config
        
        headers = {
            "Authorization": f"Bearer {api_cfg['api_key']}",
            "Content-Type": "application/json"
        }
        base_url = api_cfg["base_url"]
        is_openai_like = "/v1/chat/completions" in base_url or "compatible-mode" in base_url
        force_downgrade = api_cfg.get("force_user_role", False)

        def _downgrade_tool_msgs(msgs: list) -> list:
            out = []
            for m in msgs:
                if m.get("role") == "tool":
                     out.append({"role": "user", "content": f"[TOOL OBS] {m.get('content')}"})
                else:
                     out.append(m)
            return out

        payload: Dict[str, Any] = {}
        if is_openai_like:
            msgs_for_send = _downgrade_tool_msgs(messages) if force_downgrade else messages
            payload = {
                "model": api_cfg["model_name"],
                "messages": msgs_for_send,
                "max_tokens": api_cfg.get("max_tokens", 1024),
                "temperature": api_cfg.get("temperature", 0.2),
            }
        else:
            norm_msgs = _downgrade_tool_msgs(messages)
            payload = {
                "model": api_cfg["model_name"],
                "input": {"messages": norm_msgs},
                "parameters": {}
            }
            if expect_json: payload["parameters"]["result_format"] = "message"

        # Retry Loop
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(base_url, headers=headers, json=payload, timeout=1200)
                resp.raise_for_status()
                data = resp.json()
                content = (((data.get("choices") or [{}])[0].get("message") or {}) or {}).get("content")
                if not content: content = data.get("output_text") or data.get("content")
                
                if not content or not str(content).strip(): raise ValueError("Empty content")
                return content.strip()
            except Exception as e:
                logger.error(f"🟥 [QWEN CHAT] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1: time.sleep(3); continue
                return json.dumps({"type": "error", "content": f"Failed: {e}"}, ensure_ascii=False)