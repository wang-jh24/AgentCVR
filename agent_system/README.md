# Agent System

ReAct-style video/image agent: a **master LLM** drives the loop and calls two tools — **active_perception** (VLM over frames) and **get_caption** (on-demand Whisper). Supports standard video inputs and UAV image-folder contexts.

## Structure

| Path | Description |
|------|-------------|
| `agent_executor.py` | Orchestrates the loop: master dialogue, tool dispatch, log saving |
| `qwen_agent.py` | LLM/VLM client: master chat API and active_perception (tool) API; config from env |
| `utils/` | Helpers: config, logging, sanitize, text, answer parsing, `video_processor`, `caption_generator`, `frame_bbox` |
| `run_tasks/` | Task entry scripts (PSS, FSA, PEA, CC, NC, BU, PI, CCQA, MOC, MSR) |
| `question/` | **Not included in repo** — create this directory and add task JSON files (see [Question / test data](#question--test-data) below) |
| `prompts/` | Master prompt templates (e.g. `master_PSS.prompt`) and optional build prompts |
| `.env.example` | Template for environment variables; copy to `.env` and fill in |

## Prerequisites

- Python 3.8+
- ffmpeg (for caption segment extraction when using `get_caption`)
- API: OpenAI-compatible chat for master and tool (VLM); optional scoring API for CCQA

## Installation

Install **all dependencies** (core + caption) with one command. From the `agent_system` directory, using a virtual environment is recommended:

```bash
cd agent_system
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` includes core packages (decord, huggingface-hub, numpy, opencv-python-headless, requests) and caption-generation packages (openai-whisper, torch, tqdm, regex). No separate optional install is needed.

**System dependency:** For caption support, install ffmpeg (e.g. Ubuntu/Debian: `sudo apt install ffmpeg`). For a full pinned caption environment (e.g. conda + CUDA), see `caption.yml`.

## Setup

1. Copy the env template and set your API keys and paths:
   ```bash
   cp .env.example .env
   # Edit .env: MASTER_API_*, TOOL_API_*, LOCAL_VIDEO_ROOT, REMOTE_VIDEO_BASE_URL, UAV_DATA_DIR (if needed), SCORING_* (for CCQA)
   ```

2. Create the `question/` directory and add task JSON files (see [Question / test data](#question--test-data) below). Master prompts are already in `prompts/`.

---

## Question / test data

The task scripts (`run_tasks/run_*_agent.py`) read question files from a **`question/`** directory. This directory is **not** shipped in the repo — you must create it yourself and place the JSON files there.

- **Required files**: `PSS.json`, `FSA.json`, `PEA.json`, `CC.json`, `NC.json`, `BU.json`, `PI.json`, `CCQA.json`, `MOC.json`, `MSR.json` (one per task you want to run).
- **Data source**: Evaluation data follows the [**CrossVid**](https://github.com/chuntianli666/CrossVid) benchmark. Get the question/eval data from [CrossVid – eval](https://github.com/chuntianli666/CrossVid/tree/main/eval) (task definitions and scripts: PSS, FSA, PEA, CC, NC, BU, PI, CCQA, MOC, MSR).
- **Steps**: Create `agent_system/question/`, obtain the JSON files in the format expected by each task (see CrossVid’s eval scripts and data), and place them in `question/` so that `run_*_agent.py` can find them. Then run the scripts from the `agent_system/` directory as in [How to run](#how-to-run).

## How to run

Run from the **agent_system** directory so that `question/`, `prompts/`, and `logs/` resolve correctly. Each script in `run_tasks/` adds the project root to `sys.path` automatically.

```bash
cd agent_system
python run_tasks/run_PSS_agent.py
python run_tasks/run_FSA_agent.py
python run_tasks/run_PEA_agent.py
python run_tasks/run_CC_agent.py
python run_tasks/run_NC_agent.py
python run_tasks/run_BU_agent.py
python run_tasks/run_PI_agent.py
python run_tasks/run_CCQA_agent.py
python run_tasks/run_MOC_agent.py
python run_tasks/run_MSR_agent.py
```

Logs are written under `logs/<task_name>/<question_id>/` (JSON + readable TXT).

## Tasks overview

| Script | Task code | Input | Output |
|--------|-----------|--------|--------|
| run_PSS_agent.py | PSS (sort) | Video segments (order unknown) | Correct order string (e.g. `2->3->1->4`) |
| run_FSA_agent.py | FSA (grounding) | Two videos + reference segment | Time interval in second video |
| run_PEA_agent.py | PEA (assembly) | Multi-video clips | Multi-choice answer |
| run_CC_agent.py / run_NC_agent.py / run_PI_agent.py / run_BU_agent.py | CC / NC / PI / BU | Video(s) + question | Multi-choice or multi-select |
| run_CCQA_agent.py | CCQA (open) | Two videos + question | Free-form answer (scored by external API) |
| run_MOC_agent.py / run_MSR_agent.py | MOC / MSR (uav_count / uav_position) | UAV image folders + bbox | Count or position answer |

## Using as a submodule or copy

You can drop this folder into your repo as the "agent system" component. Entry points are the `run_tasks/*.py` scripts; they depend only on the structure above and on `.env`. Provide your own `question/` and `prompts/` (and optionally adjust `utils/config` defaults) to match your data and APIs.

---

## Checklist for open-source use

- **Self-contained**: All agent logic lives under this folder; no hardcoded secrets (API keys and paths come from `.env`).
- **Clear entry points**: Run from project root (`agent_system`) via `python run_tasks/run_<task>_agent.py`; scripts add the root to `sys.path` so no extra `PYTHONPATH` is needed.
- **Config**: `.env.example` documents required variables; copy to `.env` and fill in.
- **Dependencies**: `requirements.txt` in this directory installs all dependencies (core + caption) with `pip install -r requirements.txt`.
- **Data**: `question/` is not in the repo; users create it and add CrossVid-format JSON files (see [Question / test data](#question--test-data)). `prompts/` is included.
- **Logs**: Written under `logs/`; you may add `logs/` to `.gitignore` in the parent repo.
