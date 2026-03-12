# AgentCVR

**AgentCVR** is an agent system for complex video understanding: it trains a Master Agent via **script-simulated reinforcement learning** in a pure-text environment, then runs multi-turn reasoning and tool calls on real-world video and audio to perform sorting, alignment, assembly, multiple-choice, open-ended QA, and UAV multi-view tasks.

---

## Method Overview

The pipeline has two stages:

- **Panel A — Script-simulated RL training**: Offline generation of semantic text scripts (Procedural Script Generation), then an LLM simulator provides a pure-text environment where the agent learns through simulated multi-turn tool interaction; formatting reward (Rfmt) and correctness reward (Rans) drive GRPO updates to obtain the Master Agent πθ.
- **Panel B — Real-world inference**: The trained Master Agent is deployed on real video/audio to perform multi-turn reasoning given user queries, calling vision (Observe) and hearing (Listen) tools and returning final answers.

The figure below is the main architecture from the paper.

![AgentCVR Architecture](assets/architecture.png)

---

## Project Structure

This repo contains three core modules corresponding to **script data synthesis → Agentic RL training → Agent inference & evaluation**. **Each module requires its own environment setup, model deployment/API configuration, and running the corresponding scripts to run experiments.**

| Directory | Purpose | Description |
|------------|---------|-------------|
| **data_synthesis/** | **Script data synthesis** | Uses a Generator LLM (e.g. Gemini-compatible API) to generate semantic text scripts offline and produce synthetic JSON for each task type, for later RL training. See [data_synthesis/README.md](data_synthesis/README.md) |
| **verl/** | **Agentic RL** | Script-based RL training framework (e.g. GRPO): multi-turn tool interaction and reward learning in a pure-text simulated environment to train the Master Agent. You need to deploy or connect inference and training backends. See submodules under `verl/` |
| **agent_system/** | **Agent system setup & usage** | ReAct-style video/image agent: Master LLM drives the loop and calls `active_perception` (VLM over frames) and `get_caption` (on-demand Whisper) for inference and evaluation on real video/audio. Test data uses the [CrossVid](https://github.com/chuntianli666/CrossVid/tree/main/eval) benchmark; place question JSONs in `agent_system/question/`. See [agent_system/README.md](agent_system/README.md) |
| **assets/** | Resources | Paper figures and assets (including main architecture figure `architecture.png`) |

---

## Usage Overview

First clone the repo: `git clone https://github.com/YOUR_USERNAME/AgentCVR.git && cd AgentCVR`.

To run the full pipeline, complete the following three steps in order. **For each step, set up a separate environment in the corresponding directory, deploy/configure the relevant models and APIs, and run the relevant scripts.**

1. **Step 1 — Script data synthesis** (`data_synthesis/`): Set up environment → configure synthesis API (e.g. Gemini) → run generation scripts to obtain synthetic data.
2. **Step 2 — Agentic RL training** (`verl/`): Set up environment → deploy/configure training and inference models (e.g. vLLM, SGLang) → run RL training scripts.
3. **Step 3 — Agent inference & evaluation** (`agent_system/`): Set up environment → configure Master / VLM / caption APIs and data paths → run task scripts for inference and evaluation.

---

## Step 1: Script Data Synthesis (data_synthesis)

- **Purpose**: Generate semantic text scripts and synthetic task JSON for RL training.
- **Environment**: Create a virtual environment under `data_synthesis/` and install dependencies (see `requirements.txt` and README in that directory).
- **Deployment / API**: Configure a Gemini-compatible API (e.g. `GEMINI_API_KEY`, `GEMINI_ENDPOINT`). No local model deployment required.
- **Run scripts**: Execute the generation scripts for each task, e.g.:
  ```bash
  cd data_synthesis
  pip install -r requirements.txt
  cp .env.example .env   # set GEMINI_API_KEY etc.
  python generate_full_dataset_sort.py
  python generate_full_dataset_grounding.py
  # see data_synthesis/README.md for other scripts
  ```

---

## Step 2: Agentic RL Training (verl)

- **Purpose**: Multi-turn tool interaction and GRPO-style RL in a script-simulated pure-text environment to train the Master Agent.
- **Environment**: Create an environment under `verl/` and install dependencies as required by the framework (see `verl/requirements.txt` and docs).
- **Deploy models**: Deploy or configure inference/training backends (e.g. vLLM, SGLang, distributed training) per the framework documentation.
- **API / config**: Configure model endpoints, reward model, data paths, etc. as required by the training scripts.
- **Run scripts**: Execute the RL training entry scripts; see examples and docs under `verl/` for exact commands.

---

## Step 3: Agent System Setup & Usage (agent_system)

- **Purpose**: Run the trained (or off-the-shelf) Master Agent on real video/audio for inference and evaluation on sorting, alignment, multiple-choice, open QA, UAV multi-view, etc.
- **Test data**: Question files (`PSS.json`, `FSA.json`, etc.) are **not** included in this repo. They follow the [**CrossVid**](https://github.com/chuntianli666/CrossVid) benchmark. Create `agent_system/question/`, get the eval data from [CrossVid/eval](https://github.com/chuntianli666/CrossVid/tree/main/eval), and place the JSON files there. See [agent_system/README.md](agent_system/README.md#question--test-data).
- **Environment**: Create a virtual environment under `agent_system/` and install dependencies (see `agent_system/requirements.txt`). Requires Python 3.8+ and ffmpeg if using `get_caption`.
- **Deploy models / API**: Configure OpenAI-compatible Chat API (Master LLM + tool-side VLM); optional local or remote Whisper for captions; scoring API for CCQA if needed. Copy `agent_system/.env.example` to `.env` and set `MASTER_API_*`, `TOOL_API_*`, `LOCAL_VIDEO_ROOT`, `REMOTE_VIDEO_BASE_URL`, etc.
- **Run scripts**: From the `agent_system/` directory, run the task scripts:
  ```bash
  cd agent_system
  pip install -r requirements.txt
  cp .env.example .env   # set API keys and data paths
  python run_tasks/run_PSS_agent.py
  python run_tasks/run_FSA_agent.py
  python run_tasks/run_PEA_agent.py
  python run_tasks/run_CC_agent.py
  python run_tasks/run_CCQA_agent.py
  python run_tasks/run_MOC_agent.py
  python run_tasks/run_MSR_agent.py
  # see agent_system/README.md for other tasks
  ```
  Logs are written under `agent_system/logs/<task_name>/<question_id>/` (JSON + readable TXT).

---