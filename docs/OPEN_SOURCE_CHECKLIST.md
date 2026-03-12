# AgentCVR Open-Source Readiness Checklist

This document summarizes what is already in place and what is recommended to add before publishing the repo.

---

## ✅ Already in place

| Item | Status |
|------|--------|
| Root README (English) | ✅ Describes method, structure, three modules, and step-by-step usage |
| Paper figure in README | ✅ `assets/architecture.png` linked |
| agent_system | ✅ README, requirements.txt, .env.example, run scripts, prompts |
| data_synthesis | ✅ README, requirements.txt, .env.example, .gitignore, generation scripts |
| verl | ✅ LICENSE (Apache-2.0), requirements, framework docs |
| No hardcoded secrets in agent_system | ✅ Config via .env; .env.example only |
| data_synthesis .gitignore | ✅ .env, venv, __pycache__, generated dirs |

---

## ⚠️ Recommended additions

### 1. Root-level files

| File | Purpose |
|------|--------|
| **LICENSE** | Choose a license (e.g. Apache-2.0 to align with verl, or MIT) and add a single LICENSE file at repo root. Subfolders (e.g. verl) can keep their own if they are submodules or different licenses. |
| **.gitignore** | Add at repo root to avoid committing `.env`, `logs/`, `*_synthesis/` outputs, `__pycache__/`, `.venv/`, IDE files, and large data/caches. Prevents accidental leak of keys and clutter. |

### 2. agent_system: question files and docs

| Item | Purpose |
|------|--------|
| **agent_system/question/** | This directory is **missing**. All run scripts expect e.g. `question/PSS.json`, `question/FSA.json`, etc. Either: (a) create `question/` and add a **README or schema** describing the expected JSON format per task (and optionally 1–2 minimal example JSONs), or (b) document in agent_system/README that users must create `question/` and obtain task JSONs (e.g. from data_synthesis or your benchmark). |
| **Question JSON schema / example** | Document or add minimal examples (e.g. `question/README.md` with field descriptions, or `question/PSS_example.json`) so users know the shape of each task file. |
| **agent_system/.gitignore** | Add so that `.env` and `logs/` are not committed. Right now only data_synthesis has a .gitignore. |

### 3. data_synthesis

| Item | Status / Suggestion |
|------|---------------------|
| .env.example, README, .gitignore | ✅ Already good. |
| Optional | If synthetic output is meant to be consumed by agent_system, add a short note in data_synthesis/README or root README on how to copy/move generated JSON into `agent_system/question/` (or equivalent). |

### 4. verl (Agentic RL)

| Item | Purpose |
|------|--------|
| **Entry point / example for AgentCVR** | If you have a specific script or config that runs “AgentCVR-style” GRPO (script-simulated env), add a short section in root README or in `verl/` pointing to that entry (e.g. `verl/examples/...` or a config path). So users know exactly which script to run for Step 2. |
| **verl .env or config** | If AgentCVR’s RL step needs env vars (API keys, model paths), consider adding a small `.env.example` in the relevant subfolder or document required variables in the README. |

### 5. Documentation and community

| Item | Purpose |
|------|--------|
| **CITATION / BibTeX** | You removed it from README; optional: add a `CITATION.bib` or `CITATION.cff` in the root, or a short “Citation” section in README again, so users can cite the paper. |
| **CONTRIBUTING.md** | Optional; useful if you expect external contributions (how to run tests, code style, PR process). |
| **Changelog / Release notes** | Optional; a simple `CHANGELOG.md` or GitHub Releases helps users see what changed between versions. |

### 6. Safety and repo hygiene

| Item | Action |
|------|--------|
| **Secrets scan** | Before pushing: ensure no `.env` or real API keys exist in the repo; only `.env.example` with placeholders. |
| **Large files / data** | Do not commit raw videos or large datasets; use `.gitignore` and document where to download data or how to generate it (data_synthesis). |
| **Python version** | README already mentions Python 3.8+; you can add a simple `pyproject.toml` or note in README if you want to pin 3.8/3.9/3.10+. |

---

## Summary table

| Category | Must-have before open source | Nice-to-have |
|----------|------------------------------|--------------|
| **Root** | LICENSE, .gitignore | CITATION, CONTRIBUTING, CHANGELOG |
| **agent_system** | question/ dir + doc or schema; .gitignore for .env and logs | Example question JSONs |
| **data_synthesis** | — | Note on feeding output to agent_system |
| **verl** | — | Clear AgentCVR RL entry in README; .env.example if needed |

---

## Minimal quick wins

1. **Add root `.gitignore`** – ignore `.env`, `logs/`, `.venv/`, `__pycache__/`, `*_synthesis/` (or only under data_synthesis if you want to track some outputs).
2. **Add root LICENSE** – e.g. Apache-2.0 or MIT, and state in README.
3. **Create `agent_system/question/`** – add a `README.md` that describes required task JSON files (PSS.json, FSA.json, …) and their format (or link to data_synthesis output). Optionally add one minimal `*_example.json` per task.
4. **Add `agent_system/.gitignore`** – at least `.env` and `logs/`.

After these, the repo is in good shape for opening on GitHub; the rest can follow as you iterate.
