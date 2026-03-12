# Datasyn – Synthetic Video Reasoning Dataset Generators

Scripts to generate synthetic video understanding / reasoning datasets using a Gemini-compatible API. Each script produces JSON task files for a specific benchmark type.

## Setup

1. **Clone the repository** (or copy the project).

2. **Set your API key** (required):
   - Copy `.env.example` to `.env` and add your key:
     ```bash
     cp .env.example .env
     # Edit .env and set GEMINI_API_KEY=your_actual_key
     ```
   - Or export in the shell before running:
     ```bash
     export GEMINI_API_KEY=your_actual_key
     ```

3. **(Optional)** Load `.env` automatically, e.g. with `python-dotenv`:
   ```bash
   pip install python-dotenv
   ```
   Then at the top of each script you can add:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```
   Or run with: `python -c "import dotenv; dotenv.load_dotenv(); exec(open('generate_full_dataset_assembly.py').read())"` (not recommended; better to add `load_dotenv()` once in each script if you use .env).

## Scripts Overview

| Script | Output Dir | Description |
|--------|------------|-------------|
| `generate_full_dataset_assembly.py` | `assembly_synthesis/` | Toy assembly tasks with personas and errors |
| `generate_full_dataset_behavior.py` | `behavior_synthesis/` | Animal/human behavior understanding |
| `generate_full_dataset_cooking.py` | `cooking_synthesis/` | Cooking video difference (YouCook2-style) |
| `generate_full_dataset_grounding.py` | `grounding_synthesis/` | Video alignment / grounding (FSA-style) |
| `generate_full_dataset_movie.py` | `movie_synthesis/` | Movie-style single-choice video understanding |
| `generate_full_dataset_plot.py` | `plot_synthesis/` | Plot inference (missing middle) |
| `generate_full_dataset_sort.py` | `sort_synthesis/` | Chronological order (cooking steps) |
| `generate_full_dataset_uav_count.py` | `moc_synthesis/` | Multi-view object counting (MOC) |
| `generate_full_dataset_uav_positon.py` | `msr_synthesis/` | Multi-view spatial reasoning (MSR) |

## Running

```bash
# Example: generate assembly tasks
python generate_full_dataset_assembly.py

# Example: generate cooking tasks
python generate_full_dataset_cooking.py
```

Output is written under the corresponding `*_synthesis/` directory as JSON files (e.g. `task_0.json`, `task_1.json`, …). Failed tasks may be logged in `generation_errors.json` or `failed_tasks.json` depending on the script.

## Configuration

- **API endpoint**: The default in code is a placeholder (`https://example.googleapis.com/v1:generateContent`). Set `GEMINI_ENDPOINT` in `.env` or the environment to your actual Gemini-compatible API URL.
- **Concurrency / retries**: Edit `MAX_WORKERS`, `MAX_RETRIES`, and target counts at the top of each script as needed.

## License

Use and modify as needed for your project. Ensure you comply with the API provider’s terms when generating data.
