import requests
import json
import random
import re
import time
import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================= 1. Global Configuration =================
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in environment or .env before running.")
MODEL_NAME = "gemini-3-flash-preview"
ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://example.googleapis.com/v1:generateContent")

# Output directory
OUTPUT_DIR = "grounding_synthesis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "generation_errors.json")

# Concurrency
MAX_WORKERS = 10
MAX_RETRIES = 5

# ================= 2. Diversity Matrix Parameters (1440 combinations) =================

REAL_RECIPES = [
    "Scrambled Eggs", "Pancakes", "Eggs Benedict", "Hash Browns", "Boxty", 
    "Dosa", "Naan", "Burger", "Pizza", "Corn Dog", "Fried Chicken", 
    "Buffalo Wings", "Onion Rings", "Spring Rolls", "Potstickers", "Samosas", 
    "Falafel", "Fish and Chips", "Calamari", "Burrito", "Grilled Cheese", 
    "BLT Sandwich", "California Roll", "Fried Rice", "Pad Thai", "Ramen", 
    "Miso Soup", "Kung Pao Chicken", "Mapo Tofu", "Bulgogi", "Chicken Curry", 
    "Chana Masala", "Dal Makhani", "Carbonara", "Mac and Cheese", 
    "Beef Bourguignon", "Shepherd's Pie", "Meatloaf", "Chicken Parmesan", 
    "Bratwurst", "Goulash", "Clam Chowder", "Minestrone", "Caesar Salad", 
    "Kimchi", "Foie Gras", "Escargot", "Risotto", "Tacos", "Guacamole",
    "French Toast", "Omelet", "Sushi Nigiri", "Bibimbap", "Tom Yum Soup",
    "Peking Duck", "Dim Sum", "Churros", "Brownies", "Apple Pie"
]

GRANULARITY_LEVELS = [
    "Phase Level (Aligning a broad process, e.g., 'Making the batter')",
    "Action Level (Aligning a specific action, e.g., 'Cracking an egg')",
    "Detail Level (Aligning a specific state change, e.g., 'Cheese melting')"
]

VARIATION_TYPES = [
    "Tool Variance (Video A uses knife, Video B uses mandoline/processor)",
    "Perspective Shift (Video A is Close-up/First-person, Video B is Wide/Third-person)",
    "Pacing Difference (Video A is fast/pro, Video B is slow/tutorial style)",
    "Ingredient Substitution (Minor visual diff, e.g., Red vs White Onion)"
]

DIFFICULTY_LEVELS = [
    "Normal: Visual similarities exist.",
    "Hard: Visuals are very different, requires causal reasoning."
]

# ================= 3. Core Prompt Constructor =================

def create_grounding_matrix_prompt(recipe, granularity, variation, difficulty):
    num_steps = 5  # Fixed 5 steps for stable structure
    
    return f"""
### Role
You are the **Chief Editor of a Video Alignment Benchmark**.
Your goal is to generate a "Parallel Script" for two different videos (Video A and Video B) cooking the SAME recipe: "{recipe}".

### Diversity Injection
* **Alignment Granularity**: {granularity}
* **Variation Style**: {variation}
* **Difficulty**: {difficulty}

### Task
Break the cooking process into **{num_steps} Distinct Functional Steps**.
For EACH step, generate TWO variations (Video A vs Video B) that perform the **SAME FUNCTION** but look **DIFFERENT**.

### Critical Requirements
1.  **Enforce Variation**: You MUST apply the "{variation}" rule rigorously.
2.  **Enforce Granularity**: The descriptions must focus on the "{granularity}" level.
3.  **Dense Visuals**: 
    * Each 'visual' field must be a **dense snapshot**.
    * Describe **texture, color, and consistency** of the food.
    * Show **progression** (e.g., "Egg white turns opaque").
4.  **NO TIMESTAMPS**: Just provide the list of events. My software will assign random durations.
5.  **Micro-Events**: For each step, list 3-8 micro-events for Video A and 3-8 for Video B.

### Output JSON Format (Strict)
{{
  "recipe": "{recipe}",
  "alignment_steps": [
    {{
      "step_id": 1,
      "function_desc": "Brief description of the step",
      "video_a_events": [
          {{ "visual": "Dense visual description...", "caption": "..." }},
          ...
      ],
      "video_b_events": [
          {{ "visual": "Dense visual description...", "caption": "..." }},
          ...
      ]
    }},
    ... (Total {num_steps} steps)
  ]
}}
"""

# ================= 4. Helper Functions =================

def request_gemini(prompt):
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": 65535}
    }
    try:
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except: pass
    return None

def extract_json(text):
    if not text: return None
    try: return json.loads(text)
    except:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match: 
            try: return json.loads(match.group(1))
            except: pass
    return None

def generate_micro_durations(num_events):
    """Generate 3-8s random durations for micro-actions."""
    durations = []
    for _ in range(num_events):
        d = random.choices([3,4,5,6,7,8], weights=[1,3,4,3,2,1], k=1)[0]
        durations.append(d)
    return durations

# ================= 5. Parallel Task Processing Logic =================

def process_and_save_task(task_params):
    task_id = task_params['id']
    recipe = task_params['recipe']
    gran = task_params['granularity']
    var = task_params['variation']
    diff = task_params['difficulty']
    
    prompt = create_grounding_matrix_prompt(recipe, gran, var, diff)
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            raw_text = request_gemini(prompt)
            data = extract_json(raw_text)
            
            if data and "alignment_steps" in data:
                steps = data["alignment_steps"]
                if len(steps) < 3: 
                    last_error = "Generated steps too few"
                    continue

                # --- Build dual timelines ---
                video_a_timeline = []
                video_b_timeline = []
                cursor_a = 0
                cursor_b = 0
                alignment_meta = {}
                
                for step in steps:
                    step_id = step.get("step_id")
                    func_desc = step.get("function_desc", "Unknown Step")
                    
                    # Video A
                    events_a = step.get("video_a_events", [])
                    if not events_a: continue
                    durs_a = generate_micro_durations(len(events_a))
                    
                    step_start_a = cursor_a
                    for evt, dur in zip(events_a, durs_a):
                        end_time = cursor_a + dur
                        video_a_timeline.append({
                            "start": cursor_a, "end": end_time,
                            "visual": evt.get("visual", ""), "caption": evt.get("caption", "")
                        })
                        cursor_a = end_time
                    step_end_a = cursor_a
                    
                    # Video B
                    events_b = step.get("video_b_events", [])
                    if not events_b: continue
                    durs_b = generate_micro_durations(len(events_b))
                    
                    step_start_b = cursor_b
                    for evt, dur in zip(events_b, durs_b):
                        end_time = cursor_b + dur
                        video_b_timeline.append({
                            "start": cursor_b, "end": end_time,
                            "visual": evt.get("visual", ""), "caption": evt.get("caption", "")
                        })
                        cursor_b = end_time
                    step_end_b = cursor_b
                    
                    # Record alignment intervals
                    alignment_meta[step_id] = {
                        "function": func_desc,
                        "interval_a": [step_start_a, step_end_a],
                        "interval_b": [step_start_b, step_end_b]
                    }

                # --- Pick a random step for the question ---
                valid_step_ids = list(alignment_meta.keys())
                if not valid_step_ids: 
                    last_error = "No valid alignment steps found"
                    continue
                
                target_step_id = random.choice(valid_step_ids)
                target_info = alignment_meta[target_step_id]
                
                # Define ref_segment and answer
                ref_segment = [float(f"{x:.2f}") for x in target_info['interval_a']]
                ans_segment = [float(f"{x:.2f}") for x in target_info['interval_b']]
                
                var_hint = var.split('(')[0].strip()
                question_str = (
                    f"In Video B, which temporal segment corresponds to the '{target_info['function']}' phase "
                    f"depicted in Video A's reference clip {ref_segment}? "
                    f"(Note: Video B is a variation featuring '{var_hint}')."
                )
                
                # --- Save ---
                final_json = {
                    "id": task_id,
                    "video A": f"sim_vid_A_{task_id}",
                    "video B": f"sim_vid_B_{task_id}",
                    "recipe": recipe,
                    "meta": {
                        "granularity": gran,
                        "variation": var,
                        "difficulty": diff
                    },
                    "ref_segment": ref_segment,
                    "question": question_str,
                    "answer": ans_segment,
                    "duration": [cursor_a, cursor_b],
                    "synthesized_videos": {
                        "video_A": { "duration": cursor_a, "timeline": video_a_timeline },
                        "video_B": { "duration": cursor_b, "timeline": video_b_timeline }
                    }
                }
                
                file_name = f"task_{task_id}.json"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, indent=2, ensure_ascii=False)
                    
                return {"status": "success", "id": task_id}
            
            else:
                last_error = "JSON decode failed or invalid structure"

        except Exception as e:
            last_error = str(e)
            
        time.sleep(random.uniform(1, 3))

    return {"status": "failed", "id": task_id, "error": last_error}

# ================= 6. Main =================

def main():
    print(f"🚀 Initializing Grounding (FSA) Full Generation...")
    print(f"📂 Output Directory: {OUTPUT_DIR}")
    
    combinations = list(itertools.product(REAL_RECIPES, GRANULARITY_LEVELS, VARIATION_TYPES, DIFFICULTY_LEVELS))
    total_tasks = len(combinations)
    print(f"📋 Total Tasks: {total_tasks} (Matrix Coverage)")
    
    tasks = []
    for i, (recipe, gran, var, diff) in enumerate(combinations):
        tasks.append({
            "id": i,
            "recipe": recipe,
            "granularity": gran,
            "variation": var,
            "difficulty": diff
        })
    
    failed_tasks = []
    start_time = time.time()
    
    print(f"🔥 Starting parallel execution with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_and_save_task, t): t['id'] for t in tasks}
        
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            
            if res["status"] == "success":
                if completed % 20 == 0:
                    print(f"[{completed}/{total_tasks}] ✅ Saved task_{res['id']}.json")
            else:
                print(f"[{completed}/{total_tasks}] ❌ Failed Task {res['id']}: {res.get('error')}")
                failed_tasks.append(res)
    
    duration = time.time() - start_time
    print(f"\n🏁 Generation Finished in {duration:.2f}s")
    print(f"✅ Success: {total_tasks - len(failed_tasks)}")
    print(f"❌ Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"⚠️ Saving error log to {ERROR_LOG_FILE}")
        with open(ERROR_LOG_FILE, 'w') as f:
            json.dump(failed_tasks, f, indent=2)

if __name__ == "__main__":
    main()