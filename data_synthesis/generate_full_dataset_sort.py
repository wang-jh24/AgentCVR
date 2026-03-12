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

OUTPUT_DIR = "sort_synthesis"  # Updated output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MAX_RETRIES = 5
MAX_WORKERS = 10  # Number of parallel threads

# ================= 2. Full Coverage Matrix Parameters =================

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

STEP_VARIANTS = [3, 4, 5, 6]

DIFFICULTY_LEVELS = [
    "Normal (Visually Distinct Steps)",
    "Hard (Visually Similar / Subtle Progression)"
]

FOCUS_ASPECTS = [
    "Physical State Change (e.g. melting, browning)",
    "Tool Interaction (e.g. chopping, stirring)",
    "Ingredient Addition (e.g. spices, liquids)",
    "Visual Details (e.g. steam, bubbles, texture)"
]

# ================= 3. Core Prompt Constructor =================

def create_matrix_prompt(recipe, num_steps, difficulty, focus):
    return f"""
### Role
You are the **Director of a High-Precision Video Reasoning Benchmark**.
Your goal is to generate a **chronologically ordered** cooking script for "{recipe}".

### Strict Configuration
* **Total Phases**: Exactly **{num_steps}**
* **Difficulty**: {difficulty}
* **Visual Focus**: {focus}

### Task
Break down the cooking process into **{num_steps} Distinct Sequential Phases**.
Inside EACH Phase, list **8-15 Atomic Visual Events** (Micro-Actions).

### Critical Requirements
1.  **Strict Structure**: You MUST generate exactly {num_steps} phases.
2.  **Focus Adherence**: Your visual descriptions MUST emphasize "{focus}".
3.  **Logical Flow**: The phases must follow the correct timeline (Irreversible logic).
4.  **Atomic Detail**: Do NOT write "He cooks the meat". Break it down: "Places pan", "Meat hits pan", "Searing sound", "Flipping meat".
5.  **NO TIMESTAMPS**: Just provide the list of events. My software will assign random durations later.
6.  **No Leaks**: Do not include text like "Step 1", "Finally", "Next" in the content.

### Critical Requirements 
1.  **Dense Visuals (NOT just one sentence)**: 
    * Each 'visual' field must be a **dense snapshot**.
    * **Describe the State**: Don't just say "cooking". Describe the **texture, color, and consistency** of the food at that exact moment.
    * **Show Progression**: Explicitly describe how the food looks different from the previous micro-clip (e.g., "Onions are now translucent, edges browning" vs "Onions are fully caramelized").
    * **Visual Noise**: Include tools, steam, hand positions, or background elements.

### Output JSON Format (Strict)
{{
  "recipe": "{recipe}",
  "meta": {{ 
      "num_steps": {num_steps},
      "difficulty": "{difficulty}",
      "focus": "{focus}"
  }},
  "phases": [
    {{
      "phase_id": 1,
      "description": "Short summary of phase 1",
      "events": [
        {{ "visual": "...", "caption": "..." }},
        ... (8-15 events)
      ]
    }},
    ... (Total {num_steps} phases)
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
        else:
            print(f"   ❌ API Error: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
    return None

def extract_json(text):
    if not text: return None
    try: return json.loads(text)
    except:
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m: 
            try: return json.loads(m.group(1))
            except: pass
    return None

def assign_random_timings(phases_data):
    processed_phases = []
    for phase in phases_data:
        events = phase.get("events", [])
        if not events: continue
        timed_events = []
        current_phase_time = 0
        for evt in events:
            # 3-9s random duration per event
            duration = random.choices([3,4,5,6,7,8,9], weights=[1,2,3,3,2,1,1], k=1)[0]
            start = current_phase_time
            end = current_phase_time + duration
            timed_events.append({
                "start": start, "end": end,
                "visual": evt.get("visual", ""), "caption": evt.get("caption", "")
            })
            current_phase_time = end
        processed_phases.append({
            "phase_id": phase.get("phase_id"),
            "description": phase.get("description"),
            "total_duration": current_phase_time,
            "micro_clips": timed_events
        })
    return processed_phases

# ================= 5. Single Task Processing Logic =================

def process_single_task(task_params):
    task_id = task_params['id']
    recipe = task_params['recipe']
    steps = task_params['steps']
    diff = task_params['difficulty']
    focus = task_params['focus']
    
    prompt = create_matrix_prompt(recipe, steps, diff, focus)
    
    for attempt in range(MAX_RETRIES):
        raw_text = request_gemini(prompt)
        data = extract_json(raw_text)
        
        if data and "phases" in data:
            # Relaxed validation: accept if phases count is close or exact
            # Strict validation: if len(data["phases"]) != steps: continue
            
            # 1. Inject timings
            phases_with_time = assign_random_timings(data["phases"])
            if not phases_with_time: continue

            # Update actual steps if generated count differs from requested
            actual_steps = len(phases_with_time)

            # 2. Shuffle order
            indices = list(range(1, actual_steps + 1))
            random.shuffle(indices) 
            
            # 3. Stitch and map
            segments_dict = {}
            current_global_time = 0
            real_step_to_display_pos = {}
            
            for display_idx, real_id in enumerate(indices):
                pos_str = str(display_idx + 1)
                
                # Compatibility handling: find phase by ID or index
                phase_obj = next((p for p in phases_with_time if p["phase_id"] == real_id), None)
                if not phase_obj: 
                    # Fallback to index if IDs are not 1-based integers
                    if real_id <= len(phases_with_time):
                         phase_obj = phases_with_time[real_id-1]
                    else:
                        continue # Should not happen with correct logic

                duration = phase_obj["total_duration"]
                start_g = current_global_time
                end_g = current_global_time + duration
                
                # Remap absolute timeline
                abs_timeline = []
                for clip in phase_obj["micro_clips"]:
                    abs_timeline.append({
                        "start": start_g + clip["start"],
                        "end": start_g + clip["end"],
                        "visual": clip["visual"],
                        "caption": clip["caption"]
                    })
                
                segments_dict[pos_str] = {
                    "interval": [[start_g, end_g]],
                    "detailed_timeline": abs_timeline,
                    "_debug_desc": phase_obj["description"]
                }
                
                real_step_to_display_pos[real_id] = pos_str
                current_global_time = end_g

            # 4. Generate answer
            answer_str = "->".join([real_step_to_display_pos[i] for i in range(1, actual_steps + 1)])
            
            # 5. Save
            final_json = {
                "id": task_id,
                "video": f"sim_sort_{task_id}_{int(time.time())}",
                "recipe": recipe,
                "meta": {
                    "num_steps": actual_steps, # Save actual generated steps
                    "requested_steps": steps,
                    "difficulty": diff,
                    "focus": focus
                },
                "question": f"What is the correct chronological order to make {recipe} based on essential cooking progression?",
                "answer": answer_str,
                "duration": current_global_time,
                "segments": segments_dict
            }
            
            file_path = os.path.join(OUTPUT_DIR, f"task_{task_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
                
            return {"status": "success", "id": task_id}
            
        time.sleep(random.uniform(1, 3)) # Random backoff
    
    return {"status": "failed", "id": task_id, "recipe": recipe}

# ================= 6. Main Program =================

def main():
    print(f"🚀 Initializing Matrix Plan...")
    
    # 1. Generate all possible task combinations (1920 tasks)
    all_combinations = list(itertools.product(REAL_RECIPES, STEP_VARIANTS, DIFFICULTY_LEVELS, FOCUS_ASPECTS))
    total_tasks = len(all_combinations)
    print(f"📋 Total Matrix Size: {total_tasks}")
    
    # Assign IDs
    tasks_to_run = []
    for i, (recipe, steps, diff, focus) in enumerate(all_combinations):
        tasks_to_run.append({
            "id": i,
            "recipe": recipe,
            "steps": steps,
            "difficulty": diff,
            "focus": focus
        })
        
    print(f"🔥 Starting parallel generation with {MAX_WORKERS} workers...")
    
    successful_count = 0
    failed_tasks = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_task, task): task for task in tasks_to_run}
        
        for future in as_completed(future_to_task):
            result = future.result()
            if result["status"] == "success":
                successful_count += 1
                if successful_count % 10 == 0:
                    print(f"✅ Progress: {successful_count}/{total_tasks} completed.")
            else:
                print(f"❌ Failed Task ID {result['id']} ({result['recipe']})")
                failed_tasks.append(result)
                
    print(f"\n✨ Generation Complete.")
    print(f"   Success: {successful_count}")
    print(f"   Failed: {len(failed_tasks)}")
    
    # Save failed tasks for retry
    if failed_tasks:
        error_file = os.path.join(OUTPUT_DIR, "failed_tasks.json")
        with open(error_file, "w") as f:
            json.dump(failed_tasks, f, indent=2)
        print(f"   ⚠️ Failed tasks saved to {error_file}")

if __name__ == "__main__":
    main()