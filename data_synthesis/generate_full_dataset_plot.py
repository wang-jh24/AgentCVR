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

OUTPUT_DIR = "plot_synthesis"
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "generation_errors.json")

# Concurrency
MAX_WORKERS = 10      
TARGET_TOTAL = 1000   
MAX_RETRIES = 3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 2. Dimension Pools =================

GENRES = [
    "Crime Mystery", "Sci-Fi Survival", "Supernatural Horror", 
    "Post-Apocalyptic", "Historical War", "Espionage Thriller",
    "Psychological Drama", "Adventure/Heist"
]

CONFLICTS = [
    "A betrayal by a trusted ally",
    "A sudden mechanical/structural failure",
    "A discovery of a hidden dangerous object",
    "A physical injury causing incapacitation",
    "A loss of critical communication equipment",
    "A sudden change in weather/environment blocking the path",
    "An encounter with a hostile entity (human or creature)",
    "A mistaken identity leading to confrontation",
    "A theft of a vital resource",
    "A moral dilemma forcing a split in the group",
    "A trap triggered by carelessness",
    "A realization of being watched/followed"
]

SETTINGS = [
    "An abandoned subway station", "A dense foggy forest", "A high-tech laboratory",
    "A crowded marketplace", "A desolate highway motel", "A crumbling ancient ruin",
    "A luxury penthouse during a storm", "A submarine or spaceship interior",
    "A snowy mountain cabin", "An underground bunker", "A chaotic hospital wing",
    "A quiet library archive"
]

# ================= 3. Time-Slot Logic =================

def generate_fine_grained_slots(start_time, end_time):
    slots = []
    current = start_time
    while current < end_time:
        remaining = end_time - current
        if remaining <= 20:
            slots.append({"start": current, "end": end_time})
            break
        step = random.randint(5, 15)
        slots.append({"start": current, "end": current + step})
        current += step
    return slots

def calculate_structure():
    dur_begin = random.randint(60, 150)
    dur_gap = random.randint(60, 150)
    dur_end = random.randint(60, 150)
    
    t1 = dur_begin               
    t2 = dur_begin + dur_gap     
    total = t2 + dur_end         
    
    return {
        "total_duration": total,
        "beginning_range": [0, t1],
        "gap_range": [t1, t2],
        "ending_range": [t2, total],
        "beginning_slots": generate_fine_grained_slots(0, t1),
        "ending_slots": generate_fine_grained_slots(t2, total)
    }

# ================= 4. Prompt Constructor (option format fixed) =================

def create_dense_prompt(genre, conflict, setting, structure):
    target_letter = random.choice(["A", "B", "C", "D", "E", "F"])
    
    begin_slots_str = "\n".join([f"- Slot {i+1}: {s['start']}s to {s['end']}s" for i, s in enumerate(structure['beginning_slots'])])
    end_slots_str = "\n".join([f"- Slot {i+1}: {s['start']}s to {s['end']}s" for i, s in enumerate(structure['ending_slots'])])
    
    gap_duration = structure['gap_range'][1] - structure['gap_range'][0]
    
    return f"""
### Role
You are a **Screenwriter**.
Task: Create a **Plot Inference (Missing Middle)** challenge.
* **Genre**: {genre}
* **Setting**: {setting}
* **Core Conflict (The Missing Event)**: {conflict}

### Structure
* **Act 1 (Beginning)**: Setup.
* **Act 2 (HIDDEN Middle)**: Duration {gap_duration} seconds. The "{conflict}" happens here. DO NOT write the script, but ensure Act 3 reflects its consequences.
* **Act 3 (Ending)**: Aftermath. Visual state must change drastically due to Act 2.

### 🛑 INSTRUCTION: FILL THE TIMELINE
I have pre-calculated the time slots. **You must generate 'visual' and 'caption' for EACH slot.**

**Act 1 Slots (Beginning):**
{begin_slots_str}

**Act 3 Slots (Ending):**
{end_slots_str}

### Content Requirements (Strict)
1.  **Visual**: Focus ONLY on Narrative Content (Characters, Actions, Environment, Plot). NO camera/lighting jargon.
2.  **Caption**: Dialogue ONLY. Empty if silent.

### Option Generation (Crucial)
Generate 6 distinct plot summaries for the missing Act 2.
* **Correct Answer ({target_letter})**: Accurately describes the "{conflict}" that bridges Act 1 and Act 3.
* **Distractors**: 5 plausible but incorrect events that fail to explain the specific visual changes in Act 3.

### Output JSON Format (Strict)
{{
  "genre": "{genre}",
  "question": "Based on the beginning and ending clips, what event most likely occurred in the missing timeframe?",
  "options": [
      "Summary of event A...",
      "Summary of event B...",
      "Summary of event C...",
      "Summary of event D...",
      "Summary of event E...",
      "Summary of event F..."
  ],
  "correct_answer": "{target_letter}",
  "logic_reasoning": "Explanation...",
  "scripts": {{
      "beginning_content": [
          {{ "visual": "The man walks into the room looking confused.", "caption": "Hello?" }},
          ...
      ],
      "ending_content": [
          {{ "visual": "The man runs out bleeding.", "caption": "Help me!" }},
          ...
      ]
  }}
}}
"""

# ================= 5. API Request =================

def request_gemini(prompt):
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": 65535}
    }
    try:
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=240)
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        pass
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

# ================= 6. Task Execution Logic =================

def generate_unique_tasks(target_count):
    print("🧮 Calculating unique combinations...")
    core_combinations = list(itertools.product(GENRES, CONFLICTS, SETTINGS))
    random.shuffle(core_combinations)
    
    tasks = []
    for i in range(target_count):
        combo = core_combinations[i % len(core_combinations)]
        tasks.append({
            "id": i,
            "genre": combo[0],
            "conflict": combo[1],
            "setting": combo[2]
        })
    return tasks

def process_and_save_task(task_params):
    task_id = task_params['id']
    structure = calculate_structure()
    
    prompt = create_dense_prompt(
        task_params['genre'], 
        task_params['conflict'], 
        task_params['setting'],
        structure
    )
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            raw_text = request_gemini(prompt)
            data = extract_json(raw_text)
            
            if data and "scripts" in data and "options" in data:
                # 1. Format options (A: Content)
                raw_options = data["options"]
                if len(raw_options) != 6:
                    last_error = f"Options count incorrect: {len(raw_options)}"
                    continue
                
                formatted_options = []
                letters = ["A", "B", "C", "D", "E", "F"]
                for i, opt in enumerate(raw_options):
                    # Strip optional "Option A:" or "A." prefix; Python adds label
                    clean_opt = re.sub(r'^(Option\s?)?[A-F][.:\)]\s*', '', opt, flags=re.IGNORECASE).strip()
                    formatted_options.append(f"{letters[i]}: {clean_opt}")

                # 2. Merge script timelines
                llm_begin = data["scripts"].get("beginning_content", [])
                llm_end = data["scripts"].get("ending_content", [])
                
                slots_begin = structure["beginning_slots"]
                slots_end = structure["ending_slots"]
                
                min_len_begin = min(len(llm_begin), len(slots_begin))
                min_len_end = min(len(llm_end), len(slots_end))
                
                if min_len_begin < 3 or min_len_end < 3:
                    last_error = "LLM generated too few segments"
                    continue

                final_begin_timeline = []
                for i in range(min_len_begin):
                    slot = slots_begin[i]
                    content = llm_begin[i]
                    final_begin_timeline.append({
                        "start": slot["start"],
                        "end": slot["end"],
                        "visual": content.get("visual", ""),
                        "caption": content.get("caption", "")
                    })

                final_end_timeline = []
                for i in range(min_len_end):
                    slot = slots_end[i]
                    content = llm_end[i]
                    final_end_timeline.append({
                        "start": slot["start"],
                        "end": slot["end"],
                        "visual": content.get("visual", ""),
                        "caption": content.get("caption", "")
                    })

                target_letter = data.get("correct_answer") or data.get("answer")
                if not target_letter or target_letter not in letters: 
                    # Fallback: if model returns index (0-5)
                    try:
                        idx = int(target_letter)
                        target_letter = letters[idx]
                    except:
                        continue

                # 3. Build final output
                final_json = {
                    "id": task_id,
                    "meta": {
                        "genre": task_params['genre'],
                        "setting": task_params['setting'],
                        "hidden_conflict": task_params['conflict'],
                        "logic_reasoning": data.get("logic_reasoning")
                    },
                    "video_metadata": {
                        "total_duration": structure['total_duration'],
                        "gap_interval": structure['gap_range']
                    },
                    "question": data["question"],
                    "options": formatted_options, # [ "A: ...", "B: ...", ... ]
                    "correct_answer": target_letter, # "A", "B"...
                    "scripts": {
                        "beginning": {
                            "duration": structure['beginning_range'][1],
                            "timeline": final_begin_timeline
                        },
                        "ending": {
                            "duration": structure['ending_range'][1] - structure['ending_range'][0],
                            "timeline": final_end_timeline
                        }
                    }
                }
                
                file_name = f"task_{task_id}.json"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, indent=2, ensure_ascii=False)
                    
                return {"status": "success", "id": task_id}
            else:
                last_error = "Invalid JSON structure"
        except Exception as e:
            last_error = str(e)
        
        time.sleep(random.uniform(1, 2))
        
    return {"status": "failed", "id": task_id, "error": last_error}

# ================= 7. Main =================

def main():
    print(f"🚀 Starting Plot Inference (Formatted Options A-F)...")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    all_tasks = generate_unique_tasks(TARGET_TOTAL)
    
    failed_tasks = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_and_save_task, t): t['id'] for t in all_tasks}
        
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            completed += 1
            
            if result['status'] == 'success':
                if completed % 20 == 0:
                    print(f"[{completed}/{TARGET_TOTAL}] ✅ Saved task_{result['id']}.json")
            else:
                print(f"[{completed}/{TARGET_TOTAL}] ❌ FAILED Task {result['id']}: {result.get('error')}")
                failed_tasks.append(result)

    print(f"\n✨ Done in {time.time() - start_time:.2f}s")
    if failed_tasks:
        with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_tasks, f, indent=2)

if __name__ == "__main__":
    main()