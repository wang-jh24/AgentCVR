import requests
import json
import random
import re
import time
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

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
OUTPUT_DIR = "cooking_synthesis"
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "generation_errors.json")

MAX_WORKERS = 10
MAX_RETRIES = 5

# ================= 2. Recipe Library =================

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

DIFFERENCE_TYPES = [
    {"type": "Tool Usage", "desc": "Tools used (e.g., whisk vs fork, cast iron vs non-stick)"}, 
    {"type": "Ingredient Uniqueness", "desc": "Ingredients added/omitted (e.g., butter vs oil, parsley vs cilantro)"}, 
    {"type": "Processing Method", "desc": "Action style (e.g., chopping vs grating, peeling vs unpeeled)"}, 
    {"type": "Cooking Technique", "desc": "Heat application (e.g., frying vs baking, boiling vs steaming)"}, 
    {"type": "Sequence of Events", "desc": "Order of steps (e.g., salt before vs after, meat before veg)"}
]

ROUNDS_CONFIG = [
    {"label": "Video 1", "letter": "A", "style": "Standard: Clearly visible difference."},
    {"label": "Video 2", "letter": "B", "style": "Subtle: Very fine-grained visual detail."},
    {"label": "Video 3", "letter": "C", "style": "Hard: Distractors look 90% identical to target."},
    {"label": "Video 4", "letter": "D", "style": "Negative: Focus on what is NOT done or missing."}
]

# ================= 3. Core Logic: Duration and Density =================

def get_realistic_duration():
    """Generate random duration from cooking.json-style distribution."""
    rand = random.random()
    if rand < 0.10: return random.randint(45, 90)
    elif rand < 0.65: return random.randint(120, 400)
    elif rand < 0.95: return random.randint(400, 700)
    else: return random.randint(700, 1100)

def calculate_segment_constraints(duration):
    """Compute required segment count (3-8s per segment)."""
    if duration < 100:
        min_segs = int(duration / 6)
        max_segs = int(duration / 4)
    else:
        min_segs = int(duration / 10) 
        max_segs = int(duration / 6)
    
    min_segs = max(5, min_segs)
    max_segs = min(100, max_segs) 
    return min_segs, max_segs

def extract_json(text):
    if not text: return None
    try:
        return json.loads(text)
    except:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try: return json.loads(text[start:end+1])
            except: pass
    return None

def request_gemini_pro(prompt: str) -> Optional[str]:
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.85,
            "maxOutputTokens": 65535,
            "topP": 0.95,
        },
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return None
    except Exception:
        return None

# ================= 4. Prompt Constructor =================

def create_prompt(task_id: int, recipe: str, diff: Dict, config: Dict) -> str:
    durs = {str(i): get_realistic_duration() for i in range(1, 5)}
    seg_info = {}
    for i in range(1, 5):
        min_s, max_s = calculate_segment_constraints(durs[str(i)])
        seg_info[str(i)] = f"{min_s} to {max_s}"

    return f"""
### Role
You are the **Director of a High-Precision Video Reasoning Benchmark**. Your goal is to generate "Hard Negative" training data that mimics real-world cooking videos (YouCook2 style).

### Task Assignment
- **Task ID**: {task_id}
- **Recipe**: {recipe} (ALL 4 videos must cook this dish)
- **Focus**: {diff['type']} ({diff['desc']})
- **Target**: {config['label']} (Option {config['letter']})
- **Style**: {config['style']}

### 1. The Question (Single-Choice)
Create a difficult **Single-Choice Question** where **Option {config['letter']}** is the ONLY correct answer.
The distinction must rely on **visual details** (textures, tools, colors) or **caption nuances**.

### 2. Video Scripts (CRITICAL: HYPER-GRANULARITY & VISUAL DENSITY)
Generate detailed timelines for 4 videos. You MUST follow these strict constraints:

* **A. Atomic Action Decomposition (3-8s per segment)**
    * Do NOT summarize. Break actions down into micro-steps.
    * Example: "He places pan on heat (4s)" -> "Adds oil (3s)" -> "Oil shimmers (3s)".
    * **Fill the Duration**: You MUST generate enough segments to cover the exact duration listed below.

* **B. Dense Visuals (NOT just one sentence)** [CRITICAL UPGRADE]
    * Each 'visual' field must be a **dense snapshot**.
    * **Describe the State**: Don't just say "cooking". Describe the **texture, color, and consistency** of the food at that exact moment.
    * **Show Progression**: Explicitly describe how the food looks different from the previous micro-clip (e.g., "Onions are now translucent, edges browning" vs "Onions are fully caramelized").
    * **Visual Noise**: Include tools, steam, hand positions, or background elements to make it realistic.

* **C. Irregular Timestamps (No Round Numbers)**
    * **Strictly Forbidden**: Ending timestamps with 0 or 5 (e.g., 10, 15, 20).
    * **Required**: Use realistic, messy numbers (e.g., 0-7, 7-14, 14-23).
    * **No Gaps**: The timeline must be continuous from 0 to the exact Duration.

* **D. The Logical Trap (Hard Negative)**
    * **{config['label']} (Target)**: Must contain the specific evidence for the answer.
    * **Distractors**: The other 3 videos must look 90% similar but fail on the specific detail required by the question.

### Timing Constraints (You MUST adhere to these)
* **Video 1**: Duration **{durs['1']}s**. Generate **{seg_info['1']}** segments.
* **Video 2**: Duration **{durs['2']}s**. Generate **{seg_info['2']}** segments.
* **Video 3**: Duration **{durs['3']}s**. Generate **{seg_info['3']}** segments.
* **Video 4**: Duration **{durs['4']}s**. Generate **{seg_info['4']}** segments.

### Output JSON Format
{{
  "id": {task_id},
  "meta": {{ "recipe": "{recipe}", "focus": "{diff['type']}", "target": "{config['label']}" }},
  "question": "...",
  "options": ["A. Video 1", "B. Video 2", "C. Video 3", "D. Video 4"],
  "correct_answer": "{config['letter']}",
  "reasoning": "...",
  "videos": {{
      "1": {{ "duration": {durs['1']}, "timeline": [ {{ "start": 0, "end": 7, "visual": "Dense visual description...", "caption": "..." }}, ... ] }},
      "2": {{ "duration": {durs['2']}, "timeline": [...] }},
      "3": {{ "duration": {durs['3']}, "timeline": [...] }},
      "4": {{ "duration": {durs['4']}, "timeline": [...] }}
  }}
}}
"""

# ================= 5. Worker Logic =================

def process_and_save_task(task_params: Dict) -> Dict:
    """Generate and save a single task."""
    task_id = task_params['id']
    prompt = create_prompt(
        task_id=task_id,
        recipe=task_params['recipe'],
        diff=task_params['diff'],
        config=task_params['config']
    )
    
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_text = request_gemini_pro(prompt)
            if raw_text:
                data = extract_json(raw_text)
                if data and 'videos' in data and len(data['videos']) == 4:
                    # Save as separate file
                    file_name = f"task_{task_id}.json"
                    file_path = os.path.join(OUTPUT_DIR, file_name)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        
                    return {"status": "success", "id": task_id}
                else:
                    last_error = "Invalid JSON structure"
            else:
                last_error = "API request failed or empty response"
        except Exception as e:
            last_error = str(e)
        
        time.sleep(random.uniform(2, 5))
    
    return {
        "status": "failed",
        "id": task_id,
        "params": task_params,
        "last_error": last_error
    }

# ================= 6. Main =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"📂 Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    
    print(f"🚀 Starting Cooking Dataset Generation (With Dense Visuals)")
    print(f"Target: 1200 Files -> {OUTPUT_DIR}/task_X.json")
    
    all_task_params = []
    global_id = 0
    for round_conf in ROUNDS_CONFIG:
        for diff in DIFFERENCE_TYPES:
            for recipe in REAL_RECIPES:
                all_task_params.append({
                    "id": global_id,
                    "config": round_conf,
                    "diff": diff,
                    "recipe": recipe
                })
                global_id += 1
                
    print(f"📋 Scheduled {len(all_task_params)} tasks.")
    
    failed_tasks = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(process_and_save_task, params): params['id'] for params in all_task_params}
        
        total = len(all_task_params)
        completed = 0
        
        for future in as_completed(future_to_id):
            result = future.result()
            completed += 1
            
            if result['status'] == 'success':
                if completed % 10 == 0:
                    print(f"[{completed}/{total}] ✅ Saved task_{result['id']}.json")
            else:
                print(f"[{completed}/{total}] ❌ FAILED Task {result['id']}: {result['last_error']}")
                failed_tasks.append(result)

    duration = time.time() - start_time
    print(f"\n🏁 Pipeline Finished in {duration:.2f} seconds.")
    print(f"✅ Generated: {total - len(failed_tasks)} files in '{OUTPUT_DIR}'")
    
    if failed_tasks:
        print(f"⚠️ Saving {len(failed_tasks)} errors to {ERROR_LOG_FILE}...")
        error_log = [{"id": f["id"], "params": f["params"], "error": f["last_error"]} for f in failed_tasks]
        with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)
    else:
        print("🎉 100% Success Rate. No errors.")

if __name__ == "__main__":
    main()