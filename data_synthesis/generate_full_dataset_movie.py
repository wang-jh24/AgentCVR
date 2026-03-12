import requests
import json
import random
import re
import time
import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================= 1. Global Configuration (Gemini 3 Flash Preview) =================
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in environment or .env before running.")
MODEL_NAME = "gemini-3-flash-preview"
ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://example.googleapis.com/v1:generateContent")

# Output directory
OUTPUT_DIR = "movie_synthesis"
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "generation_errors.json")

# Concurrency (Flash model is fast; can increase workers)
MAX_WORKERS = 10
TARGET_TOTAL = 2000
MAX_RETRIES = 3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 2. Dimension Pools (generic, high compatibility) =================

# Genres compatible with the generic elements below
MOVIE_GENRES = ["Sci-Fi Thriller", "Crime Mystery", "Survival Horror", "War Espionage"]

# Core narrative elements: generic movie-style scenarios
# Example: War Espionage + A handwritten note + Hiding behind cover + An abandoned warehouse
# Example: Sci-Fi Thriller + A flashlight + Trying to open a locked door + A dimly lit hallway

DIMENSION_POOL = {
    "Key_Object": [
        "a flashlight", 
        "a set of rusted keys", 
        "a handwritten note or letter",
        "a heavy backpack", 
        "a mobile phone with a cracked screen", 
        "a faded photograph",
        "a detailed map", 
        "a pocket watch", 
        "a small metal box",
        "a weapon (knife or handgun)"
    ],
    "Specific_Action": [
        "searching frantically through a messy room",
        "hiding behind cover to avoid detection",
        "trying to force open a locked door",
        "running away from an unseen pursuer",
        "making a 'hush' gesture to stay quiet",
        "checking the time nervously",
        "bandaging a bleeding wound",
        "closely examining a mysterious object",
        "looking over shoulder while walking",
        "passing an object to someone secretly"
    ],
    "Environment": [
        "a dimly lit hallway",
        "an empty parking lot at night",
        "a messy office filled with papers",
        "a quiet living room with curtains drawn",
        "a dense forest at twilight",
        "inside a moving vehicle",
        "an abandoned warehouse",
        "a rainy street corner",
        "a narrow staircase",
        "a public restroom"
    ]
}

# Extra dimensions for variety
EXTRA_DIMENSIONS = {
    "Character_Dynamic": [
        "two strangers forced to trust each other", 
        "a mentor guiding a novice",
        "a parent protecting a child", 
        "partners having a disagreement",
        "someone trying to help an injured person",
        "a lone survivor talking to themselves"
    ],
    "Visual_Clue": [
        "hands trembling with adrenaline", 
        "clothes stained with dirt or mud",
        "a shadow moving in the background", 
        "lights flickering intermittently",
        "a ticking sound amplifying the silence", 
        "heavy breathing visible in cold air"
    ]
}
# ================= 3. Core Logic: Full Permutation Task Generation =================

def generate_unique_tasks(target_count):
    """Generate unique task combinations via Cartesian product."""
    print("🧮 Calculating unique combinations...")
    
    # All combinations of (Genre, Object, Action, Setting)
    core_combinations = list(itertools.product(
        MOVIE_GENRES,
        DIMENSION_POOL["Key_Object"],
        DIMENSION_POOL["Specific_Action"],
        DIMENSION_POOL["Environment"]
    ))
    
    print(f"📋 Total unique scenarios available: {len(core_combinations)}")
    
    # Shuffle to avoid clustering by genre
    random.shuffle(core_combinations)
    
    selected = core_combinations[:target_count]
    
    tasks = []
    for i, (genre, obj, act, env) in enumerate(selected):
        keywords = {
            "Object": obj,
            "Action": act,
            "Setting": env
        }
        
        # Optionally add extra dimensions for richness
        if random.random() > 0.5:
            keywords["Dynamic"] = random.choice(EXTRA_DIMENSIONS["Character_Dynamic"])
        if random.random() > 0.5:
            keywords["Visual_Clue"] = random.choice(EXTRA_DIMENSIONS["Visual_Clue"])
            
        tasks.append({
            "id": i,
            "genre": genre,
            "keywords": keywords
        })
        
    return tasks

# ================= 4. Prompt Constructor (V14: Natural Visuals + Captions) =================

def create_prompt(genre, keywords):
    target_letter = random.choice(["A", "B", "C", "D"])
    
    # Per-video duration config
    video_configs = {}
    video_instructions = ""
    for letter in ["A", "B", "C", "D"]:
        dur = random.choice([420, 450]) 
        video_configs[letter] = dur
        video_instructions += f"   - **Video {letter}**: Total Duration **{dur}s**. Split into **30 to 50** variable-length segments.\n"

    kw_str = "\n".join([f"* **{k}**: {v}" for k, v in keywords.items()])
    
    return f"""
### Role
You are a **Benchmark Dataset Designer**.
Your task is to create a **Hard Single-Choice Video Understanding Problem**.
Genre: **"{genre}"**.

### Input Elements (The Target Scenario)
The **Correct Answer** must depict this specific scenario:
{kw_str}

### 🛑 CRITICAL CONSTRAINT: THE LOGIC PUZZLE (Strict Single-Choice)
You are generating 4 different video clips (A, B, C, D), but **ONLY ONE** is the correct answer.

1.  **Target Video ({target_letter})**: 
    * This is the **Correct Answer**.
    * It MUST perfectly integrate ALL the Input Elements ({kw_str}) into the plot.
    
2.  **Distractor Videos (The other 3)**:
    * These are **Incorrect Answers**.
    * They MUST belong to the same Genre and style.
    * **Hard Negatives**: They should be confusingly similar (e.g., same setting but different action, or same action but different object), but they **MUST FAIL** to match the full description of the Target Scenario.

### 🛑 VISUAL & FORMAT CONSTRAINTS
1.  **Independent Durations**:
{video_instructions}

2.  **Variable Pacing**: Use natural editing (Short Cuts 3-8s + Long Takes 15-30s). Timestamps must be seamless.

3.  **Content Requirements (Visual + Caption)**:
    For EACH segment, provide:
    * **visual**: A natural, descriptive sentence of what is seen (Action + Camera + Environment). No rigid templates.
    * **caption**: Subtitles (Dialogue, Voiceover) or specific Sound Effects (e.g., [Siren wails]).

### Output JSON Format (Strict)
{{
  "genre": "{genre}",
  "logic_adaptation": {{ "original_input": {json.dumps(keywords)}, "distractor_strategy": "Explain how the other 3 videos differ from the target..." }},
  "question": "Which video features a scene where [describe the unique target scenario]?",
  "correct_answer": "{target_letter}",
  "videos": {{
      "A": {{ 
          "duration": {video_configs['A']}, 
          "timeline": [ 
              {{ 
                  "start": 0, 
                  "end": 12, 
                  "visual": "Rain lashes against the window... (Full description)",
                  "caption": "[Thunder rumbles]"
              }},
              ... (Generate 30+ items)
          ] 
      }},
      "B": {{ "duration": {video_configs['B']}, "timeline": [...] }},
      "C": {{ "duration": {video_configs['C']}, "timeline": [...] }},
      "D": {{ "duration": {video_configs['D']}, "timeline": [...] }}
  }}
}}
"""

# ================= 5. API Request and Handling =================

def request_gemini(prompt):
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": 65535}
    }
    try:
        # Flash is fast but long output may need longer timeout
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=300)
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
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except: pass
    return None

# ================= 6. Single-Task Execution Logic =================

def process_and_save_task(task_params):
    task_id = task_params['id']
    genre = task_params['genre']
    keywords = task_params['keywords']
    
    prompt = create_prompt(genre, keywords)
    
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_text = request_gemini(prompt)
            data = extract_json(raw_text)
            
            if data and "videos" in data:
                # Validate segment count
                target_letter = data.get("correct_answer", "A")
                first_video_key = list(data["videos"].keys())[0]
                timeline = data["videos"][first_video_key].get("timeline", [])
                
                if len(timeline) < 20:
                    last_error = f"Segments too few ({len(timeline)})"
                else:
                    # Map letters A->1, B->2, ...
                    letter_to_num = {"A": "1", "B": "2", "C": "3", "D": "4"}
                    formatted_videos = {}
                    durations = []
                    
                    for letter in ["A", "B", "C", "D"]:
                        if letter in data["videos"]:
                            vid_data = data["videos"][letter]
                            num_key = letter_to_num.get(letter, letter)
                            formatted_videos[num_key] = vid_data
                            durations.append(vid_data.get("duration"))

                    final_json = {
                        "id": task_id,
                        "meta": {
                            "genre": genre,
                            "keywords": keywords,
                            "target": target_letter,
                            "logic_check": data.get("logic_adaptation", {})
                        },
                        "question": data["question"],
                        "options": ["A. Video 1", "B. Video 2", "C. Video 3", "D. Video 4"],
                        "correct_answer": target_letter,
                        "duration": durations,
                        "videos": formatted_videos
                    }
                    
                    file_name = f"task_{task_id}.json"
                    file_path = os.path.join(OUTPUT_DIR, file_name)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(final_json, f, indent=4, ensure_ascii=False)
                        
                    return {"status": "success", "id": task_id}
            else:
                last_error = "Invalid JSON structure or empty response"
                
        except Exception as e:
            last_error = str(e)
            
        # Random backoff on failure
        time.sleep(random.uniform(1, 3))
        
    return {"status": "failed", "id": task_id, "error": last_error}

# ================= 7. Main =================

def main():
    print(f"🚀 Starting Full Scale Movie Generation (v14 Config)...")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    all_tasks = generate_unique_tasks(TARGET_TOTAL)
    print(f"🎯 Scheduled {len(all_tasks)} tasks | {MAX_WORKERS} Threads")
    
    failed_tasks = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(process_and_save_task, t): t['id'] for t in all_tasks}
        
        completed = 0
        for future in as_completed(future_to_id):
            result = future.result()
            completed += 1
            
            if result['status'] == 'success':
                if completed % 10 == 0:
                    print(f"[{completed}/{len(all_tasks)}] ✅ Saved task_{result['id']}.json")
            else:
                print(f"[{completed}/{len(all_tasks)}] ❌ FAILED Task {result['id']}: {result.get('error')}")
                failed_tasks.append(result)

    duration = time.time() - start_time
    print(f"\n🏁 Pipeline Finished in {duration:.2f} seconds.")
    print(f"✅ Success: {len(all_tasks) - len(failed_tasks)}")
    print(f"❌ Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"⚠️ Saving error log to {ERROR_LOG_FILE}")
        with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_tasks, f, indent=2)

if __name__ == "__main__":
    main()