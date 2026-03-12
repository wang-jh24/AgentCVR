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

# ================= 1. Base Configuration =================
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in environment or .env before running.")
ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://example.googleapis.com/v1:generateContent")

# Use moc_synthesis output dir (separate from msr)
OUTPUT_DIR = "moc_synthesis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MAX_WORKERS = 10     
MAX_RETRIES = 3
TARGET_TOTAL = 1000

# ================= 2. MOC Task Configuration Matrix (extended) =================

SCENARIO_TYPES = [
    "Urban Intersection with Crosswalks", 
    "Residential Street with Dense Roadside Parking",
    "Wide Boulevard with U-Turn Lane", 
    "T-Junction with Bus Stop (View blocked by Bus)",
    "Industrial Zone with Large Tankers/Trucks", 
    "Fork in the Road (Y-Junction)",
    "Signalized Intersection with Waiting Queues", 
    "Two-way Narrow Road with Oncoming Traffic",
    "Highway Exit Ramp with Deceleration Lane", 
    "Roundabout with Multiple Entries",
    "Construction Zone Narrowing", 
    "School Zone with Speed Bumps",
    "Parking Lot Entrance/Exit Queue", 
    "Curved Mountain Road (Blind Spots)",
    "Underpass/Bridge Segment", 
    "Market Street with Mixed Cycles/Pedestrians",
    "One-way Alley with Delivery Vans", 
    "Multi-lane Highway Merge"
]

CAMERA_CONFIGS = [
    {"type": "Overlapping-Adjacent", "desc": "View A captures North side, View B captures South side. Overlap in center."},
    {"type": "Orthogonal-Intersection", "desc": "View A looks East-West, View B looks North-South (90 degree difference)."},
    {"type": "Front-Rear", "desc": "View A captures oncoming traffic (Front), View B captures departing traffic (Rear)."},
    {"type": "Top-Down-Oblique", "desc": "View A is strict Top-Down (Map view), View B is 45-degree Oblique side view."},
    {"type": "Wide-Telephoto", "desc": "View A is Wide Angle covering whole scene, View B is Telephoto focused on the intersection."},
    {"type": "Sequential-Linear", "desc": "View A covers the start of the block, View B covers the end. Objects move A -> B."},
    {"type": "Occluded-Complementary", "desc": "View A blocked by trees on left, View B blocked by building on right. Complementary vision."},
    {"type": "Split-Lane", "desc": "View A focuses on Left-Turn lane, View B focuses on Straight/Right lane."}
]

TRAFFIC_DENSITY = [
    "Sparse (Free Flow)", 
    "Medium (Steady Flow)", 
    "High (Platoon Movement)",
    "Congested (Stop-and-Go)", 
    "Gridlock (Stationary)", 
    "Heavy Vehicle Dominant (High Occlusion)",
    "Cycle/Pedestrian Heavy (Small Objects)", 
    "Asymmetric Flow (One side busy, one empty)"
]

# ================= 3. Prompt Builder (MOC-optimized) =================

def build_prompt(params):
    scenario = params['scenario']
    camera = params['camera']
    density = params['density']
    duration = random.randint(40, 60)
    
    return f"""
### Role
You are the **Lead Data Simulator for the MOC (Multi-view Object Counting) Benchmark**.
Your task is to generate high-difficulty synthetic data that tests if an AI can correctly count objects across synchronized videos, handling duplicates and occlusions.

### Scenario Settings
* **Scene**: {scenario}
* **Traffic Density**: {density}
* **Camera Configuration**: {camera['desc']}
* **Total Duration**: {duration} seconds.
* **Sampling Rate**: Detailed description every 2 seconds.

### 🛑 CRITICAL INSTRUCTION: COUNTING COMPLEXITY & OCCLUSION
To make this challenging:
1.  **Occlusion Zones**: Define zones where objects vanish in one view (e.g., "Under the bridge in View A") but are visible in the other.
2.  **Cross-View Transition**: Objects must move from View A's field of view into View B's.
3.  **Distractors**: Include objects that SHOULD NOT be counted (e.g., "Count only Red Cars", but include Red Trucks and Blue Cars).
4.  **Deduplication**: {{A1}} and {{B1}} are the SAME physical car. The AI must count it as 1, not 2.

### Task 1: Generate Synchronized Video Scripts
Generate a timeline (0s to {duration}s) with a step of 2 seconds.
For EACH timestamp, provide:
* **View A Visual**: Describe visible objects ({{AX}}) and occlusions.
* **View B Visual**: Describe visible objects ({{BX}}).
* **Tracking Note**: Briefly note which objects are currently duplicates (e.g., "{{A1}} is now {{B1}}").

### Task 2: Generate 1 Complex MOC Question
Create ONE single-choice question focused on **Object Counting**.
* **Type**: Global Unique Count or Conditional Count.
* **Example**: "How many UNIQUE red sedans pass through the intersection in total?" (Requires de-duplication).
* **Example**: "How many pedestrians are visible in View B while the bus blocks View A?"

### Output JSON Format
{{
  "id": {params['id']},
  "meta": {{
      "scenario": "{scenario}",
      "camera_config": "{camera['type']}",
      "counting_logic": "Explain how duplicates are handled..."
  }},
  "video_scripts": [
      {{
          "timestamp": 0,
          "view_a": "Detailed description...",
          "view_b": "Detailed description..."
      }},
      ... (Continue every 2s until {duration}s)
  ],
  "question": {{
      "type": "MOC",
      "text": "How many unique Red Sedans appear in total?",
      "options": ["A. 3", "B. 4", "C. 5", "D. 6"],
      "correct_answer": "B",
      "cot_reasoning": "{{A1}} is {{B1}} (Count 1). {{A2}} is only in A (Count 2)... Total is 4."
  }}
}}
"""

# ================= 4. Task Generation Logic =================
def generate_tasks():
    print("🧮 Calculating combinations...")
    core_combinations = list(itertools.product(SCENARIO_TYPES, CAMERA_CONFIGS, TRAFFIC_DENSITY))
    random.shuffle(core_combinations)
    
    tasks = []
    for i in range(TARGET_TOTAL):
        combo = core_combinations[i % len(core_combinations)]
        tasks.append({
            "id": i,
            "scenario": combo[0],
            "camera": combo[1],
            "density": combo[2]
        })
    return tasks

# ================= 5. API Handler (aligned with position script) =================
def call_gemini(prompt):
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 8192
        }
    }
    
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}", "details": response.text}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def worker(task_params):
    prompt = build_prompt(task_params)
    task_id = task_params['id']
    
    for attempt in range(MAX_RETRIES):
        res = call_gemini(prompt)
        
        if "error" in res:
            print(f"⚠️ [Task {task_id}] Retry {attempt+1}. Error: {res['error']}")
            time.sleep(2)
            continue
            
        try:
            if "candidates" not in res:
                 print(f"❌ [Task {task_id}] No candidates.")
                 return {"status": "failed"}

            raw = res['candidates'][0]['content']['parts'][0]['text']
            clean = re.sub(r"```json|```", "", raw).strip()
            
            # JSON extraction (robust to extra text)
            start = clean.find('{')
            end = clean.rfind('}') + 1
            if start != -1 and end != 0:
                clean = clean[start:end]
                
            data = json.loads(clean)
            
            fname = f"moc_task_{task_id}.json"
            fpath = os.path.join(OUTPUT_DIR, fname)
            
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return {"status": "success", "id": task_id}
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"❌ [Task {task_id}] Parse Error: {e}")
            
    return {"status": "failed", "id": task_id}

# ================= 6. Main =================
if __name__ == "__main__":
    print(f"🚀 Starting MOC (Object Counting) Generation...")
    
    all_tasks = generate_tasks()
    print(f"📋 Total Tasks Scheduled: {len(all_tasks)}")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, t): t['id'] for t in all_tasks}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res['status'] == 'success':
                completed += 1
                if completed % 20 == 0:
                    print(f"✅ [MOC] Saved Task {res['id']} ({completed}/{TARGET_TOTAL})")
            else:
                print(f"❌ [MOC] FAILED Task {res['id']}")