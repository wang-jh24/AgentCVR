import requests
import json
import random
import re
import time
import os
import uuid
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

OUTPUT_DIR = "behavior_synthesis"
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "generation_errors.json")

MAX_WORKERS = 10      
TARGET_TOTAL = 1200   
MAX_RETRIES = 3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 2. Expansion Matrix (Subject -> Behaviors / Setting -> Activities) =================

# --- Animal behavior library (Subject -> Behaviors) ---
ANIMAL_DATA = {
    "Lion": ["Stalking prey", "Social grooming", "Roaring/Territorial", "Sleeping/Resting", "Cub carrying"],
    "Eagle": ["Diving for fish", "Nest maintenance", "Soaring/Patrolling", "Feeding chicks", "Mating display"],
    "Chimpanzee": ["Using tools (nuts/termites)", "Grooming session", "Aggressive display", "Climbing/Swinging", "Foraging on ground"],
    "Penguin": ["Huddling for warmth", "Sliding on ice", "Feeding young", "Swimming/Hunting", "waddling inland"],
    "Wolf": ["Pack hunting", "Howling", "Puppy play", "Sleeping in den", "Dominance display"],
    "Elephant": ["Dust bathing", "Using trunk for water", "Protecting calf", "Knocking down trees", "Greeting ceremony"],
    "Octopus": ["Camouflage changing", "Opening shell", "Jet propulsion escape", "Crawling on reef", "Hiding in crevice"],
    "Spider": ["Web weaving", "Wrapping prey", "Mating dance", "Waiting in ambush", "Carrying egg sac"],
    "Kangaroo": ["Boxing/Fighting", "Grazing", "Hopping transit", "Pouch care", "Resting in shade"],
    "Dolphin": ["Pod hunting", "Breaching/Jumping", "Echo-location scanning", "Playing with seaweed", "Social rubbing"],
    "Bear": ["Salmon fishing", "Tree scratching", "Hibernation prep", "Cub defense", "Berry foraging"],
    "Snake": ["Striking prey", "Constricting", "Shedding skin", "Sun basking", "Slithering/Climbing"],
    "Frog": ["Croaking/Calling", "Catching fly", "Leaping", "Swimming", "Camouflage motionless"],
    "Peacock": ["Fan display", "Shaking feathers", "Roosting in tree", "Foraging seeds", "Preening"],
    "Meerkat": ["Sentry duty (standing)", "Digging burrow", "Group foraging", "Sleeping pile", "Alarm calling"],
    "Crocodile": ["Death roll", "Sun basking (mouth open)", "Ambush lunging", "Guarding nest", "Swimming quietly"],
    "Bee": ["Pollinating flower", "Waggle dance", "Hive defense", "Grooming antennae", "Collecting water"],
    "Crab": ["Scuttling sideways", "Burrow digging", "Fighting with claws", "Filter feeding", "Molting shell"],
    "Owl": ["Head turning scan", "Silent flight", "Capturing rodent", "Regurgitating pellet", "Camouflage sleeping"],
    "Horse": ["Galloping", "Grazing", "Social grooming", "Bucking/Kicking", "Rolling in dirt"]
}

# --- Human activity library (Setting -> Activities) ---
HUMAN_DATA = {
    "Kitchen": ["Making a sandwich", "Washing dishes", "Chopping vegetables", "Brewing coffee", "Cleaning spill"],
    "Bedroom": ["Making the bed", "Folding laundry", "Reading a book", "Dressing/Undressing", "Packing a suitcase"],
    "Living Room": ["Vacuuming", "Searching for lost remote", "Doing yoga", "Eating pizza on sofa", "Playing VR game"],
    "Office": ["Fixing a paper jam", "Video conferencing", "Organizing files", "Writing on whiteboard", "Changing lightbulb"],
    "Gym": ["Lifting weights", "Running on treadmill", "Stretching", "Boxing training", "Drinking water"],
    "Garden": ["Mowing lawn", "Planting flowers", "Raking leaves", "Watering plants", "Pruning bushes"],
    "Supermarket": ["Checking fruit ripeness", "Pushing cart", "Scanning items", "Reaching high shelf", "Bagging groceries"],
    "Garage": ["Changing tire", "Woodworking", "Washing car", "Organizing tools", "Painting wall"],
    "Library": ["Shelving books", "Reading quietly", "Searching computer", "Whispering/Talking", "Studying with notes"],
    "Restaurant": ["Ordering food", "Eating with chopsticks", "Paying bill", "Spilling drink", "Waitering/Serving"],
    "Laboratory": ["Pipetting liquid", "Looking in microscope", "Writing notes", "Mixing chemicals", "Cleaning equipment"],
    "Park": ["Jogging", "Walking dog", "Throwing frisbee", "Feeding ducks", "Picnicking"],
    "Bathroom": ["Brushing teeth", "Washing face", "Cleaning mirror", "Applying makeup", "Shaving"],
    "Construction Site": ["Hammering nail", "Reviewing blueprints", "Carrying lumber", "Mixing cement", "Digging with shovel"],
    "Art Studio": ["Painting canvas", "Sculpting clay", "Sketching", "Cleaning brushes", "Framing art"],
    "Music Room": ["Tuning guitar", "Playing piano", "Setting up mic", "Drumming", "Reading sheet music"],
    "Subway": ["Checking phone", "Sleeping", "Reading paper", "Looking at map", "Listening to music"],
    "Beach": ["Applying sunscreen", "Building sandcastle", "Surfing", "Collecting shells", "Reading under umbrella"],
    "Camping": ["Setting up tent", "Starting fire", "Cooking on stove", "Chopping wood", "Unrolling sleeping bag"],
    "Airport": ["Checking in", "Going through security", "Pulling luggage", "Looking at schedule", "Sleeping on chair"]
}

# --- Context modifiers for diversity ---
CONTEXT_MODIFIERS = [
    "Standard/Calm", 
    "Urgent/Rushed", 
    "Failed/Clumsy Attempt", 
    "Aggressive/Intense", 
    "Tired/Lazy", 
    "Group/Social Interaction"
]

# ================= 3. Task Generation and Queue =================

def generate_task_queue(target_count):
    """Generate unique task configs via full permutation."""
    print("🧮 Calculating Matrix Combinations...")
    all_tasks = []
    
    for subject, behaviors in ANIMAL_DATA.items():
        for behavior in behaviors:
            for context in CONTEXT_MODIFIERS:
                all_tasks.append({
                    "type": "Animal",
                    "subject": subject,
                    "target": behavior,
                    "context": context,
                    "distractors": [b for b in behaviors if b != behavior]
                })
                
    for setting, activities in HUMAN_DATA.items():
        for activity in activities:
            for context in CONTEXT_MODIFIERS:
                all_tasks.append({
                    "type": "Human",
                    "subject": setting,
                    "target": activity,
                    "context": context,
                    "distractors": [a for a in activities if a != activity]
                })
    
    print(f"   Total Unique Matrix Combinations: {len(all_tasks)}")
    
    random.shuffle(all_tasks)
    final_tasks = all_tasks[:target_count]
    
    for i, task in enumerate(final_tasks):
        task["id"] = i
        task["num_videos"] = 4
        
        # Balance: 50% single-answer, 50% multi-answer
        if i % 2 == 0:
            task["num_correct"] = 1
        else:
            task["num_correct"] = random.choice([2, 3])
                
    return final_tasks

def generate_video_structure(task_type):
    if task_type == "Animal":
        total_duration = round(random.uniform(3.0, 20.0), 2)
        num_segments = random.choice([1, 2])
    else:
        total_duration = round(random.uniform(30.0, 60.0), 2)
        min_segs = int(total_duration // 12) + 1
        max_segs = int(total_duration // 6)
        num_segments = random.randint(min(min_segs, max_segs), max(min_segs, max_segs) + 2)

    timeline = []
    cursor = 0.0
    avg_len = total_duration / num_segments
    
    for i in range(num_segments):
        if i == num_segments - 1:
            end = total_duration
        else:
            variance = random.uniform(-2.0, 2.0)
            end = min(cursor + avg_len + variance, total_duration - 1.0)
            if end <= cursor: end = cursor + 1.0
        timeline.append({"start": round(cursor, 2), "end": round(end, 2)})
        cursor = end
        
    return {"duration": total_duration, "timeline_slots": timeline}

# ================= 4. Prompt Constructor (matrix-driven) =================

def create_matrix_prompt(task, video_structures):
    if task["type"] == "Animal":
        context_str = f"**Subject**: {task['subject']}\n**Target Behavior**: \"{task['target']}\"\n**Context/Nuance**: \"{task['context']}\" (Apply this mood/style to the target)\n**Distractor Candidates**: {', '.join(task['distractors'])}"
    else:
        context_str = f"**Setting**: {task['subject']}\n**Target Activity**: \"{task['target']}\"\n**Context/Nuance**: \"{task['context']}\" (Apply this mood/style to the target)\n**Distractor Candidates**: {', '.join(task['distractors'])}"

    timeline_instructions = ""
    for i, struct in enumerate(video_structures):
        label = f"Video {i+1}"
        timeline_instructions += f"\n**{label} Structure ({struct['duration']}s)**:\n"
        for j, slot in enumerate(struct['timeline_slots']):
            timeline_instructions += f"   - Slot {j+1}: {slot['start']}s - {slot['end']}s\n"

    return f"""
### Role
You are an **Expert Examiner** creating a Behavior Understanding dataset.

### Task Specs
* **Type**: {task['type']}
* **Format**: 4 Options (A, B, C, D)
* **Goal**: Test understanding of **Intent**, **Purpose**, and **Nuance**.
{context_str}

### Step 1: Scenario Expansion
1.  **Target Scenario**: Describe the "{task['target']}" specifically performed in a **{task['context']}** manner.
    * *Example*: If target="Walking dog" and context="Rushed", scenario="Person checking watch, pulling leash, running slightly."
2.  **Distractor Scenarios**: Choose behaviors from the candidate list that fit the same Subject/Setting but imply a **different intent**.

### Step 2: Formulate Question (Intent-Based)
Write a concise question focusing on the **Specific Intent** or **Nuance**.
* **BAD**: "Which video shows the lion sleeping?"
* **GOOD**: "Which video depicts a lion **resting due to exhaustion** rather than just sleeping?" (Reflecting the context)
* **GOOD**: "Which video shows an **urgent attempt** to fix a paper jam?"

### Step 3: Script Generation
Generate 4 videos (Video 1, 2, 3, 4). Options A-D correspond to these videos.
* **Exactly {task['num_correct']} Videos** must depict the Target Scenario.
* **{4 - task['num_correct']} Videos** must be Distractors.
{timeline_instructions}

### Requirements
* **Visual**: Dense description (Character, Action, Environment).
* **No Captions**.

### Output JSON Format
{{
  "expanded_scenario": "...",
  "question": "...",
  "correct_video_indices": [0, ...], 
  "videos_content": [
      {{ "video_label": "Video 1", "timeline": [ {{ "visual": "..." }}, ... ] }},
      ... (Total 4)
  ]
}}
"""

# ================= 5. API Request =================

def request_gemini(prompt):
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.9, "maxOutputTokens": 65535}}
    try:
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=240)
        if resp.status_code == 200: return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
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

# ================= 6. Processing Logic =================

def process_and_save_task(task_params):
    video_structures = [generate_video_structure(task_params["type"]) for _ in range(task_params["num_videos"])]
    prompt = create_matrix_prompt(task_params, video_structures)
    
    for attempt in range(MAX_RETRIES):
        try:
            raw_text = request_gemini(prompt)
            data = extract_json(raw_text)
            
            if data and "videos_content" in data:
                llm_videos = data["videos_content"]
                if len(llm_videos) != 4: continue
                
                formatted_videos = {}
                durations_list = []
                
                for idx, (struct, llm_vid) in enumerate(zip(video_structures, llm_videos)):
                    key = str(idx + 1)
                    pre_calc_slots = struct['timeline_slots']
                    llm_slots = llm_vid.get("timeline", [])
                    valid_len = min(len(pre_calc_slots), len(llm_slots))
                    
                    merged_timeline = []
                    for k in range(valid_len):
                        merged_timeline.append({
                            "start": pre_calc_slots[k]["start"],
                            "end": pre_calc_slots[k]["end"],
                            "visual": llm_slots[k].get("visual", "")
                        })
                    
                    formatted_videos[key] = {
                        "duration": struct["duration"],
                        "timeline": merged_timeline
                    }
                    durations_list.append(struct["duration"])

                letter_map = ["A", "B", "C", "D"]
                correct_indices = data.get("correct_video_indices", [])
                correct_letters = [letter_map[i] for i in correct_indices if i < 4]
                if not correct_letters:
                    continue

                final_json = {
                    "id": task_params["id"],
                    "meta": {
                        "task_type": task_params["type"],
                        "subject": task_params["subject"],
                        "target_action": task_params["target"],
                        "nuance_context": task_params["context"],
                        "expanded_scenario": data.get("expanded_scenario")
                    },
                    "question": data["question"],
                    "options": ["A. Video 1", "B. Video 2", "C. Video 3", "D. Video 4"],
                    "answer": sorted(correct_letters),
                    "correct": True,
                    "quality": "Generated",
                    "duration": durations_list,
                    "videos": formatted_videos
                }
                
                file_name = f"task_{task_params['id']}.json"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, indent=2, ensure_ascii=False)
                    
                return {"status": "success", "id": task_params["id"]}
        except: pass
        time.sleep(random.uniform(1, 2))
        
    return {"status": "failed", "id": task_params["id"], "error": "Max retries"}

# ================= 7. Main =================

def main():
    print(f"🚀 Starting Matrix-Based Behavior Generation...")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    all_unique_tasks = generate_task_queue(TARGET_TOTAL)
    print(f"📋 Generated {len(all_unique_tasks)} unique task configs.")
    
    failed_tasks = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_and_save_task, t): t['id'] for t in all_unique_tasks}
        
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            completed += 1
            if result['status'] == 'success':
                if completed % 20 == 0:
                    print(f"[{completed}/{TARGET_TOTAL}] ✅ Saved task_{result['id']}.json")
            else:
                print(f"[{completed}/{TARGET_TOTAL}] ❌ FAILED Task {result['id']}")
                failed_tasks.append(result)

    print(f"\n✨ Done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()