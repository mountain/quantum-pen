import os
import json
import re
import redis
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- 1. CONFIGURATION ---
# ========================

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# For attribution on OpenRouter
HTTP_REFERER = "https://github.com/mountain/quantum-pen"
SITE_NAME = "Quantum Pen Project"
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", 10))


# Model Selection for each role
# Note: Provided model names were futuristic. Replaced with current top-tier available models.
# You can update these when new models are released.
DIRECTOR_MODEL = os.getenv("DIRECTOR_MODEL", "openai/gpt-4o")
WRITER_MODEL = os.getenv("WRITER_MODEL", "google/gemini-2.5-pro")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "google/gemini-2.5-pro")

# System Parameters
TEXT_POOL_SIZE = int(os.getenv("TEXT_POOL_SIZE", 3))
DIRECTOR_BRANCH_FACTOR = int(os.getenv("DIRECTOR_BRANCH_FACTOR", 3))  # 3 -> 9
WRITER_BRANCH_FACTOR = int(os.getenv("WRITER_BRANCH_FACTOR", 3))  # 9 -> 27
EVALUATION_DIMENSIONS = [
    "PlotAdvancement", "CharacterDevelopment", "TensionAndPacing",
    "ProseAndStyle", "Coherence"
]

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# File Storage
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "story_progress")
STARTER_FILE = os.getenv("STARTER_FILE", "starter.md")  # Initial story file
INTENT_FILE = os.getenv("INTENT_FILE", "intention.md")    # Author's intent file
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. CORE PROMPTS ---
# =======================

# [FIXED] Doubled the curly braces in JSON examples to escape them for the .format() method.

DIRECTOR_PROMPT_TEMPLATE = """
You are an expert literary director. Your task is to generate THREE distinct and creative briefs for the next chapter of a story, based on the author's intent and the story so far.

The briefs must be genuinely different to maximize creative exploration.

**Author's Intent:**
{author_intent}

**Story So Far:**
---
{story_context}
---

Generate a JSON object containing a list of exactly three briefs. The JSON schema should be:
{{
  "briefs": [
    {{
      "brief_id": "brief_1",
      "goal": "Primary objective for this chapter.",
      "pacing_and_atmosphere": "Describe the desired pacing (e.g., slow, tense, fast-paced) and atmosphere (e.g., melancholic, mysterious).",
      "key_plot_points": ["A list of 2-3 essential events or reveals that must happen."],
      "character_focus": "Which character's perspective or development is central?",
      "creative_constraints": "Any specific stylistic notes or things to avoid."
    }},
    {{
      "brief_id": "brief_2",
      "goal": "...",
      "pacing_and_atmosphere": "...",
      "key_plot_points": ["..."],
      "character_focus": "...",
      "creative_constraints": "..."
    }},
    {{
      "brief_id": "brief_3",
      "goal": "...",
      "pacing_and_atmosphere": "...",
      "key_plot_points": ["..."],
      "character_focus": "...",
      "creative_constraints": "..."
    }}
  ]
}}
"""

WRITER_PROMPT_TEMPLATE = """
You are a talented novelist. Your task is to write the next chapter of a story, faithfully following the creative brief provided. Your writing style must be consistent with the story so far.

**Creative Brief:**
---
{brief}
---

**Story So Far:**
---
{story_context}
---

Generate a JSON object containing the chapter text. The JSON schema should be:
{{
  "chapter_text": "The full text of the new chapter..."
}}
"""

EVALUATOR_PROMPT_TEMPLATE = """
You are a sharp and insightful literary critic. Your task is to evaluate a candidate chapter based on the story so far, across five specific dimensions. For each dimension, provide a score from 1 (poor) to 10 (excellent) and a concise justification.

**Evaluation Dimensions:**
1.  **PlotAdvancement:** Does the chapter meaningfully move the main plot forward?
2.  **CharacterDevelopment:** Are characters explored more deeply or do they show growth/change?
3.  **TensionAndPacing:** Is the rhythm and suspense effective for the story's goals?
4.  **ProseAndStyle:** Is the quality of the writing (word choice, sentence structure, imagery) high?
5.  **Coherence:** Does the chapter fit logically and tonally with the preceding text?

**Candidate Chapter:**
---
{candidate_text}
---

**Story So Far:**
---
{story_context}
---

Generate a JSON object containing your evaluation. The JSON schema should be:
{{
  "evaluations": [
    {{"dimension": "PlotAdvancement", "score": <int>, "justification": "<string>"}},
    {{"dimension": "CharacterDevelopment", "score": <int>, "justification": "<string>"}},
    {{"dimension": "TensionAndPacing", "score": <int>, "justification": "<string>"}},
    {{"dimension": "ProseAndStyle", "score": <int>, "justification": "<string>"}},
    {{"dimension": "Coherence", "score": <int>, "justification": "<string>"}}
  ]
}}
"""


# --- 3. API & HELPER FUNCTIONS ---
# =================================

# Initialize OpenRouter Client
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    default_headers={
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": SITE_NAME,
    },
    timeout=OPENAI_TIMEOUT,
)


def call_openrouter(prompt: str, model: str, system_message: str) -> Dict[str, Any]:
    """A robust function to call the OpenRouter API and parse JSON response."""
    print(f"  > Calling model: {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"  ! API Call Error: {e}")
        return None


def save_text_pool(cycle: int, text_pool: List[Dict[str, Any]]):
    """Saves the current text pool to local markdown files."""
    for i, item in enumerate(text_pool):
        filename = os.path.join(OUTPUT_DIR, f"cycle_{cycle:02d}_pool_{i}.md")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(item['full_text'])
        print(f"  > Saved text pool item {i} to {filename}")

# --- 4. CORE LOGIC PHASES ---
# ============================
# (Core logic functions remain unchanged)

def get_latest_cycle_from_story_progress() -> Optional[int]:
    """
    Scans the OUTPUT_DIR for story files and determines the latest cycle number.
    Filenames are expected in the format 'cycle_XX_...'.
    """
    if not os.path.exists(OUTPUT_DIR):
        return None

    latest_cycle = -1
    # Regex to find 'cycle_XX' and capture the number XX.
    cycle_regex = re.compile(r"cycle_(\d+)_")

    for filename in os.listdir(OUTPUT_DIR):
        match = cycle_regex.match(filename)
        if match:
            cycle_num = int(match.group(1))
            if cycle_num > latest_cycle:
                latest_cycle = cycle_num

    return latest_cycle if latest_cycle != -1 else None


def run_director_phase(cycle_num: int, text_pool: List[Dict[str, Any]], author_intent: str) -> List[Dict[str, Any]]:
    print("\n--- Running Director Phase ---")
    all_briefs = []
    for i, parent_text in enumerate(text_pool):
        print(f"  Generating briefs for pool item {i}...")
        prompt = DIRECTOR_PROMPT_TEMPLATE.format(
            author_intent=author_intent,
            story_context=parent_text['full_text']
        )
        response_data = call_openrouter(prompt, DIRECTOR_MODEL, "You are a creative director generating JSON.")
        if response_data and 'briefs' in response_data:
            for brief in response_data['briefs']:
                brief['parent_text_id'] = parent_text['id']
                brief['parent_full_text'] = parent_text['full_text']
            all_briefs.extend(response_data['briefs'])

    print(f"  Generated {len(all_briefs)} briefs.")
    if all_briefs:
        r.hset(f"cycle:{cycle_num}", "briefs", json.dumps(all_briefs))
        r.hset(f"cycle:{cycle_num}", "status", "director_complete")
        print(f"  > Saved {len(all_briefs)} briefs to Redis for cycle {cycle_num}.")
    return all_briefs


def run_writer_phase(cycle_num: int, briefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running Writer Phase ---")
    candidates_key = f"cycle:{cycle_num}:candidates"

    # Load already generated candidates to resume
    existing_candidates_json = r.lrange(candidates_key, 0, -1)
    candidates = [json.loads(c) for c in existing_candidates_json]
    num_existing = len(candidates)

    total_candidates_to_generate = len(briefs) * WRITER_BRANCH_FACTOR

    if num_existing > 0:
        print(f"  Found {num_existing} existing candidates. Resuming generation.")

    if num_existing >= total_candidates_to_generate:
        print("  All candidates already generated.")
        # Ensure status is set correctly even if we just resume
        r.hset(f"cycle:{cycle_num}", "status", "writer_complete")
        return candidates

    # Loop through all potential candidates and skip the ones we already have
    for i in range(len(briefs)):
        for j in range(WRITER_BRANCH_FACTOR):
            # This is the overall index of the candidate we are about to generate
            current_candidate_index = i * WRITER_BRANCH_FACTOR + j

            # If we have already generated this candidate, skip.
            if current_candidate_index < num_existing:
                continue

            # This is the "live" count for user display
            live_generated_count = current_candidate_index + 1
            brief = briefs[i]
            print(f"  Generating candidate {live_generated_count}/{total_candidates_to_generate} from brief {i + 1}...")

            prompt = WRITER_PROMPT_TEMPLATE.format(
                brief=json.dumps(brief, indent=2),
                story_context=brief['parent_full_text']
            )
            response_data = call_openrouter(prompt, WRITER_MODEL, "You are a novelist writing a chapter in JSON.")
            if response_data and 'chapter_text' in response_data:
                candidate = {
                    'id': f"candidate_{current_candidate_index}",
                    'brief': brief,
                    'chapter_text': response_data['chapter_text'],
                    'parent_full_text': brief['parent_full_text']
                }
                candidates.append(candidate)
                # Save each candidate to Redis incrementally
                r.rpush(candidates_key, json.dumps(candidate))

    print(f"  Generated a total of {len(candidates)} candidates.")
    if len(candidates) >= total_candidates_to_generate:
        r.hset(f"cycle:{cycle_num}", "status", "writer_complete")
        print(f"  > All {len(candidates)} candidates saved to Redis for cycle {cycle_num}.")
    return candidates


def run_evaluator_phase(cycle_num: int, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running Evaluator Phase ---")
    scored_candidates = []
    for i, candidate in enumerate(candidates):
        print(f"  Evaluating candidate {i + 1}/{len(candidates)}...")
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(
            candidate_text=candidate['chapter_text'],
            story_context=candidate['parent_full_text']
        )
        response_data = call_openrouter(prompt, EVALUATOR_MODEL,
                                        "You are a literary critic providing evaluation in JSON.")
        if response_data and 'evaluations' in response_data:
            candidate['evaluations'] = response_data['evaluations']
            scores = [e['score'] for e in response_data['evaluations']]
            candidate['composite_score'] = sum(scores) / len(scores) if scores else 0
            scored_candidates.append(candidate)

    print(f"  Evaluated {len(scored_candidates)} candidates.")
    if scored_candidates:
        r.hset(f"cycle:{cycle_num}", "scored_candidates", json.dumps(scored_candidates))
        r.hset(f"cycle:{cycle_num}", "status", "evaluator_complete")
        print(f"  > Saved {len(scored_candidates)} scored candidates to Redis for cycle {cycle_num}.")
    return scored_candidates


def run_selection_phase(cycle_num: int, scored_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running Selection Phase ---")
    if not scored_candidates:
        print("  ! No valid candidates to select from.")
        return []

    scored_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    top_2 = scored_candidates[:2]

    remaining_candidates = scored_candidates[2:]
    best_potential = None
    max_single_score = 0
    if remaining_candidates:
        for candidate in remaining_candidates:
            for evaluation in candidate.get('evaluations', []):
                if evaluation['score'] > max_single_score:
                    max_single_score = evaluation['score']
                    best_potential = candidate

    new_pool_candidates = top_2
    if best_potential and best_potential['id'] not in [c['id'] for c in top_2]:
        new_pool_candidates.append(best_potential)

    while len(new_pool_candidates) < TEXT_POOL_SIZE and len(scored_candidates) > len(new_pool_candidates):
        new_pool_candidates.append(scored_candidates[len(new_pool_candidates)])

    next_text_pool = []
    for i, candidate in enumerate(new_pool_candidates):
        full_text = candidate['parent_full_text'] + "\n\n" + candidate['chapter_text']
        next_text_pool.append({
            'id': f"cycle_{cycle_num}_pool_{i}",
            'full_text': full_text,
            'source_candidate': candidate['id'],
            'composite_score': candidate['composite_score']
        })

    print(f"  Selected {len(next_text_pool)} candidates for the next text pool.")
    for item in next_text_pool:
        print(f"    - ID: {item['id']}, Score: {item['composite_score']:.2f}")

    if next_text_pool:
        r.hset(f"cycle:{cycle_num}", "new_text_pool", json.dumps(next_text_pool))
        r.hset(f"cycle:{cycle_num}", "status", "selection_complete")
        print(f"  > Saved new text pool to Redis for cycle {cycle_num}.")

    return next_text_pool


# --- 5. MAIN EXECUTION LOOP ---
# ==============================

def interactive_setup():
    """
    Creates starter and intention files from examples if they don't exist.
    """
    starter_example = """# The Chronos Key

The antique shop smelled of dust and forgotten time, a scent Elias knew better than his own name. He was an appraiser of histories, a man who could read the soul of an object from the scratches on its surface. But the device that lay on the velvet cloth before him was silent. It was a pocket watch crafted from a metal that shimmered like captured starlight, its face a complex astrolabe of unknown constellations. It had no hands to tell the time, only a single, keyhole-shaped aperture at its center."""

    intention_example = "Deepen the mystery. Introduce a character who is also interested in the central object, creating a sense of competition or threat."

    print("--- Welcome to Quantum Pen! ---")
    print("It looks like this is your first time running in this folder.")
    print("I'm creating starter files for you from the examples.\n")

    with open(STARTER_FILE, 'w', encoding='utf-8') as f:
        f.write(starter_example)
    print(f"✅ Created '{STARTER_FILE}'. You can edit this file to change the beginning of your story.")

    with open(INTENT_FILE, 'w', encoding='utf-8') as f:
        f.write(intention_example)
    print(f"✅ Created '{INTENT_FILE}'. You can edit this file to change the initial creative goal.")

    print("\n--- Setup Complete! ---")
    print("\n[IMPORTANT] Next, you need to provide your API key.")
    print("1. Create a file named '.env' in this folder.")
    print("2. Inside '.env', add this line: OPENROUTER_API_KEY=\"sk-or-your-key-here\"")
    print("\nOnce your .env file is ready, and you've reviewed the starter files, run 'qp' again.")


def main():
    print("=== Quantum Pen Initializing ===")

    # --- Initial Setup: Check for starter files ---
    if not os.path.exists(STARTER_FILE):
        interactive_setup()
        return

    try:
        r.ping()
        print("Redis connection successful.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}")
        print(f"Please ensure Redis is running on redis://{REDIS_HOST}:{REDIS_PORT}")
        return

    # --- Redis State Initialization ---
    # Sync with the latest cycle from the story_progress folder
    latest_cycle_from_files = get_latest_cycle_from_story_progress()
    if latest_cycle_from_files is not None:
        print(f"Found latest cycle {latest_cycle_from_files} in '{OUTPUT_DIR}'. Syncing Redis.")
        r.set('current_cycle', latest_cycle_from_files)
    elif not r.exists('current_cycle'):
        print("No story progress found and no state in Redis. Initializing to cycle 0.")
        r.set('current_cycle', 0)
    else:
        print("Using existing state from Redis.")

    # Determine the current and next cycle numbers
    last_completed_cycle = int(r.get('current_cycle'))
    cycle_num = last_completed_cycle + 1

    print(f"\n\n>>>>>>>>>> STARTING/RESUMING CYCLE {cycle_num} <<<<<<<<<<")

    cycle_key = f"cycle:{cycle_num}"
    cycle_state = r.hgetall(cycle_key)
    status = cycle_state.get("status")

    # If cycle hasn't even started, initialize it.
    if not status:
        r.hset(cycle_key, "status", "pending")
        status = "pending"
        print(f"Initialized new state for cycle {cycle_num} in Redis.")

    print(f"  > Current cycle status: {status}")

    # --- Load static inputs for the cycle (text pool and intent) ---
    # This logic runs regardless of the state, as it's the input for the whole cycle.
    text_pool = []
    if last_completed_cycle == 0:
        print(f"Loading initial text from '{STARTER_FILE}' for Cycle 1.")
        if not os.path.exists(STARTER_FILE):
            print(f"\n[FATAL ERROR] Starter file '{STARTER_FILE}' not found.")
            return
        with open(STARTER_FILE, 'r', encoding='utf-8') as f:
            starter_text = f.read().strip()
        if not starter_text:
            print(f"\n[FATAL ERROR] Starter file '{STARTER_FILE}' is empty.")
            return
        print("Successfully loaded initial story.")
        initial_text = {'id': 'cycle_0_pool_0', 'full_text': starter_text}
        text_pool = [initial_text.copy() for _ in range(TEXT_POOL_SIZE)]
        for j, item in enumerate(text_pool):
            item['id'] = f'cycle_0_pool_{j}'
        save_text_pool(0, text_pool)
    else:
        print(f"Loading text pool from files of cycle {last_completed_cycle}...")
        for pool_index in range(TEXT_POOL_SIZE):
            filename = os.path.join(OUTPUT_DIR, f"cycle_{last_completed_cycle:02d}_pool_{pool_index}.md")
            if not os.path.exists(filename):
                print(f"\n[FATAL ERROR] Cannot find required file for next cycle: {filename}")
                return
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            pool_item = {'id': f'cycle_{last_completed_cycle}_pool_{pool_index}', 'full_text': content}
            text_pool.append(pool_item)
        print("Text pool loaded successfully from files.")

    if os.path.exists(INTENT_FILE):
        with open(INTENT_FILE, 'r', encoding='utf-8') as f:
            author_intent = f.read().strip()
        if not author_intent:
            print(f"\n[WARNING] Intent file '{INTENT_FILE}' is empty. Using default intent.")
            author_intent = "Deepen the mystery."
        else:
            print(f"Loaded author's intent from '{INTENT_FILE}'.")
    else:
        author_intent = "Deepen the mystery."

    # --- State Machine for Cycle Execution ---

    # PHASE 1: DIRECTOR
    if status == "pending":
        briefs = run_director_phase(cycle_num, text_pool, author_intent)
        if not briefs or len(briefs) < DIRECTOR_BRANCH_FACTOR * TEXT_POOL_SIZE:
            print("! Director phase failed. Stopping cycle.")
            return
        status = r.hget(cycle_key, "status") # Refresh status

    # PHASE 2: WRITER
    if status == "director_complete":
        cycle_state = r.hgetall(cycle_key)
        briefs = json.loads(cycle_state['briefs'])
        print("  > Loaded briefs from Redis, proceeding to Writer phase.")

        candidates = run_writer_phase(cycle_num, briefs)
        if not candidates:
            print("! Writer phase failed. Stopping cycle.")
            return
        status = r.hget(cycle_key, "status")

    # PHASE 3: EVALUATOR
    if status == "writer_complete":
        candidates_key = f"cycle:{cycle_num}:candidates"
        candidates = [json.loads(c) for c in r.lrange(candidates_key, 0, -1)]
        print(f"  > Loaded {len(candidates)} candidates from Redis, proceeding to Evaluator phase.")

        scored_candidates = run_evaluator_phase(cycle_num, candidates)
        if not scored_candidates:
            print("! Evaluator phase failed. Stopping cycle.")
            return
        status = r.hget(cycle_key, "status")

    # PHASE 4: SELECTION
    if status == "evaluator_complete":
        cycle_state = r.hgetall(cycle_key)
        scored_candidates = json.loads(cycle_state['scored_candidates'])
        print(f"  > Loaded {len(scored_candidates)} scored candidates from Redis, proceeding to Selection phase.")

        new_text_pool = run_selection_phase(cycle_num, scored_candidates)
        if not new_text_pool:
            print("! Selection phase failed. Stopping cycle.")
            return
        status = r.hget(cycle_key, "status")

    # PHASE 5: FINALIZE
    if status == "selection_complete":
        cycle_state = r.hgetall(cycle_key)
        new_text_pool = json.loads(cycle_state['new_text_pool'])

        print("\n--- Finalizing Cycle ---")
        save_text_pool(cycle_num, new_text_pool)
        r.set('current_cycle', cycle_num)
        r.hset(cycle_key, "status", "completed")

        print(f"\n>>>>>>>>>> COMPLETED CYCLE {cycle_num} <<<<<<<<<<")
        print("\n✅ Cycle complete. New files have been saved to the 'story_progress/' directory.")
        print("   Run the tool again to start the next cycle.")

    print("\n=== Quantum Pen Session Finished ===")

if __name__ == "__main__":
    main()