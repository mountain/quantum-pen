import os
import json
import redis
from openai import OpenAI
from typing import List, Dict, Any
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

def run_director_phase(text_pool: List[Dict[str, Any]], author_intent: str) -> List[Dict[str, Any]]:
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
    return all_briefs


def run_writer_phase(briefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running Writer Phase ---")
    candidates = []
    # This loop structure results in 27 calls, which is the "direct" method.
    # For clarity and robustness, this is often better than asking for 3 variants in one call.
    for i in range(len(briefs)):
        brief = briefs[i]
        print(f"  Generating candidate {i + 1}/{len(briefs)} from brief {i + 1}...")
        prompt = WRITER_PROMPT_TEMPLATE.format(
            brief=json.dumps(brief, indent=2),
            story_context=brief['parent_full_text']
        )
        # To get 27 candidates, we can either call this 27 times,
        # or call 9 times and ask for 3 variants. Let's stick to a clear 1:1 mapping for now.
        # The prompt below is for a single candidate generation.
        # To make it 27, the calling loop in main should handle it. Let's simplify and assume 9 briefs -> 9 candidates for now.
        # No, let's keep the 27 logic. The simplest way is to just loop through the briefs 3 times.
    # The prompt as written is fine for 27 candidates from 9 briefs; it needs to be called 27 times. Let's adjust the loop.
    # The original loop was incorrect for generating 27, fixing it.

    generated_count = 0
    for i, brief in enumerate(briefs):
        for j in range(WRITER_BRANCH_FACTOR):
            generated_count += 1
            print(f"  Generating candidate {generated_count}/27 from brief {i + 1}...")
            prompt = WRITER_PROMPT_TEMPLATE.format(
                brief=json.dumps(brief, indent=2),
                story_context=brief['parent_full_text']
            )
            response_data = call_openrouter(prompt, WRITER_MODEL, "You are a novelist writing a chapter in JSON.")
            if response_data and 'chapter_text' in response_data:
                candidate = {
                    'id': f"candidate_{generated_count - 1}",
                    'brief': brief,
                    'chapter_text': response_data['chapter_text'],
                    'parent_full_text': brief['parent_full_text']
                }
                candidates.append(candidate)

    print(f"  Generated {len(candidates)} candidates.")
    return candidates


def run_evaluator_phase(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    return scored_candidates


def run_selection_phase(scored_candidates: List[Dict[str, Any]], next_cycle: int) -> List[Dict[str, Any]]:
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
            'id': f"cycle_{next_cycle}_pool_{i}",
            'full_text': full_text,
            'source_candidate': candidate['id'],
            'composite_score': candidate['composite_score']
        })

    print(f"  Selected {len(next_text_pool)} candidates for the next text pool.")
    for item in next_text_pool:
        print(f"    - ID: {item['id']}, Score: {item['composite_score']:.2f}")

    return next_text_pool


# --- 5. MAIN EXECUTION LOOP ---
# ==============================

def main():
    print("=== Quantum Pen Initializing ===")

    try:
        r.ping()
        print("Redis connection successful.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}")
        print(f"Please ensure Redis is running on redis://{REDIS_HOST}:{REDIS_PORT}")
        return

    # --- Initial Setup ---
    # We now only ensure 'current_cycle' exists. File setup is in the loop.
    if not r.exists('current_cycle'):
        print("No previous state found in Redis. Initializing system.")
        r.set('current_cycle', 0)
        print("System initialized. Ready to start cycles.")

    # --- Main Loop ---
    NUM_CYCLES_TO_RUN = 3

    for i in range(NUM_CYCLES_TO_RUN):
        # Determine the current and next cycle numbers
        last_completed_cycle = int(r.get('current_cycle'))
        cycle_num = last_completed_cycle + 1

        print(f"\n\n>>>>>>>>>> STARTING CYCLE {cycle_num} <<<<<<<<<<")

        # --- Load all inputs for the cycle ---
        text_pool = []
        if last_completed_cycle == 0:
            # For the very first cycle, we load from the starter file.
            # This ensures any user edits to starter.md before starting are captured.
            print(f"Loading initial text from '{STARTER_FILE}' for Cycle 1.")
            if not os.path.exists(STARTER_FILE):
                print(f"\n[FATAL ERROR] Starter file '{STARTER_FILE}' not found.")
                print("Please create this file and add your initial story text to it.")
                return

            with open(STARTER_FILE, 'r', encoding='utf-8') as f:
                starter_text = f.read().strip()

            if not starter_text:
                print(f"\n[FATAL ERROR] Starter file '{STARTER_FILE}' is empty.")
                print("Please add your initial story text to the file.")
                return

            print("Successfully loaded initial story.")
            initial_text = {
                'id': 'cycle_0_pool_0',
                'full_text': starter_text,
            }
            text_pool = [initial_text.copy() for _ in range(TEXT_POOL_SIZE)]
            for j, item in enumerate(text_pool):
                item['id'] = f'cycle_0_pool_{j}'

            # Save this initial pool to cycle_00 files for transparency
            save_text_pool(0, text_pool)
        else:
            # For subsequent cycles, load from the previous cycle's output files.
            print(f"Loading text pool from files of cycle {last_completed_cycle}...")
            for pool_index in range(TEXT_POOL_SIZE):
                filename = os.path.join(OUTPUT_DIR, f"cycle_{last_completed_cycle:02d}_pool_{pool_index}.md")
                if not os.path.exists(filename):
                    print(f"\n[FATAL ERROR] Cannot find required file for next cycle: {filename}")
                    print("Aborting.")
                    return

                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()

                pool_item = {
                    'id': f'cycle_{last_completed_cycle}_pool_{pool_index}',
                    'full_text': content
                }
                text_pool.append(pool_item)
            print("Text pool loaded successfully from files.")

        # Author provides their intent for this cycle
        if os.path.exists(INTENT_FILE):
            with open(INTENT_FILE, 'r', encoding='utf-8') as f:
                author_intent = f.read().strip()
            if not author_intent:
                print(f"\n[WARNING] Intent file '{INTENT_FILE}' is empty. Using default intent.")
                author_intent = "Deepen the mystery. Introduce a character who is also interested in the central object, creating a sense of competition or threat."
            else:
                print(f"Loaded author's intent from '{INTENT_FILE}'.")
        else:
            author_intent = "Deepen the mystery. Introduce a character who is also interested in the central object, creating a sense of competition or threat."

        # 1. Director Phase
        briefs = run_director_phase(text_pool, author_intent)
        if not briefs or len(briefs) < DIRECTOR_BRANCH_FACTOR * TEXT_POOL_SIZE:
            print("! Director phase failed to produce enough briefs. Stopping cycle.")
            break

        # 2. Writer Phase
        candidates = run_writer_phase(briefs)
        if not candidates:
            print("! Writer phase failed to produce candidates. Stopping cycle.")
            break

        # 3. Evaluator Phase
        scored_candidates = run_evaluator_phase(candidates)

        # 4. Selection Phase
        new_text_pool = run_selection_phase(scored_candidates, cycle_num)
        if not new_text_pool:
            print("! Selection phase failed to produce a new pool. Stopping cycle.")
            break

        # 5. Update State and Wait for User
        print("\n--- Updating State for Next Cycle ---")
        save_text_pool(cycle_num, new_text_pool)
        r.set('current_cycle', cycle_num)

        print(f">>>>>>>>>> COMPLETED CYCLE {cycle_num} <<<<<<<<<<")

        # Add pause for user intervention ---
        print("\n✅ Cycle complete. New files have been saved to the 'story_progress/' directory.")
        print("   You can now review, compare, and edit the three new .md files before proceeding.")
        input("   Press Enter to start the next cycle...")

    print("\n=== Quantum Pen Session Finished ===")

if __name__ == "__main__":
    main()