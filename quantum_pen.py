import os
import json
import redis
import time
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv


# Load environment variables from .env file if present
load_dotenv()


# --- 1. CONFIGURATION ---
# ========================

# OpenRouter API Configuration
# 将你的OpenRouter API Key放在环境变量中，或者直接在这里赋值
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# For attribution on OpenRouter
HTTP_REFERER = "https://github.com/mountain/quantum-pen"  # 可以换成你的项目地址
SITE_NAME = "Quantum Pen Project"

# Model Selection for each role
# 你可以根据模型的特性和成本，为不同角色选择不同模型
DIRECTOR_MODEL = "openai/gpt-5"
WRITER_MODEL = "anthropic/gemini-pro-2.5"
EVALUATOR_MODEL = "google/gemini-pro-1.5"

# System Parameters
TEXT_POOL_SIZE = 3
DIRECTOR_BRANCH_FACTOR = 3  # 3 -> 9
WRITER_BRANCH_FACTOR = 3  # 9 -> 27
EVALUATION_DIMENSIONS = [
    "PlotAdvancement", "CharacterDevelopment", "TensionAndPacing",
    "ProseAndStyle", "Coherence"
]

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# File Storage
OUTPUT_DIR = "story_progress"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. CORE PROMPTS ---
# =======================

# 经过仔细斟酌的Prompt，要求以JSON格式输出，便于程序解析

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
{
  "briefs": [
    {
      "brief_id": "brief_1",
      "goal": "Primary objective for this chapter.",
      "pacing_and_atmosphere": "Describe the desired pacing (e.g., slow, tense, fast-paced) and atmosphere (e.g., melancholic, mysterious).",
      "key_plot_points": ["A list of 2-3 essential events or reveals that must happen."],
      "character_focus": "Which character's perspective or development is central?",
      "creative_constraints": "Any specific stylistic notes or things to avoid."
    },
    // ... two more briefs ...
  ]
}
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
{
  "chapter_text": "The full text of the new chapter..."
}
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
{
  "evaluations": [
    {"dimension": "PlotAdvancement", "score": <int>, "justification": "<string>"},
    {"dimension": "CharacterDevelopment", "score": <int>, "justification": "<string>"},
    {"dimension": "TensionAndPacing", "score": <int>, "justification": "<string>"},
    {"dimension": "ProseAndStyle", "score": <int>, "justification": "<string>"},
    {"dimension": "Coherence", "score": <int>, "justification": "<string>"}
  ]
}
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
            temperature=0.8,  # Higher temperature for creative tasks
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"  ! API Call Error: {e}")
        return None


def save_text_pool(cycle: int, text_pool: List[Dict[str, Any]]):
    """Saves the current text pool to local files."""
    for i, item in enumerate(text_pool):
        filename = os.path.join(OUTPUT_DIR, f"cycle_{cycle:02d}_pool_{i}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(item['full_text'])
        print(f"  > Saved text pool item {i} to {filename}")


# --- 4. CORE LOGIC PHASES ---
# ============================

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
            # Add parent context to each brief for the writer
            for brief in response_data['briefs']:
                brief['parent_text_id'] = parent_text['id']
                brief['parent_full_text'] = parent_text['full_text']
            all_briefs.extend(response_data['briefs'])
    print(f"  Generated {len(all_briefs)} briefs.")
    return all_briefs


def run_writer_phase(briefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running Writer Phase ---")
    candidates = []
    for i, brief in enumerate(briefs):
        # We aim for 27 candidates from 9 briefs
        for j in range(WRITER_BRANCH_FACTOR):
            print(f"  Generating candidate {i * WRITER_BRANCH_FACTOR + j + 1}/27 from brief {i + 1}...")
            prompt = WRITER_PROMPT_TEMPLATE.format(
                brief=json.dumps(brief, indent=2),
                story_context=brief['parent_full_text']
            )
            response_data = call_openrouter(prompt, WRITER_MODEL, "You are a novelist writing a chapter in JSON.")
            if response_data and 'chapter_text' in response_data:
                candidate = {
                    'id': f"candidate_{i * WRITER_BRANCH_FACTOR + j}",
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
            # Calculate composite score
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

    # "Exploitation": Get the top 2 based on composite score
    scored_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    top_2 = scored_candidates[:2]

    # "Exploration": Find the most promising "potential" one from the rest
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
    if best_potential:
        # Ensure the potential candidate is not already in the top 2
        if best_potential['id'] not in [c['id'] for c in top_2]:
            new_pool_candidates.append(best_potential)

    # In case there are fewer than 3 candidates, handle gracefully
    while len(new_pool_candidates) < TEXT_POOL_SIZE and len(scored_candidates) > len(new_pool_candidates):
        new_pool_candidates.append(scored_candidates[len(new_pool_candidates)])

    # Prepare the final pool for the next cycle
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
    print("=== Writer Initializing ===")

    # --- Initial Setup (Cycle 0) ---
    try:
        r.ping()
        print("Redis connection successful.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}")
        print("Please ensure Redis is running on redis://{REDIS_HOST}:{REDIS_PORT}")
        return

    # Check if we need to initialize or continue
    if not r.exists('current_cycle'):
        print("No previous state found in Redis. Initializing system.")
        r.set('current_cycle', 0)

        # Create a seed story to start the process
        initial_text = {
            'id': 'cycle_0_pool_0',
            'full_text': "The old clockmaker, Alistair, wiped the grease from his hands. For fifty years, he had tended to the town's timepieces, but none was as important as the one before him: a strange, star-shaped clock recovered from a fallen meteorite. It didn't tick; it hummed, and the hum was getting stronger.",
        }
        # To start with a full pool of 3, we just duplicate the initial text.
        initial_pool = [initial_text.copy() for _ in range(TEXT_POOL_SIZE)]
        for i, item in enumerate(initial_pool):
            item['id'] = f'cycle_0_pool_{i}'

        r.set('text_pool', json.dumps(initial_pool))
        save_text_pool(0, initial_pool)

    # --- Main Loop ---
    current_cycle = int(r.get('current_cycle'))
    NUM_CYCLES_TO_RUN = 3  # Define how many cycles to run in this session

    for i in range(NUM_CYCLES_TO_RUN):
        cycle_num = current_cycle + i + 1
        print(f"\n\n>>>>>>>>>> STARTING CYCLE {cycle_num} <<<<<<<<<<")

        # Load current state from Redis
        text_pool = json.loads(r.get('text_pool'))

        # Author provides their intent for this cycle
        author_intent = "Deepen the mystery of the clock. Introduce a character who is also interested in it, creating a sense of competition or threat."

        # 1. Director Phase
        briefs = run_director_phase(text_pool, author_intent)
        if not briefs or len(briefs) < TEXT_POOL_SIZE * DIRECTOR_BRANCH_FACTOR:
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

        # 5. Update State
        print("\n--- Updating State for Next Cycle ---")
        r.set('current_cycle', cycle_num)
        r.set('text_pool', json.dumps(new_text_pool))
        save_text_pool(cycle_num, new_text_pool)

        print(f">>>>>>>>>> COMPLETED CYCLE {cycle_num} <<<<<<<<<<")
        # Add a small delay to avoid hitting API rate limits
        time.sleep(5)

    print("\n=== Session Finished ===")


if __name__ == "__main__":
    main()
