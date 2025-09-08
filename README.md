# Quantum Pen

**Stories are not written, but discovered.**

`Quantum Pen` is an experimental AI narrative co-creation engine. It transforms the solitary, linear path of writing into a thrilling expedition into the narrative multiverse. Here, every authorial "intent" is cast into a quantum field, instantly branching into dozens of parallel story realities.

You, the author, are the sole **Observer** in this narrative universe. Your choice is the act of measurement that **collapses the wave function** of fiction, turning one of countless possibilities into the one true canon.

This isn't just writing. It's navigating the narrative wave function.

## Core Concepts

The workflow of `Quantum Pen` is a fusion of several key concepts:

1.  **Evolutionary Algorithm:** The story itself is treated as a population. In each cycle, the fittest narrative paths survive and propagate through a process of "reproduction" (generating new chapters), "mutation" (diverse creative briefs), and "selection" (evaluation and filtering).
2.  **AI Agent Ensemble:** The system employs three specialized AI agents working in concert:
      * **The Director:** The strategist. Based on the author's intent and the current story, it generates multiple forward-looking creative briefs.
      * **The Writer:** The novelist. It translates the Director's briefs into well-crafted, prose-rich chapters.
      * **The Evaluator:** The critic. It quantitatively scores the Writers' outputs across multiple literary dimensions, providing the basis for natural selection.
3.  **Divergence & Convergence:** Each creative cycle is a breath of creative expansion and contraction:
      * **Divergence:** `3` parent texts → `9` director's briefs → `27` candidate chapters, maximizing the exploration of creative possibilities.
      * **Convergence:** From `27` candidates, a `2+1` elite selection strategy filters the pool down to `3`, ensuring the story evolves towards a high-quality, coherent path.

## How It Works

The system operates in iterative "cycles." Here’s a breakdown of a single cycle:

```
[ Author's Intent ] -> [ 3 Parent Texts ] -> (Director) -> [ 9 Briefs ] -> (Writer) -> [ 27 Candidates ] -> (Evaluator) -> [ 27 Scored Candidates ] -> (Selection) -> [ 3 New Parent Texts ] -> (Loop)
```

1.  **Input:** The author defines the creative goal for the cycle in `intention.md`.
2.  **Director Phase:** The Director reads the `3` texts from the current "text pool" and generates `9` unique creative briefs.
3.  **Writer Phase:** The Writer AI takes the `9` briefs and generates `27` new candidate chapters.
4.  **Evaluator Phase:** The Evaluator AI scores all `27` candidates based on predefined literary dimensions.
5.  **Selection Phase:** A `2+1` algorithm selects the next generation:
      * **Exploitation:** The top `2` highest-scoring candidates are chosen to preserve the strongest narrative threads.
      * **Exploration:** `1` "promising" candidate—one that may not have the highest average score but excels in a key dimension (like creativity)—is also selected to prevent creative stagnation.
6.  **Update Phase:** These `3` chosen candidates form the new text pool for the next cycle. Their full text is saved to the `story_progress/` directory.

## Getting Started

Follow these steps to begin your first co-creation session.

### 1\. Prerequisites

*   Python 3.12+
*   A running Redis instance (e.g., `redis-server` on your local machine).

### 2\. Create Your Writing Project

First, create a dedicated folder for your story. This will keep your narrative files organized and separate from the tool's installation.

```bash
mkdir my-awesome-story
cd my-awesome-story
```

### 3\. Install Quantum Pen

Install the engine using pip. It's recommended to use a virtual environment for your project.

```bash
pip install quantum-pen
```

### 4\. Initialize Your Story

Inside your project folder (`my-awesome-story`), create three essential files:

**1\. `.env`**

This file holds your API key. `Quantum Pen` uses OpenRouter to access various language models.

```
OPENROUTER_API_KEY="sk-or-your-key-here"
```

**2\. `starter.md`**

This is the seed of your story—the opening paragraph or chapter.

```markdown
# The Chronos Key

The antique shop smelled of dust and forgotten time, a scent Elias knew better than his own name. He was an appraiser of histories, a man who could read the soul of an object from the scratches on its surface. But the device that lay on the velvet cloth before him was silent. It was a pocket watch crafted from a metal that shimmered like captured starlight, its face a complex astrolabe of unknown constellations. It had no hands to tell the time, only a single, keyhole-shaped aperture at its center.
```

**3\. `intention.md`**

This file directs the AI for the next creative cycle. State your goal for the upcoming chapter.

```markdown
Deepen the mystery of the watch. Introduce a character who is also interested in it, creating a sense of competition or threat. The atmosphere should be tense and filled with suspicion.
```

### 5\. Run the Engine

With your Redis server running, you're ready to start. From inside your project folder, run the command:

```bash
qp
```

The engine will start its first cycle. A new `story_progress/` directory will be created to store the generated chapters. To continue the story, simply update `intention.md` with your new goal and run `qp` again.

## Your Writing Project Structure

After running the engine, your project folder will look like this:

```
my-awesome-story/
├── .env                # Your API key
├── starter.md          # The story's starting point
├── intention.md        # Your creative goal for the next cycle
└── story_progress/     # (Auto-generated) Stores the output of each cycle
    ├── cycle_01_cand_01.md
    ├── cycle_01_cand_02.md
    └── ...
```

## Customization

You can easily customize the engine by editing the `CONFIGURATION` section in `quantum_pen.py`:

  * **Change Models:** Swap out `DIRECTOR_MODEL`, `WRITER_MODEL`, and `EVALUATOR_MODEL` with any supported model on OpenRouter.
  * **Adjust Parameters:** Modify `TEXT_POOL_SIZE` or `WRITER_BRANCH_FACTOR` to change the scale of the evolutionary process.
  * **Refine Evaluation:** Add or change dimensions in `EVALUATION_DIMENSIONS` and update the `EVALUATOR_PROMPT_TEMPLATE` to match, fine-tuning the selection criteria.

## License

This project is licensed under the MIT License.