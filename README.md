# Long-Context Summarization — OpenEnv Environment

An RL environment that trains language models to **compress long documents into compact summaries**, then tests whether those summaries preserve enough information to answer questions. Reward signal drives the model toward concise but information-dense compression.

Inspired by [Cursor's self-summarization approach](https://cursor.com/blog/self-summarization).

---

## Why This Matters

Long-context models do not fail only because they miss retrieval. They also fail because they
compress the wrong details.

That matters in at least three common settings:

- **Coding assistants** need to read long specs, PRs, logs, stack traces, and multi-file code slices,
  then preserve exactly the facts needed to make or review a change.
- **General-purpose assistants** need to compress long policies, reports, transcripts, and research
  documents into a working memory that still supports reliable downstream reasoning.
- **Needle-in-the-middle scenarios** punish systems that remember the headline but lose the one buried
  detail that later determines the correct answer.

This benchmark focuses on that compression step directly. Instead of asking whether a model can read
a long document once, it asks whether the model can turn partial context into a compact memory artifact
that remains useful later.

---

## Core Idea

```
truncated document
       ↓
  [agent: summarize]
       ↓
  summary + question
       ↓
  [agent: answer]
       ↓
  reward = F1(answer, ground_truth)
```

The agent only sees a **truncated version** of the document. A good summary → correct answer → high reward. A lossy summary → wrong answer → low reward. Over many RL steps, the model learns to write summaries that preserve answer-critical information in as few words as possible.

---

## Real-World Scenarios

Although the current environment is document-based, the structure maps cleanly onto real agent workflows.

- **Spec to implementation memory**
  A coding agent reads a long design doc or API contract, compresses it, and later needs the exact
  constraint, field, or edge case to implement a change correctly.

- **Logs to diagnosis memory**
  An agent reads noisy traces, incident notes, and error logs, then must answer what actually broke,
  where, and why.

- **Repo slice to bug-fix memory**
  An agent sees only part of a codebase, summarizes architecture and invariants, then later has to
  identify the right file, call path, or hidden dependency.

- **PR or review compression**
  An agent reads a long diff or review thread, distills the semantic changes, and later answers
  regression-risk or test-gap questions.

- **Needle in the middle**
  A crucial fact appears in the middle of a long context window rather than the beginning or end.
  The benchmark measures whether compression preserved that fact.

---

## Why Summarization Is The Right Primitive

For long-context agents, summarization is not just a convenience feature. It is a practical form of
working memory.

When an agent cannot keep an entire codebase, paper, or report in active context, it has to:

1. identify what matters,
2. compress it without introducing distortion,
3. carry that memory forward across later steps.

This benchmark isolates that skill. It rewards summaries that are compact, factual, and useful for
later reasoning instead of summaries that merely sound fluent.

---

## Tasks

| Task | Dataset | Context length | Truncation | Steps | Grader |
|------|---------|---------------|------------|-------|--------|
| **easy** | SQuAD v1 | 300–700 chars | 70% shown | 2 | Token F1 |
| **medium** | SQuAD v1 (long) | 900–2500 chars | 65% shown | 2 | Token F1 |
| **hard** | QASPER | 2000–10000 chars | 55% shown (2 chunks) | 3 | Token F1 |

### Hard Task — Chained Summarization

The hard task splits the document into two chunks and runs a **3-step episode**:

1. Show **chunk 1** → model produces `summary_1`
2. Show model's summary + **chunk 2** → model produces `summary_2` (updated)
3. Show `summary_2` + question → model answers → reward computed

This tests whether information survives **two compression rounds**, mirroring real-world long-context scenarios.

---

## Relation To Coding Benchmarks

This environment is not a replacement for software-engineering benchmarks; it complements them.

- **SWE-bench** focuses on issue-resolution and patch-level software engineering tasks across real repositories.
- **CrossCodeEval** focuses on cross-file code understanding and repository-level dependencies.
- **Long Code Arena** studies long-context code tasks such as bug localization, project completion, build repair, and module summarization.

Our benchmark targets an earlier failure point in the same pipeline: whether a model can create a
faithful compressed memory from partial long context before attempting the final coding action.

That makes it a useful building block for:

- repository-aware coding agents,
- retrieval-augmented coding systems,
- memory modules for long-horizon assistants,
- ablations on context compression versus direct long-context inference.

References:

- SWE-bench: https://www.swebench.com/SWE-bench/guides/datasets/
- CrossCodeEval: https://crosscodeeval.github.io/
- Long Code Arena: https://arxiv.org/abs/2406.11612

---

## Labeling Pipeline

The repository now includes a separate labeling-pipeline implementation that can assign `easy`,
`medium`, or `hard` to candidate samples automatically instead of relying only on hand-authored
task buckets.

The idea is:

1. ingest candidate samples from one or more datasets,
2. normalize them into a shared schema,
3. ask a labeling model to predict the difficulty based on context length, technicality,
   number of facts that must be retained, and whether multi-stage memory updates are required,
4. route the sample into the corresponding difficulty pool.

This would make it easier to:

- combine multiple datasets into the same benchmark family,
- rebalance difficulty distributions over time,
- support both natural-language and code-oriented long-context tracks,
- analyze where a model starts to fail as context pressure increases.

A natural-language version of this pipeline could combine sources such as:

- `LongBench v2`
- `Loong`
- `NovelQA`
- `QuALITY`
- `QASPER`
- `PeerQA`
- `RULER`

A code-focused version could later combine:

- `RepoQA`
- `LongCodeBench`
- `LongCodeQA`
- `LONGCODEU`
- `Long Code Arena`
- `CrossCodeEval`

The current runtime does **not** depend on this labeling pipeline yet. For stability and
submission safety, the environment still uses the existing task setup and deterministic defaults,
but the code for normalized sample curation and optional LLM-assisted labeling now exists under
`data/`.

Key modules:

- `data/schema.py`
  Canonical normalized sample schema.

- `data/pipeline.py`
  Offline normalization utilities and source configuration scaffolding.

- `data/labeling.py`
  Heuristic and optional LLM-assisted difficulty labeling for curation workflows.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode. Body: `{"task_name": "easy"|"medium"|"hard", "seed": int}` |
| `/step` | POST | Send action. Body: `{"action": {"response": "your text"}}` |
| `/state` | GET | Episode metadata (step_count, task_name, step_type) |
| `/health` | GET | Liveness check |
| `/schema` | GET | Action/Observation/State JSON schemas |

### Reset Response

```json
{
  "observation": {
    "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "step_type": "summarize",
    "task_name": "easy",
    "context_length": 520,
    "truncation_ratio": 0.7,
    "done": false,
    "reward": null
  },
  "reward": null,
  "done": false
}
```

### Step Response (final step)

```json
{
  "observation": {
    "messages": [...],
    "step_type": "done",
    "done": true,
    "reward": 0.82
  },
  "reward": 0.82,
  "done": true
}
```

---

## Quick Start

### 1. Start the server

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 2. Run the baseline inference script

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"

python inference.py
```

**Expected output:**
```
[START] task=easy env=long-context-summarization model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=The excerpt describes the Amazon basin and notes that the rainforest spans territory in nine nations. reward=0.00 done=false error=null
[STEP] step=2 action=nine reward=1.00 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.00,1.00
```

### 3. Docker

```bash
docker build -t summarization-env .
docker run -p 7860:7860 summarization-env
```

### 4. Using the Python client

```python
from client import SummarizationClient

client = SummarizationClient(base_url="http://localhost:7860")

obs = client.reset(task_name="hard", seed=42)
print(obs.step_type)   # "summarize"
print(obs.messages[-1]["content"][:200])  # first chunk of the paper

# Agent produces a summary...
obs = client.step("The paper proposes a new gradient checkpointing method...")
print(obs.step_type)   # "update_summary" (hard) or "answer" (easy/medium)

# Agent updates the summary with chunk 2...
obs = client.step("Updated summary incorporating methods section findings...")
print(obs.step_type)   # "answer"

# Agent answers the question...
obs = client.step("60%")
print(obs.done, obs.reward)  # True, 1.05 (capped to 1.0)
```

---

## Reward Function

```
reward = F1(normalize(predicted_answer), normalize(ground_truth))
       + conciseness_bonus  # +0.05 if answer correct AND summary ≤ 150 words
```

`F1` is computed at the token level after lowercasing and removing punctuation/articles, identical to the SQuAD official evaluation script.

**Partial progress signals:**
- Intermediate steps return `reward: null`
- Final step returns `reward ∈ [0.0, 1.0]`
- Incorrect but partially overlapping answers receive partial credit via F1

Normalization also handles common short-answer variants such as comma-formatted numbers and
percentage spellings like `60 percent` vs `60%`.

An optional hybrid grading mode can be enabled with `USE_LLM_GRADER=true`. In that mode, the
environment keeps deterministic answer grading as the main signal and adds an auxiliary
LLM-judged summary-quality score. If the LLM grader fails for any reason, grading falls back
to the deterministic path automatically.

---

## Testing

These are the core checks used during submission preparation:

```bash
openenv validate --verbose
docker build -t openenv-summarization:test .
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
HF_TOKEN="hf_..." \
ENV_URL="http://localhost:7860" \
python inference.py
python -m unittest discover -s tests -p 'test_*.py'
```

The environment is designed to run on a 2 vCPU / 8 GB machine and the baseline inference flow
completes well under the 20 minute submission limit.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes (inference.py) | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Yes (inference.py) | LLM model identifier |
| `HF_TOKEN` | Yes (inference.py) | HuggingFace / API key |
| `ENV_URL` | No | Environment server URL (default: `http://localhost:7860`) |
| `USE_LLM_GRADER` | No | Enable optional hybrid LLM summary grading (`true`/`false`) |
| `LLM_GRADER_MODEL` | No | Override model used for optional LLM grading |
| `GRADER_ANSWER_REWARD_THRESHOLD` | No | Minimum answer reward for deterministic conciseness bonus |
| `GRADER_SHORT_SUMMARY_WORD_LIMIT` | No | Word limit for the larger conciseness bonus |
| `GRADER_LONG_SUMMARY_WORD_LIMIT` | No | Word limit for the smaller conciseness bonus |
| `GRADER_SHORT_SUMMARY_BONUS` | No | Bonus applied to short good summaries |
| `GRADER_LONG_SUMMARY_BONUS` | No | Bonus applied to moderately short good summaries |
| `GRADER_ANSWER_WEIGHT` | No | Hybrid-grader weight for deterministic answer correctness |
| `GRADER_LLM_SUMMARY_WEIGHT` | No | Hybrid-grader weight for LLM-judged summary quality |
| `GRADER_CONCISENESS_WEIGHT` | No | Hybrid-grader weight for concise summaries |
| `GRADER_LLM_BLEND_MIN_ANSWER_REWARD` | No | Minimum answer reward required before LLM blending is allowed |

---

## Project Structure

```
openenv/
├── models.py            # Pydantic Action / Observation / State
├── client.py            # HTTP client
├── inference.py         # Baseline script ([START]/[STEP]/[END] logging)
├── openenv.yaml         # Environment manifest
├── requirements.txt
├── Dockerfile
├── graders/
│   └── qa_grader.py     # Token-level F1 + conciseness bonus
├── tasks/
│   ├── easy.py          # SQuAD short passages
│   ├── medium.py        # SQuAD longer passages
│   └── hard.py          # QASPER scientific papers (chained summarization)
└── server/
    ├── environment.py   # SummarizationEnvironment (OpenEnv base class)
    └── app.py           # FastAPI app via create_fastapi_app()
```

---

## Hardware Requirements

- CPU: 2 vCPU
- RAM: 8 GB (dataset loading uses ~300 MB; fallback hardcoded samples use ~0 MB)
- No GPU required
- Inference runtime: < 20 minutes for full baseline run
