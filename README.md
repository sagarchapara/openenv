# Long-Context Summarization — OpenEnv Environment

An RL environment that trains language models to **compress long documents into compact summaries**, then tests whether those summaries preserve enough information to answer questions. Reward signal drives the model toward concise but information-dense compression.

Inspired by [Cursor's self-summarization approach](https://cursor.com/blog/self-summarization).

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
[START] {"task": "easy", "episode_id": "ep_42311", "episode_num": 0}
[STEP] {"step": 1, "type": "summarize", "action_preview": "The Amazon rainforest...", "reward": null}
[STEP] {"step": 2, "type": "answer", "action_preview": "nine", "reward": null}
[END] {"task": "easy", "episode_id": "ep_42311", "total_steps": 2, "final_reward": 1.0, "success": true}
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

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes (inference.py) | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Yes (inference.py) | LLM model identifier |
| `HF_TOKEN` | Yes (inference.py) | HuggingFace / API key |
| `ENV_URL` | No | Environment server URL (default: `http://localhost:7860`) |

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
