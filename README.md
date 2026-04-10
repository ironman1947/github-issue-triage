---
title: GitHub Issue Triage — OpenEnv
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - nlp
  - triage
  - real-world
---

# 🐛 GitHub Issue Triage — OpenEnv Environment

**Meta PyTorch OpenEnv Hackathon × Scaler School of Technology**  
**Team Astra.AI** · Om Chougule (Lead) · Shraman Patil

---

## What is this?

A real-world **Reinforcement Learning environment** where an AI agent reads GitHub
issues and performs structured triage decisions — the exact task that software
engineers do dozens of times per day.

The agent must:
1. Read an issue title, body, author, and existing comments
2. Assign a **label** (bug / feature / docs / question)
3. Route it to the correct **team** (frontend / backend / ml / devops / docs)
4. Score its **priority** (critical / high / medium / low)
5. Suggest a **concrete fix action**

This directly trains agents for real developer productivity tools (GitHub Copilot,
Linear, Jira auto-assign, etc.).

---

## Environment Design

### Action Space

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `string` | Always | `bug` · `feature` · `docs` · `question` |
| `team` | `string\|null` | Medium + Hard | `frontend` · `backend` · `ml` · `devops` · `docs` |
| `priority` | `string\|null` | Hard only | `critical` · `high` · `medium` · `low` |
| `suggested_action` | `string\|null` | Hard only | Brief concrete fix recommendation |
| `reasoning` | `string\|null` | Optional | Agent's justification (not graded) |

### Observation Space

| Field | Type | Description |
|---|---|---|
| `issue_id` | `string` | GitHub issue number |
| `issue_title` | `string` | Title of the issue |
| `issue_body` | `string` | Full issue body |
| `author` | `string` | Who filed the issue |
| `existing_comments` | `list[str]` | Prior comments (context) |
| `task_id` | `string` | Current difficulty: `easy` / `medium` / `hard` |
| `task_description` | `string` | What the agent must do this episode |
| `last_reward` | `float` | Reward from previous step |
| `feedback` | `string` | Grader feedback explaining the score |

---

## Tasks and Grading

### 🟢 Easy — Label Assignment
**Objective:** Assign the correct label to the issue.  
**Grader:** `1.0` if correct, `0.0` if wrong.  
**Challenge level:** Straightforward for capable LLMs.

### 🟡 Medium — Label + Team Routing
**Objective:** Assign correct label AND route to the correct engineering team.  
**Grader:** `label (0.5)` + `team (0.5)` — partial credit if one is correct.  
**Challenge level:** Requires understanding of org structure and issue context.

### 🔴 Hard — Full Triage (Label + Team + Priority + Fix)
**Objective:** Full triage decision — label, team, priority, and a concrete fix action.  
**Grader:** `label (0.30)` + `team (0.30)` + `priority (0.20)` + `fix quality (0.20)`  
**Challenge level:** Genuinely challenges frontier models on multi-criteria reasoning.

> **Reward function design:** All rewards are continuous `[0.0, 1.0]`, providing
> partial credit at every step. The fix suggestion uses keyword-overlap scoring so
> specificity is rewarded — vague answers get partial credit, precise answers get full.

---

## Baseline Scores (Llama 3.1 8B via HF Router)

| Task | Baseline Score |
|---|---|
| 🟢 Easy | **1.0000** ✅ |
| 🟡 Medium | **1.0000** ✅ |
| 🔴 Hard | **0.8667** ✅ |
| **Average** | **0.9556** |

---

## Quick Start

### Option 1 — Use the hosted Space
```bash
curl -X POST https://om192006-github-issue-triage.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

### Option 2 — Run locally with Docker
```bash
git clone https://github.com/ironman1947/github-issue-triage
cd github-issue-triage

docker build -t github-issue-triage:latest .
docker run -d -p 8000:8000 github-issue-triage:latest

# Test
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "hard"}'
```

### Option 3 — Run inference script
```bash
pip install openenv-core openai
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/novita/v3/openai
export MODEL_NAME=meta-llama/llama-3.1-8b-instruct
export ENV_BASE_URL=https://om192006-github-issue-triage.hf.space

python inference.py
```

### Validate
```bash
pip install openenv-core
openenv validate
```

---

## Project Structure

```
github-issue-triage/
├── inference.py                          # Baseline agent (run me!)
├── models.py                             # Typed Pydantic models
├── client.py                             # Python client helper
├── openenv.yaml                          # OpenEnv spec metadata
├── Dockerfile                            # Root Dockerfile
├── README.md
├── pyproject.toml
└── server/
    ├── app.py                            # FastAPI server
    └── github_issue_triage_environment.py  # Environment + grader logic
```

---

## Real-World Motivation

GitHub issue triage is a **bottleneck in every software team**. Issues pile up
unlabelled, unassigned, with no priority. Human triagers spend hours per week on
this. This environment enables:

- **Training** LLM agents to triage automatically
- **Evaluating** how well a model understands developer context
- **Benchmarking** different models on a grounded, reproducible task

Companies like GitHub, GitLab, Linear, and Jira are actively investing in
AI-powered triage — this environment enables RL research directly applicable to that.

---

## Team

| Name | Role |
|---|---|
| Om Chougule | Team Lead · Environment Design · Backend |
| Shraman Patil | Grader Logic · Inference Script |

**Team:** Astra.AI  
**Hackathon:** Meta PyTorch OpenEnv Hackathon × Scaler School of Technology  
**GitHub:** https://github.com/ironman1947/github-issue-triage