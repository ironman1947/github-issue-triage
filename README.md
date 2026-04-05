---
title: GitHub Issue Triage Environment
emoji: 🐛
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - agent
  - github
  - triage
---

# 🐛 GitHub Issue Triage — OpenEnv Environment

A real-world reinforcement learning environment where an AI agent learns to
triage GitHub issues — assigning labels, teams, priorities, and suggesting
first fix steps.

Built for the **Meta PyTorch OpenEnv Hackathon 2026**.

---

## 👥 Team Astra.AI

| Name | Role |
|------|------|
| **Om Chougule** | Team Lead |
| **Shraman Patil** | Member |

---

## 🎯 What the Agent Does

The agent reads a GitHub issue (title, body, comments) and must make a
triage decision — just like a real engineer triaging an issue queue.

### 3 Tasks (Easy → Hard)

| Task | What Agent Must Do | Max Reward |
|------|--------------------|------------|
| 🟢 **Easy** | Assign correct **label** (`bug` / `feature` / `docs` / `question`) | 1.0 |
| 🟡 **Medium** | Assign correct **label** + **team** (`frontend` / `backend` / `ml` / `devops`) | 1.0 |
| 🔴 **Hard** | Assign **label** + **team** + **priority** + suggest a **first fix action** | 1.0 |

---

## 🔌 API Usage

### Quick Start

```python
from github_issue_triage import GithubIssueTriageEnv, GithubIssueTriageAction

# Connect to this Space
async with GithubIssueTriageEnv(
    base_url="https://om192006-github-issue-triage.hf.space"
) as env:

    # Start easy task
    result = await env.reset(task_id="easy")
    print(result.observation.issue_title)
    print(result.observation.task_description)

    # Agent makes a decision
    action = GithubIssueTriageAction(label="bug")
    result = await env.step(action)

    print(result.reward)               # 0.0 to 1.0
    print(result.observation.feedback) # what was right/wrong
```

### Hard Task Example

```python
action = GithubIssueTriageAction(
    label="bug",
    team="backend",
    priority="critical",
    suggested_action="Investigate memory allocation in file_handler.py and add upload size validation.",
    reasoning="Stack trace shows MemoryError after recent update."
)
result = await env.step(action)
print(result.reward)  # up to 1.0
```

---

## 📐 Action & Observation Space

### Action
| Field | Type | Required | Values |
|-------|------|----------|--------|
| `label` | str | Always | `bug`, `feature`, `docs`, `question` |
| `team` | str | Medium + Hard | `frontend`, `backend`, `ml`, `devops` |
| `priority` | str | Hard only | `critical`, `high`, `medium`, `low` |
| `suggested_action` | str | Hard only | Free text (graded by keyword matching) |
| `reasoning` | str | Optional | Free text (for logging only) |

### Observation
| Field | Description |
|-------|-------------|
| `issue_id` | Issue number e.g. `#101` |
| `issue_title` | Title of the GitHub issue |
| `issue_body` | Full description |
| `repo_name` | Repository name |
| `author` | GitHub username |
| `existing_comments` | Any existing comments |
| `task_description` | What the agent needs to do |
| `feedback` | Grader feedback from last step |
| `last_reward` | Reward from previous action |
| `done` | Whether episode is over |

---

## 🏆 Reward Function

### Easy Task
- ✅ Correct label → `1.0`
- ❌ Wrong label → `0.0`

### Medium Task
- ✅ Correct label → `+0.5`
- ✅ Correct team → `+0.5`

### Hard Task
- ✅ Correct label → `+0.3`
- ✅ Correct team → `+0.3`
- ✅ Correct priority → `+0.2`
- 💡 Suggested action quality → `+0.0 to 0.2` (keyword matching)

Rewards are **partial** — the agent gets credit for each correct field,
making the reward signal rich and non-sparse.

---

## 🚀 Setup & Run Locally

```bash
# Clone and install
git clone https://huggingface.co/spaces/om192006/github-issue-triage
cd github-issue-triage
uv sync

# Run server locally
uv run server

# Or with Docker
docker build -t github_issue_triage-env:latest .
docker run -p 8000:8000 github_issue_triage-env:latest
```

---

## 📊 Baseline Scores

Run the inference script:

```bash
export HF_TOKEN="your_token_here"
export API_BASE_URL="https://router.huggingface.co/novita/v3/openai"
export MODEL_NAME="meta-llama/llama-3.1-8b-instruct"

uv run inference.py
```

| Task | Baseline Score (Llama 3.1 8B) |
|------|-------------------------------|
| Easy | 1.0000  |
| Medium | 1.0000  |
| Hard | 0.8667  |
| **Average** | **0.9556** |

---

## 📁 Project Structure

```
github_issue_triage/
├── models.py                              # Action & Observation types
├── client.py                              # HTTP/WebSocket client
├── inference.py                           # Hackathon inference script
├── openenv.yaml                           # OpenEnv manifest
├── Dockerfile                             # Container definition (root)
├── pyproject.toml                         # Project metadata
└── server/
    ├── app.py                             # FastAPI server
    └── github_issue_triage_environment.py # Environment logic + grader
```

---

## 🔗 Links

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Meta PyTorch OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/)
- [Hugging Face OpenEnv Hub](https://huggingface.co/openenv)