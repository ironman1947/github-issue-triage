"""
inference.py — GitHub Issue Triage Environment
===============================================
Hackathon inference script.

Runs an LLM agent against the GitHub Issue Triage environment
across all 3 tasks (easy, medium, hard) and logs results in the
required [START] / [STEP] / [END] format.

Usage:
    uv run inference.py

Required environment variables:
    API_BASE_URL   → HF Router base URL
    MODEL_NAME     → Model identifier
    HF_TOKEN       → Your Hugging Face token (used as API key)
"""

import asyncio
import json
import os
import sys
from typing import List

from openai import OpenAI

# ──────────────────────────────────────────
# CONFIG — read from environment variables
# ──────────────────────────────────────────
API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://router.huggingface.co/novita/v3/openai"
)
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"
)
API_KEY = os.environ.get("HF_TOKEN", "")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "github_issue_triage-env:latest")

MAX_STEPS = 1          # one triage decision per episode
TEMPERATURE = 0.2
MAX_TOKENS = 512

# Tasks to evaluate
TASKS = ["easy", "medium", "hard"]
MAX_TOTAL_REWARD = len(TASKS) * 1.0   # max possible = 3.0
SUCCESS_THRESHOLD = 0.6                # 60% = pass

# ──────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert GitHub issue triager working on the meta-pytorch/OpenEnv repository.

You will be shown a GitHub issue. Your job is to triage it by responding with a JSON object.

The JSON must have these fields:
  - "label"            : one of "bug", "feature", "docs", "question"
  - "team"             : one of "frontend", "backend", "ml", "devops"
  - "priority"         : one of "critical", "high", "medium", "low"
  - "suggested_action" : a short first fix step (1-2 sentences)
  - "reasoning"        : brief explanation of your decision

Respond ONLY with valid JSON. No extra text, no markdown, no code blocks.

Example response:
{
  "label": "bug",
  "team": "backend",
  "priority": "high",
  "suggested_action": "Investigate memory allocation in file_handler.py and add size validation.",
  "reasoning": "Stack trace points to a MemoryError in file handling code after a recent update."
}"""


# ──────────────────────────────────────────
# LOGGING — required [START] [STEP] [END] format
# ──────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model,
    }), flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error=None) -> None:
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": str(error) if error else None,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "event": "END",
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": rewards,
    }), flush=True)


# ──────────────────────────────────────────
# BUILD PROMPT from observation
# ──────────────────────────────────────────
def build_user_prompt(obs) -> str:
    """Convert observation into a clear prompt for the LLM."""
    comments_text = ""
    if obs.existing_comments:
        comments_text = "\n".join(
            f"  - {c}" for c in obs.existing_comments
        )
        comments_text = f"\nExisting Comments:\n{comments_text}"

    return f"""
{obs.task_description}

---
Repository: {obs.repo_name}
Issue {obs.issue_id}: {obs.issue_title}

Description:
{obs.issue_body}
{comments_text}
---

Respond with a JSON object containing your triage decision.
""".strip()


# ──────────────────────────────────────────
# CALL LLM
# ──────────────────────────────────────────
def get_agent_action(client: OpenAI, obs) -> dict:
    """
    Call the LLM and parse its JSON triage decision.
    Falls back to a safe default if parsing fails.
    """
    user_prompt = build_user_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return parsed

    except Exception as exc:
        print(f"[DEBUG] LLM call or parse failed: {exc}", flush=True)
        # Safe fallback
        return {
            "label": "bug",
            "team": "backend",
            "priority": "medium",
            "suggested_action": "Investigate the reported issue.",
            "reasoning": "Fallback response due to parse error.",
        }


# ──────────────────────────────────────────
# RUN ONE TASK
# ──────────────────────────────────────────
async def run_task(task_id: str, client: OpenAI) -> float:
    """
    Run one full episode for a given task (easy / medium / hard).
    Returns the reward score (0.0 to 1.0).
    """
    from github_issue_triage import GithubIssueTriageEnv
    from github_issue_triage.models import GithubIssueTriageAction

    log_start(task=task_id, env="github_issue_triage", model=MODEL_NAME)

    reward = 0.0
    steps_taken = 0
    success = False
    rewards = []

    try:
        # Connect to environment
        env = await GithubIssueTriageEnv.from_docker_image(IMAGE_NAME)

        async with env:
            # Reset with the correct task
            result = await env.reset(task_id=task_id)
            obs = result.observation

            # One step per episode
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                # Get agent decision
                action_dict = get_agent_action(client, obs)

                action = GithubIssueTriageAction(
                    label=action_dict.get("label", "bug"),
                    team=action_dict.get("team"),
                    priority=action_dict.get("priority"),
                    suggested_action=action_dict.get("suggested_action"),
                    reasoning=action_dict.get("reasoning"),
                )

                # Step the environment
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_dict,
                    reward=reward,
                    done=done,
                    error=None,
                )

                if done:
                    break

        score = reward  # single step episode, reward IS the score
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task '{task_id}' failed: {exc}", flush=True)
        score = 0.0
        success = False
        rewards = [0.0]
        steps_taken = 0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ──────────────────────────────────────────
# MAIN — run all 3 tasks
# ──────────────────────────────────────────
async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable is not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("\n" + "="*60, flush=True)
    print("  GitHub Issue Triage — Inference Script", flush=True)
    print("="*60 + "\n", flush=True)

    all_scores = []

    for task_id in TASKS:
        print(f"\n--- Running Task: {task_id.upper()} ---\n", flush=True)
        score = await run_task(task_id, client)
        all_scores.append(score)
        print(f"Task '{task_id}' score: {score:.4f}\n", flush=True)

    # Final summary
    total_score = sum(all_scores) / len(all_scores)
    total_score = min(max(total_score, 0.0), 1.0)

    print("\n" + "="*60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("="*60, flush=True)
    for task_id, score in zip(TASKS, all_scores):
        print(f"  {task_id.upper():8s} → {score:.4f}", flush=True)
    print(f"  {'TOTAL':8s} → {total_score:.4f}", flush=True)
    print("="*60 + "\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())