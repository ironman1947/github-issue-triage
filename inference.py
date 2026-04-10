#!/usr/bin/env python3
"""
GitHub Issue Triage — OpenEnv Hackathon Inference Script
Team Astra.AI: Om Chougule (Lead), Shraman Patil

Mandatory log format: [START] / [STEP] / [END]
All LLM calls use OpenAI client configured via environment variables.
"""

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

# ── Environment variables (mandatory per spec) ────────────────────────────────
API_BASE_URL: str = os.environ.get(
    "API_BASE_URL", "https://router.huggingface.co/novita/v3/openai"
)
MODEL_NAME: str = os.environ.get(
    "MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"
)
API_KEY: str = os.environ.get("HF_TOKEN", "")          # no default — required
ENV_BASE_URL: str = os.environ.get(
    "ENV_BASE_URL", "https://om192006-github-issue-triage.hf.space"
)

# ── Inference hyper-params ────────────────────────────────────────────────────
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 512
MAX_STEPS: int = 1          # single-step episode: one triage decision per issue
SUCCESS_SCORE_THRESHOLD: float = 0.7

TASK_IDS: List[str] = ["easy", "medium", "hard"]
BENCHMARK: str = "github_issue_triage"

# ── Reward weights per task (must sum to MAX_TOTAL_REWARD per task) ──────────
MAX_TOTAL_REWARD: float = 1.0   # per task, clamped to [0,1]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert GitHub issue triager at a large software company.
Your job is to read a GitHub issue and make a structured triage decision.

Always respond with ONLY a valid JSON object — no markdown, no explanation, no extra text.

JSON schema:
{
  "label": "<one of: bug | feature | docs | question>",
  "team": "<one of: frontend | backend | ml | devops | docs | null>",
  "priority": "<one of: critical | high | medium | low | null>",
  "suggested_action": "<a brief concrete action the team should take, or null>",
  "reasoning": "<one sentence explaining your decision>"
}

Rules:
- label is ALWAYS required.
- team is required for medium and hard tasks (set null only for easy task).
- priority is required for hard tasks (set null for easy/medium).
- suggested_action is required for hard tasks (set null for easy/medium).
- Choose priority based on impact: critical=data loss/security, high=blocks users,
  medium=degrades experience, low=minor/cosmetic.
"""


# ── Mandatory log helpers (exact format validated by judges) ──────────────────
def log_start(task: str, env: str, model: str) -> None:
    """Print [START] block. Must be first output for each task."""
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Print [STEP] block after each environment step."""
    error_str = error if error is not None else "null"
    done_str = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Print [END] block as the final output for each task."""
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    success_str = str(success).lower()
    print(
        f"[END] success={success_str} steps={steps} score={score:.4f} rewards=[{rewards_str}]",
        flush=True,
    )


# ── HTTP helpers (sync — avoids async client dependency) ─────────────────────
import urllib.request
import urllib.error


def _http_post(url: str, body: dict) -> dict:
    """Simple sync HTTP POST, returns parsed JSON. Raises on error."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def env_reset(task_id: str) -> dict:
    url = f"{ENV_BASE_URL}/reset"
    return _http_post(url, {"task_id": task_id})


def env_step(action: dict, task_id: str = "easy") -> dict:
    url = f"{ENV_BASE_URL}/step"
    return _http_post(url, {"action": action, "task_id": task_id})


# ── LLM call ─────────────────────────────────────────────────────────────────
def build_user_prompt(observation: dict) -> str:
    issue = observation.get("issue_title", "")
    body = observation.get("issue_body", "")
    author = observation.get("author", "")
    comments = observation.get("existing_comments", [])
    task_desc = observation.get("task_description", "")
    feedback = observation.get("feedback", "")
    last_reward = observation.get("last_reward", 0.0)

    comments_str = "\n".join(f"  - {c}" for c in comments) if comments else "  (none)"

    return f"""=== GitHub Issue ===
Title: {issue}
Author: {author}
Body:
{body}

Existing comments:
{comments_str}

=== Your Task ===
{task_desc}

Previous feedback: {feedback}
Previous reward: {last_reward:.2f}

Respond with ONLY a JSON object as specified."""


def get_model_action(
    client: OpenAI,
    observation: dict,
) -> dict:
    """Call the LLM and return a parsed action dict. Falls back to safe default."""
    user_prompt = build_user_prompt(observation)
    task_id = observation.get("task_id", "easy")

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

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        # Normalise — enforce required fields exist
        action = {
            "label": parsed.get("label", "question"),
            "team": parsed.get("team", None),
            "priority": parsed.get("priority", None),
            "suggested_action": parsed.get("suggested_action", None),
            "reasoning": parsed.get("reasoning", "No reasoning provided"),
        }
        return action

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Safe fallback — attempt a reasonable default per task
        fallback = {"label": "bug", "team": None, "priority": None,
                    "suggested_action": None, "reasoning": "fallback default"}
        if task_id in ("medium", "hard"):
            fallback["team"] = "backend"
        if task_id == "hard":
            fallback["priority"] = "high"
            fallback["suggested_action"] = "investigate and fix the reported issue"
        return fallback


# ── Run one task episode ──────────────────────────────────────────────────────
def run_task(client: OpenAI, task_id: str) -> float:
    """
    Runs a single-episode task.
    Returns normalised score in [0.0, 1.0].
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ──────────────────────────────────────────────────────────
        try:
            reset_result = env_reset(task_id)
        except Exception as exc:
            print(f"[DEBUG] reset() failed for task={task_id}: {exc}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        observation = reset_result.get("observation", {})
        done = reset_result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # ── agent decides ──────────────────────────────────────────────
            action = get_model_action(client, observation)
            action_str = json.dumps(action)

            # ── step ───────────────────────────────────────────────────────
            error_msg: Optional[str] = None
            try:
                step_result = env_step(action, task_id=task_id)
                reward = float(step_result.get("reward", 0.0))
                done = bool(step_result.get("done", True))
                observation = step_result.get("observation", {})
            except Exception as exc:
                print(f"[DEBUG] step() failed: {exc}", flush=True)
                reward = 0.0
                done = True
                error_msg = str(exc)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

        # ── compute score ──────────────────────────────────────────────────
        if rewards:
            score = sum(rewards) / (MAX_TOTAL_REWARD * len(rewards))
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unexpected error in task={task_id}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not API_KEY:
        print("[DEBUG] WARNING: HF_TOKEN is not set. LLM calls will fail.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    for task_id in TASK_IDS:
        score = run_task(client, task_id)
        results[task_id] = score
        print(flush=True)   # blank line between tasks for readability

    # ── Summary ───────────────────────────────────────────────────────────
    total = sum(results.values()) / len(results)
    print("=" * 60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for task_id, score in results.items():
        bar = "✅" if score >= SUCCESS_SCORE_THRESHOLD else "❌"
        print(f"  {task_id.upper():<8} → {score:.4f}  {bar}", flush=True)
    print(f"  {'TOTAL':<8} → {total:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()