"""
models.py — GitHub Issue Triage Environment
Team Astra.AI: Om Chougule (Lead), Shraman Patil

Typed Pydantic models for Action, Observation, and State.
All fields documented for OpenEnv spec compliance.
"""

from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


class GithubIssueTriageAction(Action):
    """
    The agent's triage decision for one GitHub issue.

    Fields:
        label            — Required. Issue category. One of: bug | feature | docs | question
        team             — Team to route to. One of: frontend | backend | ml | devops | docs
                           Required for medium + hard tasks. Null for easy task.
        priority         — Urgency level. One of: critical | high | medium | low
                           Required for hard task only. Null for easy + medium.
        suggested_action — A brief, concrete action the assignee should take.
                           Required for hard task only. Null for easy + medium.
        reasoning        — One-sentence justification for the triage decision.
                           Always encouraged; not graded but helps debugging.
    """

    label: str
    team: Optional[str] = None
    priority: Optional[str] = None
    suggested_action: Optional[str] = None
    reasoning: Optional[str] = None


class GithubIssueTriageObservation(Observation):
    """
    What the agent sees at each step.

    Fields:
        issue_id          — GitHub issue number, e.g. "#105"
        issue_title       — Title of the issue
        issue_body        — Full body text of the issue
        repo_name         — Repository the issue belongs to
        author            — GitHub username who filed the issue
        existing_comments — List of comments already on the issue (context)
        task_id           — Current task difficulty: easy | medium | hard
        task_description  — Natural-language description of what the agent must do
        last_reward       — Reward received from the previous step (0.0 on reset)
        feedback          — Human-readable feedback from the last grader result
        step_number       — Current step index within the episode
    """

    issue_id: str = ""
    issue_title: str = ""
    issue_body: str = ""
    repo_name: str = "meta-pytorch/OpenEnv"
    author: str = ""
    existing_comments: List[str] = []
    task_id: str = "easy"
    task_description: str = ""
    last_reward: float = 0.0
    feedback: str = ""
    step_number: int = 0


class GithubIssueTriageState(State):
    """
    Full internal state of the environment (not shown to agent).

    Fields:
        episode_id   — Unique ID for this episode
        task_id      — Current task difficulty
        issue_id     — ID of the issue currently being triaged
        step_count   — Total steps taken in this episode
        total_reward — Cumulative reward across all steps
        last_reward  — Reward from the most recent step
        done         — Whether the episode has ended
    """

    episode_id: str = ""
    task_id: str = "easy"
    issue_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    done: bool = False