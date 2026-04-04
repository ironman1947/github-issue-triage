# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the GitHub Issue Triage Environment.

The agent reads a GitHub issue and must:
  - Easy:   Assign the correct LABEL (bug / feature / docs / question)
  - Medium: Assign label + correct TEAM (frontend / backend / ml / devops)
  - Hard:   Assign label + team + PRIORITY (critical/high/medium/low)
            + suggest a first fix ACTION
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ──────────────────────────────────────────
# ACTION — what the agent decides
# ──────────────────────────────────────────
class GithubIssueTriageAction(Action):
    """
    The agent's triage decision for a GitHub issue.

    For easy task   → only label is required.
    For medium task → label + team are required.
    For hard task   → label + team + priority + suggested_action required.
    """

    label: str = Field(
        ...,
        description="Issue label. One of: 'bug', 'feature', 'docs', 'question'"
    )
    team: Optional[str] = Field(
        default=None,
        description="Team to assign. One of: 'frontend', 'backend', 'ml', 'devops'"
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority level. One of: 'critical', 'high', 'medium', 'low'"
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description="A short first fix suggestion, e.g. 'Check null pointer in auth module'"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Why the agent made this decision (used for logging)"
    )


# ──────────────────────────────────────────
# OBSERVATION — what the agent sees
# ──────────────────────────────────────────
class GithubIssueTriageObservation(Observation):
    """
    What the agent sees at each step — a GitHub issue to triage.
    """

    # The issue content
    issue_id: str = Field(default="", description="Issue ID, e.g. '#142'")
    issue_title: str = Field(default="", description="Title of the GitHub issue")
    issue_body: str = Field(default="", description="Full body/description of the issue")
    repo_name: str = Field(default="", description="Repository name, e.g. 'meta-pytorch/OpenEnv'")
    author: str = Field(default="", description="GitHub username who opened the issue")
    existing_comments: list[str] = Field(
        default_factory=list,
        description="Any existing comments on the issue"
    )

    # Task context
    task_id: str = Field(default="easy", description="Current task: 'easy', 'medium', or 'hard'")
    task_description: str = Field(default="", description="What the agent needs to do")

    # Feedback from last step
    last_reward: float = Field(default=0.0, description="Reward from previous action")
    feedback: str = Field(default="", description="Feedback message explaining last reward")

    # Episode info
    done: bool = Field(default=False, description="Whether the episode is over")
    step_number: int = Field(default=0, description="Current step number")