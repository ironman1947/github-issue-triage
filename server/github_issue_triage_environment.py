# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GitHub Issue Triage Environment Implementation.

The agent reads real-world style GitHub issues and must triage them.

3 Tasks:
  - easy   → assign correct LABEL only
  - medium → assign correct LABEL + TEAM
  - hard   → assign correct LABEL + TEAM + PRIORITY + suggest a fix action
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import GithubIssueTriageAction, GithubIssueTriageObservation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import GithubIssueTriageAction, GithubIssueTriageObservation


# ──────────────────────────────────────────────────────────────
# DATASET — realistic GitHub issues with ground truth answers
# ──────────────────────────────────────────────────────────────
ISSUES = [
    {
        "issue_id": "#101",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "App crashes when uploading files larger than 10MB",
        "issue_body": (
            "When I try to upload a file larger than 10MB, the application crashes "
            "with a 500 Internal Server Error. This worked fine in v1.2.0 but broke "
            "after the latest update. Stack trace: MemoryError at file_handler.py line 42."
        ),
        "author": "dev_user99",
        "existing_comments": ["I can reproduce this on Linux too.", "Same issue on Windows."],
        "label": "bug",
        "team": "backend",
        "priority": "high",
        "fix_keywords": ["memory", "file", "upload", "handler", "limit"],
    },
    {
        "issue_id": "#102",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "Add dark mode support to the dashboard",
        "issue_body": (
            "Many users have requested a dark mode for the dashboard UI. "
            "This would improve usability during night-time usage and reduce eye strain. "
            "Please consider adding a toggle in the settings page."
        ),
        "author": "ux_designer_01",
        "existing_comments": ["Would love this!", "+1 from our team."],
        "label": "feature",
        "team": "frontend",
        "priority": "medium",
        "fix_keywords": ["dark", "mode", "theme", "css", "toggle", "ui"],
    },
    {
        "issue_id": "#103",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "README missing setup instructions for Windows",
        "issue_body": (
            "The README only has setup instructions for Linux and Mac. "
            "Windows users are left without guidance. Could someone add "
            "Windows-specific steps for installation and running the server?"
        ),
        "author": "windows_user42",
        "existing_comments": [],
        "label": "docs",
        "team": "devops",
        "priority": "low",
        "fix_keywords": ["readme", "windows", "setup", "install", "documentation"],
    },
    {
        "issue_id": "#104",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "Model training loss goes to NaN after epoch 3",
        "issue_body": (
            "When training the default model configuration, the loss becomes NaN "
            "after epoch 3. I tried reducing the learning rate but it did not help. "
            "This happens consistently on both GPU and CPU. "
            "Gradient clipping does not seem to fix it either."
        ),
        "author": "ml_researcher_07",
        "existing_comments": ["Could be a numerical stability issue.", "Try checking for inf in inputs."],
        "label": "bug",
        "team": "ml",
        "priority": "critical",
        "fix_keywords": ["nan", "loss", "gradient", "training", "numerical", "stability"],
    },
    {
        "issue_id": "#105",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "How do I configure custom environment variables?",
        "issue_body": (
            "I am trying to configure custom environment variables for my deployment "
            "but I cannot find any documentation on this. "
            "Is there a config file or a CLI flag I should use?"
        ),
        "author": "new_contributor_22",
        "existing_comments": ["Check the openenv.yaml file.", "Also see the README deployment section."],
        "label": "question",
        "team": "devops",
        "priority": "low",
        "fix_keywords": ["config", "env", "variable", "yaml", "documentation"],
    },
    {
        "issue_id": "#106",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "Login button unresponsive on Safari browser",
        "issue_body": (
            "The login button does not respond on Safari version 17.x. "
            "Clicking it does nothing — no error, no redirect. "
            "Works fine on Chrome and Firefox. Console shows: "
            "TypeError: undefined is not a function at login.js:88."
        ),
        "author": "qa_tester_03",
        "existing_comments": ["Confirmed on Safari 17.2 on macOS Sonoma."],
        "label": "bug",
        "team": "frontend",
        "priority": "high",
        "fix_keywords": ["safari", "login", "javascript", "browser", "typeerror"],
    },
    {
        "issue_id": "#107",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "Add support for batch inference in the API",
        "issue_body": (
            "Currently the API only supports single-item inference. "
            "Adding batch support would significantly improve throughput "
            "for production use cases. A simple list input format would suffice."
        ),
        "author": "prod_engineer_11",
        "existing_comments": [],
        "label": "feature",
        "team": "ml",
        "priority": "high",
        "fix_keywords": ["batch", "inference", "api", "throughput", "list"],
    },
    {
        "issue_id": "#108",
        "repo_name": "meta-pytorch/OpenEnv",
        "issue_title": "Docker container runs out of memory on startup",
        "issue_body": (
            "The Docker container exits with OOM (Out of Memory) error during startup "
            "even on machines with 16GB RAM. "
            "docker run -p 8000:8000 my-env:latest fails immediately. "
            "No issues before the last release."
        ),
        "author": "devops_lead_05",
        "existing_comments": ["Try setting --memory=8g flag.", "Check for memory leaks in init."],
        "label": "bug",
        "team": "devops",
        "priority": "critical",
        "fix_keywords": ["docker", "memory", "oom", "container", "startup"],
    },
]

# ──────────────────────────────────────────────────────────────
# VALID VALUES
# ──────────────────────────────────────────────────────────────
VALID_LABELS = {"bug", "feature", "docs", "question"}
VALID_TEAMS = {"frontend", "backend", "ml", "devops"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}

# ──────────────────────────────────────────────────────────────
# TASK DESCRIPTIONS shown to the agent
# ──────────────────────────────────────────────────────────────
TASK_DESCRIPTIONS = {
    "easy": (
        "TASK (Easy): Read the GitHub issue carefully and assign the correct LABEL.\n"
        "Valid labels: 'bug', 'feature', 'docs', 'question'.\n"
        "Only the 'label' field in your action will be graded."
    ),
    "medium": (
        "TASK (Medium): Read the GitHub issue and assign the correct LABEL and TEAM.\n"
        "Valid labels: 'bug', 'feature', 'docs', 'question'.\n"
        "Valid teams: 'frontend', 'backend', 'ml', 'devops'.\n"
        "Both 'label' and 'team' fields will be graded."
    ),
    "hard": (
        "TASK (Hard): Read the GitHub issue and assign the correct LABEL, TEAM, and PRIORITY.\n"
        "Also provide a short 'suggested_action' (first fix step).\n"
        "Valid labels: 'bug', 'feature', 'docs', 'question'.\n"
        "Valid teams: 'frontend', 'backend', 'ml', 'devops'.\n"
        "Valid priorities: 'critical', 'high', 'medium', 'low'.\n"
        "All four fields will be graded."
    ),
}


# ──────────────────────────────────────────────────────────────
# GRADER — scores agent action against ground truth
# ──────────────────────────────────────────────────────────────
def grade_action(
    action: GithubIssueTriageAction,
    issue: dict,
    task_id: str,
) -> tuple[float, str]:
    """
    Grade the agent's action and return (reward, feedback_message).

    Scoring:
      easy   → label correct = 1.0, wrong = 0.0
      medium → label(0.5) + team(0.5)
      hard   → label(0.3) + team(0.3) + priority(0.2) + action_quality(0.2)
    """
    reward = 0.0
    feedback_parts = []

    label_correct = action.label.lower() == issue["label"]
    team_correct = (action.team or "").lower() == issue["team"]
    priority_correct = (action.priority or "").lower() == issue["priority"]

    # Check suggested_action quality (keyword matching)
    action_text = (action.suggested_action or "").lower()
    keyword_hits = sum(1 for kw in issue["fix_keywords"] if kw in action_text)
    action_score = min(keyword_hits / max(len(issue["fix_keywords"]), 1), 1.0)

    if task_id == "easy":
        if label_correct:
            reward = 1.0
            feedback_parts.append(f"✅ Correct label '{issue['label']}'! Full marks.")
        else:
            reward = 0.0
            feedback_parts.append(
                f"❌ Wrong label '{action.label}'. Correct: '{issue['label']}'."
            )

    elif task_id == "medium":
        if label_correct:
            reward += 0.5
            feedback_parts.append(f"✅ Correct label '{issue['label']}' (+0.5).")
        else:
            feedback_parts.append(
                f"❌ Wrong label '{action.label}'. Correct: '{issue['label']}' (+0.0)."
            )
        if team_correct:
            reward += 0.5
            feedback_parts.append(f"✅ Correct team '{issue['team']}' (+0.5).")
        else:
            feedback_parts.append(
                f"❌ Wrong team '{action.team}'. Correct: '{issue['team']}' (+0.0)."
            )

    elif task_id == "hard":
        if label_correct:
            reward += 0.3
            feedback_parts.append(f"✅ Correct label '{issue['label']}' (+0.3).")
        else:
            feedback_parts.append(
                f"❌ Wrong label '{action.label}'. Correct: '{issue['label']}' (+0.0)."
            )
        if team_correct:
            reward += 0.3
            feedback_parts.append(f"✅ Correct team '{issue['team']}' (+0.3).")
        else:
            feedback_parts.append(
                f"❌ Wrong team '{action.team}'. Correct: '{issue['team']}' (+0.0)."
            )
        if priority_correct:
            reward += 0.2
            feedback_parts.append(f"✅ Correct priority '{issue['priority']}' (+0.2).")
        else:
            feedback_parts.append(
                f"❌ Wrong priority '{action.priority}'. Correct: '{issue['priority']}' (+0.0)."
            )
        # Partial credit for suggested action
        reward += action_score * 0.2
        feedback_parts.append(
            f"💡 Suggested action score: {action_score:.1f} "
            f"(keywords matched: {keyword_hits}/{len(issue['fix_keywords'])}) "
            f"(+{action_score * 0.2:.2f})."
        )

    reward = max(0.01, min(0.99, reward))
    return round(reward, 4), " | ".join(feedback_parts)


# ──────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS
# ──────────────────────────────────────────────────────────────
class GithubIssueTriageEnvironment(Environment):
    """
    GitHub Issue Triage Environment.

    The agent is shown a GitHub issue and must triage it correctly.
    Each episode = one issue to triage.
    The agent gets one step per episode (one decision per issue).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "easy"
        self._current_issue: dict = {}
        self._done: bool = False
        self._last_reward: float = 0.0
        self._last_feedback: str = ""

    def reset(
        self,
        task_id: str = "easy",
        issue_index: int = -1,
    ) -> GithubIssueTriageObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: 'easy', 'medium', or 'hard'
            issue_index: which issue to use (-1 = random)
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id if task_id in TASK_DESCRIPTIONS else "easy"
        self._done = False
        self._last_reward = 0.0
        self._last_feedback = ""

        # Pick an issue
        if issue_index >= 0 and issue_index < len(ISSUES):
            self._current_issue = ISSUES[issue_index]
        else:
            self._current_issue = random.choice(ISSUES)

        return self._build_observation(
            feedback="Read the issue carefully and make your triage decision.",
            done=False,
        )

    def step(
        self, action: GithubIssueTriageAction
    ) -> GithubIssueTriageObservation:
        """
        The agent submits its triage decision.
        One step per episode — grade it and mark done.
        """
        self._state.step_count += 1

        # Auto-reset if no issue loaded (e.g. fresh container, no reset called)
        if not self._current_issue:
                    self.reset()

        if self._done:
            return self._build_observation(
                feedback="Episode already finished. Call reset() to start a new one.",
                done=True,
            )

        # Grade the action
        reward, feedback = grade_action(action, self._current_issue, self._task_id)
        self._last_reward = reward
        self._last_feedback = feedback
        self._done = True  # one issue = one step = episode done

        return self._build_observation(feedback=feedback, done=True, reward=reward)

    def _build_observation(
        self,
        feedback: str,
        done: bool,
        reward: float = 0.0,
    ) -> GithubIssueTriageObservation:
        """Helper to build observation from current state."""
        issue = self._current_issue
        return GithubIssueTriageObservation(
            issue_id=issue.get("issue_id", ""),
            issue_title=issue.get("issue_title", ""),
            issue_body=issue.get("issue_body", ""),
            repo_name=issue.get("repo_name", ""),
            author=issue.get("author", ""),
            existing_comments=issue.get("existing_comments", []),
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
            last_reward=self._last_reward,
            feedback=feedback,
            done=done,
            step_number=self._state.step_count,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state