# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GitHub Issue Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import GithubIssueTriageAction, GithubIssueTriageObservation


class GithubIssueTriageEnv(
    EnvClient[GithubIssueTriageAction, GithubIssueTriageObservation, State]
):
    """
    Client for the GitHub Issue Triage Environment.

    Connects to the environment server via WebSocket and sends
    triage decisions, receiving graded observations back.

    Example (easy task):
        >>> with GithubIssueTriageEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.issue_title)
        ...     action = GithubIssueTriageAction(label="bug")
        ...     result = env.step(action)
        ...     print(result.reward)

    Example (hard task):
        >>> with GithubIssueTriageEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     action = GithubIssueTriageAction(
        ...         label="bug",
        ...         team="backend",
        ...         priority="high",
        ...         suggested_action="Check memory limit in file_handler.py line 42"
        ...     )
        ...     result = env.step(action)
        ...     print(result.reward)   # 0.0 to 1.0
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: GithubIssueTriageAction) -> Dict:
        """
        Convert GithubIssueTriageAction to JSON payload for the step request.

        Args:
            action: The agent's triage decision

        Returns:
            Dictionary ready for JSON encoding
        """
        return {
            "label": action.label,
            "team": action.team,
            "priority": action.priority,
            "suggested_action": action.suggested_action,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GithubIssueTriageObservation]:
        """
        Parse the server's JSON response into a typed StepResult.

        Args:
            payload: Raw JSON response from the server

        Returns:
            StepResult containing the observation and reward
        """
        obs_data = payload.get("observation", {})

        observation = GithubIssueTriageObservation(
            # Issue content
            issue_id=obs_data.get("issue_id", ""),
            issue_title=obs_data.get("issue_title", ""),
            issue_body=obs_data.get("issue_body", ""),
            repo_name=obs_data.get("repo_name", ""),
            author=obs_data.get("author", ""),
            existing_comments=obs_data.get("existing_comments", []),

            # Task context
            task_id=obs_data.get("task_id", "easy"),
            task_description=obs_data.get("task_description", ""),

            # Feedback
            last_reward=obs_data.get("last_reward", 0.0),
            feedback=obs_data.get("feedback", ""),

            # Episode info
            done=payload.get("done", False),
            step_number=obs_data.get("step_number", 0),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse the server's state response into a State object.

        Args:
            payload: Raw JSON from the /state endpoint

        Returns:
            State with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )