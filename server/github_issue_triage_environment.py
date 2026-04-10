"""
GitHub Issue Triage Environment — OpenEnv Hackathon
Team Astra.AI: Om Chougule (Lead), Shraman Patil

Real-world task: An AI agent reads GitHub issues and makes structured triage
decisions (labelling, team routing, priority scoring, fix suggestion).

Tasks:
  easy   — assign correct label (bug / feature / docs / question)
  medium — assign correct label + correct team
  hard   — assign label + team + priority + suggest a concrete fix action

Grader:
  easy   → label correct = 1.0,  wrong = 0.0
  medium → label (0.5) + team (0.5)
  hard   → label (0.30) + team (0.30) + priority (0.20) + fix keywords (0.20)
"""

import random
import uuid
from typing import Optional

try:
    from models import (
        GithubIssueTriageAction,
        GithubIssueTriageObservation,
        GithubIssueTriageState,
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import (
        GithubIssueTriageAction,
        GithubIssueTriageObservation,
        GithubIssueTriageState,
    )

from openenv.core.env_server import Environment

# ── Issue Dataset ─────────────────────────────────────────────────────────────
# Each issue has hidden ground-truth fields used only by the grader.
# The agent never sees these — it only sees title, body, author, comments.
ISSUE_DATASET = [
    # ── BUG issues ────────────────────────────────────────────────────────────
    {
        "id": "#101",
        "title": "NullPointerException on login with Google SSO",
        "body": (
            "After the latest deploy (v2.4.1) clicking 'Sign in with Google' throws a "
            "NullPointerException in the auth middleware. Stack trace:\n"
            "  AuthMiddleware.java:87 — userToken is null\n"
            "Reproducible on Chrome 124 and Firefox 125. Safari unaffected."
        ),
        "author": "mobile_dev_03",
        "comments": [
            "Confirmed on staging as well.",
            "Seems related to the OAuth library upgrade in #98.",
        ],
        "label": "bug",
        "team": "backend",
        "priority": "critical",
        "fix_keywords": ["oauth", "token", "null", "auth", "middleware", "sso"],
    },
    {
        "id": "#102",
        "title": "Add dark mode support to the dashboard",
        "body": (
            "Many users have requested a dark mode for the dashboard UI. "
            "This would improve usability during night-time usage and reduce eye strain. "
            "Please consider adding a toggle in the settings page."
        ),
        "author": "ux_designer_01",
        "comments": ["Would love this!", "+1 from our team."],
        "label": "feature",
        "team": "frontend",
        "priority": "medium",
        "fix_keywords": ["dark", "theme", "css", "toggle", "settings", "ui"],
    },
    {
        "id": "#103",
        "title": "API docs missing authentication section",
        "body": (
            "The REST API documentation at docs.example.com/api does not include "
            "any examples of how to pass Bearer tokens or API keys. New integrators "
            "are confused. We need a complete authentication section with curl examples."
        ),
        "author": "enterprise_customer_42",
        "comments": ["I spent 2 hours on this. Please fix ASAP."],
        "label": "docs",
        "team": "docs",
        "priority": "high",
        "fix_keywords": ["documentation", "api", "authentication", "bearer", "token", "example"],
    },
    {
        "id": "#104",
        "title": "How do I export data to CSV?",
        "body": (
            "I'm trying to export my project data to a CSV file but I can't find the option "
            "anywhere in the UI. Is there a way to do this? I checked the docs but couldn't find it."
        ),
        "author": "new_user_99",
        "comments": ["Check Settings → Export.", "Also see the FAQ section."],
        "label": "question",
        "team": "docs",
        "priority": "low",
        "fix_keywords": ["export", "csv", "download", "settings", "guide"],
    },
    {
        "id": "#105",
        "title": "How do I configure custom environment variables?",
        "body": (
            "I am trying to configure custom environment variables for my deployment "
            "but I cannot find any documentation on this. "
            "Is there a config file or a CLI flag I should use?"
        ),
        "author": "new_contributor_22",
        "comments": ["Check the openenv.yaml file.", "Also see the README deployment section."],
        "label": "question",
        "team": "docs",
        "priority": "low",
        "fix_keywords": ["environment", "variable", "config", "yaml", "cli", "documentation"],
    },
    {
        "id": "#106",
        "title": "ML model inference latency spikes to 10s every 5 minutes",
        "body": (
            "Our production ML pipeline shows periodic latency spikes: every ~5 minutes "
            "inference time jumps from 200ms to 10s for ~30 seconds, then recovers. "
            "CPU and memory look normal. GPU utilization drops during the spike. "
            "Logs show 'CUDA context switch' warnings."
        ),
        "author": "ml_infra_lead",
        "comments": [
            "Possibly GC pauses in the Python runtime?",
            "Or CUDA memory fragmentation after large batches.",
        ],
        "label": "bug",
        "team": "ml",
        "priority": "high",
        "fix_keywords": ["cuda", "gpu", "latency", "inference", "memory", "fragmentation", "profiling"],
    },
    {
        "id": "#107",
        "title": "Add Prometheus metrics endpoint for monitoring",
        "body": (
            "We need a /metrics endpoint that exposes Prometheus-compatible metrics: "
            "request count, p50/p95/p99 latency, error rate, active connections. "
            "This is needed for our SRE team to set up alerting."
        ),
        "author": "sre_engineer_07",
        "comments": ["This would also help with capacity planning.", "FastAPI has a plugin for this."],
        "label": "feature",
        "team": "devops",
        "priority": "high",
        "fix_keywords": ["prometheus", "metrics", "monitoring", "endpoint", "alerting", "fastapi"],
    },
    {
        "id": "#108",
        "title": "Docker container runs out of memory on startup",
        "body": (
            "The Docker container exits with OOM (Out of Memory) error during startup "
            "even on machines with 16GB RAM. docker run -p 8000:8000 my-env:latest fails immediately. "
            "No issues before the last release."
        ),
        "author": "devops_lead_05",
        "comments": ["Try setting --memory=8g flag.", "Check for memory leaks in init."],
        "label": "bug",
        "team": "devops",
        "priority": "critical",
        "fix_keywords": ["memory", "oom", "docker", "startup", "leak", "container", "profile"],
    },
    {
        "id": "#109",
        "title": "Add support for SAML 2.0 single sign-on",
        "body": (
            "Our enterprise customers require SAML 2.0 SSO for compliance. "
            "Currently only OAuth2/OIDC is supported. We need SAML metadata exchange, "
            "IdP-initiated login, and SP-initiated login flows."
        ),
        "author": "enterprise_sales_03",
        "comments": ["Blocker for 3 enterprise deals.", "Okta and Azure AD are the main IdPs needed."],
        "label": "feature",
        "team": "backend",
        "priority": "high",
        "fix_keywords": ["saml", "sso", "authentication", "enterprise", "okta", "idp"],
    },
    {
        "id": "#110",
        "title": "What Python versions are supported?",
        "body": (
            "I want to know which Python versions are officially supported. "
            "I'm running Python 3.9 and getting import warnings. "
            "The README doesn't mention minimum Python version."
        ),
        "author": "open_source_contrib_11",
        "comments": ["Python 3.10+ is recommended.", "See pyproject.toml for constraints."],
        "label": "question",
        "team": "docs",
        "priority": "low",
        "fix_keywords": ["python", "version", "compatibility", "readme", "support", "documentation"],
    },
    {
        "id": "#111",
        "title": "Race condition in concurrent session handling causes data corruption",
        "body": (
            "Under load (>50 concurrent users), we see data from one user's session "
            "leaking into another user's response. This is a critical data privacy bug. "
            "Reproducible with locust at 50 VUs. Happens ~3% of requests."
        ),
        "author": "security_researcher_01",
        "comments": [
            "This is a serious security vulnerability.",
            "Likely a thread-safety issue in the session store.",
        ],
        "label": "bug",
        "team": "backend",
        "priority": "critical",
        "fix_keywords": ["race", "concurrency", "session", "thread", "lock", "mutex", "data", "privacy"],
    },
    {
        "id": "#112",
        "title": "Add batch prediction API endpoint",
        "body": (
            "Currently predictions must be sent one at a time. "
            "We need a POST /predict/batch endpoint that accepts an array of inputs "
            "and returns an array of results. This would reduce API call overhead by 10x."
        ),
        "author": "data_scientist_08",
        "comments": ["This would unblock our pipeline.", "+1, very needed for production use."],
        "label": "feature",
        "team": "ml",
        "priority": "medium",
        "fix_keywords": ["batch", "prediction", "api", "endpoint", "array", "throughput"],
    },
]

VALID_LABELS = {"bug", "feature", "docs", "question"}
VALID_TEAMS = {"frontend", "backend", "ml", "devops", "docs"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}


# ── Grader ────────────────────────────────────────────────────────────────────
def grade_action(
    action: "GithubIssueTriageAction",
    issue: dict,
    task_id: str,
) -> tuple[float, str]:
    """
    Returns (reward: float in [0,1], feedback: str).

    easy   → label correct = 1.0 | wrong = 0.0
    medium → label (0.5) + team (0.5)
    hard   → label (0.30) + team (0.30) + priority (0.20) + fix keywords (0.20)
    """
    if not issue:
        return 0.0, "No issue loaded — call reset() first."

    label_correct = (action.label or "").lower().strip() == issue["label"]
    team_correct = (action.team or "").lower().strip() == issue["team"]
    priority_correct = (action.priority or "").lower().strip() == issue["priority"]

    # Fix suggestion quality: keyword overlap with ground truth
    fix_score = 0.0
    if action.suggested_action:
        text = action.suggested_action.lower()
        keywords = issue.get("fix_keywords", [])
        if keywords:
            hits = sum(1 for kw in keywords if kw in text)
            fix_score = min(hits / max(len(keywords) * 0.4, 1), 1.0)

    # ── Easy ──────────────────────────────────────────────────────────────
    if task_id == "easy":
        if label_correct:
            return 1.0, f"✅ Correct label '{action.label}'! Full marks."
        else:
            return 0.0, (
                f"❌ Wrong label '{action.label}'. "
                f"Correct answer: '{issue['label']}'."
            )

    # ── Medium ────────────────────────────────────────────────────────────
    if task_id == "medium":
        reward = 0.0
        parts = []
        if label_correct:
            reward += 0.5
            parts.append("✅ label correct (+0.5)")
        else:
            parts.append(f"❌ label wrong (got '{action.label}', expected '{issue['label']}')")
        if team_correct:
            reward += 0.5
            parts.append("✅ team correct (+0.5)")
        else:
            parts.append(f"❌ team wrong (got '{action.team}', expected '{issue['team']}')")
        return round(reward, 4), " | ".join(parts)

    # ── Hard ──────────────────────────────────────────────────────────────
    if task_id == "hard":
        reward = 0.0
        parts = []
        if label_correct:
            reward += 0.30
            parts.append("✅ label (+0.30)")
        else:
            parts.append(f"❌ label (got '{action.label}', exp '{issue['label']}')")
        if team_correct:
            reward += 0.30
            parts.append("✅ team (+0.30)")
        else:
            parts.append(f"❌ team (got '{action.team}', exp '{issue['team']}')")
        if priority_correct:
            reward += 0.20
            parts.append("✅ priority (+0.20)")
        else:
            parts.append(f"❌ priority (got '{action.priority}', exp '{issue['priority']}')")
        if fix_score > 0:
            partial = round(fix_score * 0.20, 4)
            reward += partial
            parts.append(f"✅ fix suggestion (+{partial:.2f})")
        else:
            parts.append("❌ fix suggestion (no relevant keywords)")
        return round(reward, 4), " | ".join(parts)

    return 0.0, f"Unknown task_id '{task_id}'"


# ── Environment ───────────────────────────────────────────────────────────────
class GithubIssueTriageEnvironment(Environment):
    """
    OpenEnv-compliant environment for GitHub Issue Triage.
    One episode = one issue to triage. Clean state on every reset().
    """

    # Each request creates a fresh env with isolated state — safe for concurrency
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = GithubIssueTriageState()
        self._current_issue: dict = {}
        self._task_id: str = "easy"
        self._done: bool = False
        # Separate random pools per task so issues cycle without repetition
        self._pools: dict[str, list] = {tid: [] for tid in ("easy", "medium", "hard")}

    # ── Internal helpers ──────────────────────────────────────────────────
    def _pick_issue(self, task_id: str) -> dict:
        """Return a random issue, refilling the pool when exhausted."""
        pool = self._pools[task_id]
        if not pool:
            pool = list(ISSUE_DATASET)
            random.shuffle(pool)
            self._pools[task_id] = pool
        return pool.pop()

    def _build_observation(self, issue: dict, task_id: str,
                           feedback: str = "", last_reward: float = 0.0,
                           step_number: int = 0) -> "GithubIssueTriageObservation":
        if task_id == "easy":
            task_desc = (
                "TASK (Easy): Read the GitHub issue carefully and assign the correct LABEL.\n"
                "Valid labels: 'bug', 'feature', 'docs', 'question'.\n"
                "Only the 'label' field in your action will be graded."
            )
        elif task_id == "medium":
            task_desc = (
                "TASK (Medium): Read the GitHub issue and assign the correct LABEL and TEAM.\n"
                "Valid labels: 'bug', 'feature', 'docs', 'question'.\n"
                "Valid teams: 'frontend', 'backend', 'ml', 'devops', 'docs'.\n"
                "Both label and team fields will be graded (0.5 each)."
            )
        else:
            task_desc = (
                "TASK (Hard): Read the GitHub issue and assign LABEL, TEAM, PRIORITY, "
                "and SUGGESTED_ACTION.\n"
                "Valid labels: 'bug', 'feature', 'docs', 'question'.\n"
                "Valid teams: 'frontend', 'backend', 'ml', 'devops', 'docs'.\n"
                "Valid priorities: 'critical', 'high', 'medium', 'low'.\n"
                "All four fields are graded: label (30%) + team (30%) + priority (20%) + fix (20%)."
            )

        if not feedback:
            feedback = "Read the issue carefully and make your triage decision."

        return GithubIssueTriageObservation(
            issue_id=issue["id"],
            issue_title=issue["title"],
            issue_body=issue["body"],
            repo_name="meta-pytorch/OpenEnv",
            author=issue["author"],
            existing_comments=issue["comments"],
            task_id=task_id,
            task_description=task_desc,
            last_reward=last_reward,
            feedback=feedback,
            step_number=step_number,
        )

    # ── OpenEnv API ───────────────────────────────────────────────────────
    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "GithubIssueTriageObservation":
        """Start a new episode. Picks a random issue for the given task."""
        if task_id and task_id in ("easy", "medium", "hard"):
            self._task_id = task_id
        else:
            self._task_id = "easy"

        if seed is not None:
            random.seed(seed)

        self._current_issue = self._pick_issue(self._task_id)
        self._done = False
        self._state = GithubIssueTriageState(
            episode_id=str(uuid.uuid4()),
            task_id=self._task_id,
            issue_id=self._current_issue["id"],
        )
        return self._build_observation(
            self._current_issue, self._task_id, step_number=0
        )

    def step(
        self, action: "GithubIssueTriageAction", task_id: Optional[str] = None, **kwargs,
    ) -> "GithubIssueTriageObservation":
        """Grade the agent's triage decision and return result."""
        self._state.step_count += 1

        # In stateless HTTP mode, each call creates a fresh env.
        # Use task_id from request if provided, otherwise fall back to instance default.
        effective_task_id = task_id if task_id in ("easy", "medium", "hard") else self._task_id

        # Auto-reset if called before reset() (stateless HTTP mode)
        if not self._current_issue:
            self.reset(task_id=effective_task_id)

        if self._done:
            return self._build_observation(
                self._current_issue,
                self._task_id,
                feedback="Episode already done. Call reset() to start a new episode.",
                last_reward=0.0,
                step_number=self._state.step_count,
            )

        reward, feedback = grade_action(action, self._current_issue, self._task_id)
        self._done = True  # single-step episode
        self._state.total_reward += reward
        self._state.last_reward = reward

        obs = self._build_observation(
            self._current_issue,
            self._task_id,
            feedback=feedback,
            last_reward=reward,
            step_number=self._state.step_count,
        )
        # Set reward/done on the base Observation fields so the server serializes them
        obs.reward = reward
        obs.done = True
        return obs

    def state(self) -> "GithubIssueTriageState":
        return self._state

    def close(self) -> None:
        """Clean up resources (nothing to clean for this environment)."""
        pass