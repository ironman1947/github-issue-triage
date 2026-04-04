# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the GitHub Issue Triage Environment.
"""

import sys
import os

# Make sure the parent directory is on the path so models.py is found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from models import GithubIssueTriageAction, GithubIssueTriageObservation
from server.github_issue_triage_environment import GithubIssueTriageEnvironment

# Create the FastAPI app
app = create_app(
    GithubIssueTriageEnvironment,
    GithubIssueTriageAction,
    GithubIssueTriageObservation,
    env_name="github_issue_triage",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # call main() with no args for default, or pass parsed args
    main()