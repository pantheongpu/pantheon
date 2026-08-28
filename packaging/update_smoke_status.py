#!/usr/bin/env python3
"""Render matrix-job results into a marked README status table.

Defaults render the daily container-smoke table; the toolkit compile matrix
reuses the same result.json artifact format with its own markers and labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


START_MARKER = "<!-- DAILY_SMOKE_STATUS:START -->"
END_MARKER = "<!-- DAILY_SMOKE_STATUS:END -->"
DEFAULT_TITLE = "Daily container smoke status"
DEFAULT_DESCRIPTION = (
    "These are short mock-backend checks of every workload inside disposable containers."
)
DEFAULT_SUBJECT = "Operating system"
STATUS_LABELS = {
    "success": "Passed",
    "failure": "Failed",
    "cancelled": "Cancelled",
    "skipped": "Skipped",
}
STATUS_ICONS = {
    "success": "✅",
    "failure": "❌",
    "cancelled": "⚪",
    "skipped": "⚪",
}


def load_results(results_dir: Path) -> list[dict[str, str]]:
    """Load one status payload from each matrix-job artifact."""
    results = []
    for path in sorted(results_dir.rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid smoke status file: {path}") from exc
        name = str(payload.get("name", "")).strip()
        status = str(payload.get("status", "")).strip().lower()
        if not name or status not in STATUS_LABELS:
            raise ValueError(f"Invalid smoke status payload: {path}")
        results.append({"name": name, "status": status})
    if not results:
        raise ValueError("No operating-system smoke-test results were found.")
    return sorted(results, key=lambda item: item["name"].lower())


def render_status(
    results: list[dict[str, str]],
    completed_at: str,
    run_url: str,
    start_marker: str = START_MARKER,
    end_marker: str = END_MARKER,
    title: str = DEFAULT_TITLE,
    description: str = DEFAULT_DESCRIPTION,
    subject: str = DEFAULT_SUBJECT,
) -> str:
    """Return the bounded README fragment between the status markers."""
    passed = sum(item["status"] == "success" for item in results)
    lines = [
        start_marker,
        f"### {title}",
        "",
        f"Last completed: [{completed_at}]({run_url})<br>",
        f"Result: **{passed}/{len(results)} passed**. {description}",
        "",
        f"| {subject} | Latest result |",
        "| --- | --- |",
    ]
    for item in results:
        status = item["status"]
        lines.append(f"| {item['name']} | {STATUS_ICONS[status]} {STATUS_LABELS[status]} |")
    lines.extend([end_marker, ""])
    return "\n".join(lines)


def update_readme(readme: Path, fragment: str, start_marker: str = START_MARKER, end_marker: str = END_MARKER) -> None:
    """Replace exactly the generated status region, preserving all other README text."""
    contents = readme.read_text(encoding="utf-8")
    start = contents.find(start_marker)
    end = contents.find(end_marker)
    if start < 0 or end < start:
        raise ValueError(f"README is missing status markers: {start_marker}")
    end += len(end_marker)
    readme.write_text(contents[:start] + fragment.rstrip() + contents[end:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--completed-at", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--start-marker", default=START_MARKER)
    parser.add_argument("--end-marker", default=END_MARKER)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    args = parser.parse_args()
    fragment = render_status(
        load_results(args.results_dir),
        args.completed_at,
        args.run_url,
        start_marker=args.start_marker,
        end_marker=args.end_marker,
        title=args.title,
        description=args.description,
        subject=args.subject,
    )
    update_readme(args.readme, fragment, start_marker=args.start_marker, end_marker=args.end_marker)


if __name__ == "__main__":
    main()
