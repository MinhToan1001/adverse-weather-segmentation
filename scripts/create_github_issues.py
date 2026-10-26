"""Create the 40-task Gate A-E backlog on GitHub from docs/issues_backlog.csv.

Standard library only. Token resolution order:
  1. GH_TOKEN / GITHUB_TOKEN environment variable
  2. `git credential fill` for https://github.com (same token git uses)

Usage:
    python scripts/create_github_issues.py            # dry run, prints the plan
    python scripts/create_github_issues.py --apply    # create labels, milestones, issues

Idempotent by issue title and milestone/label name: re-running skips what exists.
Never prints the token.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs" / "issues_backlog.csv"

GATE_MILESTONES = {
    "A": ("Gate A - Scope & protocol", "2026-10-28", "Protocol, repo skeleton, backlog, risks"),
    "B": ("Gate B - Data contract verified", "2026-10-31", "Licences, manifests, audit, leakage check"),
    "C": ("Gate C - Reproducible baseline", "2026-11-08", "Smoke + full Cityscapes baseline, artifacts"),
    "D": ("Gate D - DG evaluation + ablation", "2026-11-16", "One-shot ACDC eval, ablation, analysis"),
    "E": ("Gate E - Deliverables & handoff", "2026-11-24", "Demo, report, slides, release"),
}

LABELS = {
    "area/protocol": ("5319E7", "Scope, DG protocol, approvals"),
    "area/data": ("0E8A16", "Datasets, labels, manifests, audit"),
    "area/model": ("1D76DB", "Model and training code"),
    "area/eval": ("B60205", "Metrics and evaluation"),
    "area/research": ("6F42C1", "Literature, gap, method proposal"),
    "area/ablation": ("D93F0B", "Controlled ablation"),
    "area/infra": ("0052CC", "CI, repo, environment, releases"),
    "area/docs": ("0075CA", "Documentation, report, slides"),
    "area/demo": ("FBCA04", "Demo and visual assets"),
    "priority/p0": ("B60205", "Gate-blocking"),
    "priority/p1": ("D93F0B", "Important, not gate-blocking"),
    "owner/a": ("1F77B4", "Lead: model, training, compute"),
    "owner/b": ("FF7F0E", "Data, evaluation, docs"),
}


def token():
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        out = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True, text=True, env=env, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"^password=(.+)$", out, re.MULTILINE)
    return match.group(1).strip() if match else None


def repo_from_git():
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return match.group(1) if match else None


class GitHub:
    def __init__(self, token_value, repo):
        self.token = token_value
        self.repo = repo

    def _request(self, method, path, payload=None, retries=4):
        url = path if path.startswith("http") else f"{API}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "opencode-agent",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if data:
            headers["Content-Type"] = "application/json"
        for attempt in range(retries):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read().decode()
                    return json.loads(body) if body else None
            except urllib.error.HTTPError as error:
                body = error.read().decode(errors="replace")
                if error.code == 422:
                    raise
                if error.code in (502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{method} {path} -> {error.code}: {body[:400]}") from error
        raise RuntimeError(f"{method} {path}: retries exhausted")

    def get(self, path):
        return self._request("GET", path)

    def get_all(self, path):
        items, page = [], 1
        separator = "&" if "?" in path else "?"
        while True:
            batch = self.get(f"{path}{separator}per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def post(self, path, payload):
        return self._request("POST", path, payload)

    def patch(self, path, payload):
        return self._request("PATCH", path, payload)


def load_tasks():
    with BACKLOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task_id", "gate", "title", "owner", "reviewer", "labels", "due", "depends",
                "summary", "acceptance", "artifacts"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"{BACKLOG} is missing columns: {required - set(rows[0] or {})}")
    return rows


def build_body(task):
    bullets = [part.strip() for part in task["acceptance"].split("|") if part.strip()]
    artifacts = [part.strip() for part in task["artifacts"].split("|") if part.strip()]
    depends = [part.strip() for part in task["depends"].replace(",", ";").split(";") if part.strip()]
    lines = [
        f"**Gate:** {task['gate']} | **Due:** {task['due']} | "
        f"**Owner:** @{task['owner']} | **Reviewer:** @{task['reviewer']}",
        "",
        "## Context",
        task["summary"],
        "",
        "## Definition of done",
        *[f"- [ ] {item}" for item in bullets],
        "",
        "## Artifacts",
        *[f"- `{item}`" for item in artifacts],
        "",
        "## Dependencies",
        *([f"- {item}" for item in depends] or ["- None"]),
        "",
        "---",
        "Protocol: `docs/experiment_protocol.md` | Backlog: `docs/issues_backlog.csv`",
    ]
    return "\n".join(lines)


def ensure_labels(gh, apply):
    existing = {label["name"] for label in gh.get_all(f"/repos/{gh.repo}/labels")}
    created = 0
    for name, (color, description) in LABELS.items():
        if name in existing:
            continue
        if not apply:
            print(f"  + label {name}")
            continue
        gh.post(f"/repos/{gh.repo}/labels", {"name": name, "color": color, "description": description})
        created += 1
    return created


def ensure_milestones(gh, apply):
    existing = {m["title"]: m["number"] for m in gh.get_all(f"/repos/{gh.repo}/milestones?state=all")}
    numbers = {}
    for gate, (title, due, description) in GATE_MILESTONES.items():
        if title in existing:
            numbers[gate] = existing[title]
        elif apply:
            milestone = gh.post(f"/repos/{gh.repo}/milestones", {
                "title": title,
                "state": "open",
                "due_on": f"{due}T00:00:00Z",
                "description": description,
            })
            numbers[gate] = milestone["number"]
            print(f"  + milestone {title} (#{milestone['number']})")
        else:
            print(f"  + milestone {title} (due {due})")
    return numbers


def ensure_issues(gh, tasks, milestones, apply):
    existing = {issue["title"] for issue in gh.get_all(f"/repos/{gh.repo}/issues?state=all")}
    created = skipped = 0
    for task in tasks:
        if task["title"] in existing:
            skipped += 1
            continue
        labels = [label.strip() for label in task["labels"].split(";") if label.strip()]
        payload = {
            "title": task["title"],
            "body": build_body(task),
            "assignees": [task["owner"]],
            "labels": labels,
        }
        if task["gate"] in milestones:
            payload["milestone"] = milestones[task["gate"]]
        if not apply:
            print(f"  + {task['task_id']}: {task['title']} -> @{task['owner']} [{task['due']}]")
            continue
        try:
            issue = gh.post(f"/repos/{gh.repo}/issues", payload)
            print(f"  #{issue['number']} {task['task_id']} @{task['owner']}")
            created += 1
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")[:300]
            print(f"  ! {task['task_id']} failed: {error.code} {body}", file=sys.stderr)
    return created, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="owner/name (default: origin remote)")
    parser.add_argument("--apply", action="store_true", help="create instead of dry run")
    args = parser.parse_args()

    gh_token = token()
    if not gh_token:
        raise SystemExit("No GitHub token: set GH_TOKEN or run `gh auth login` / git credential manager.")
    repo = args.repo or repo_from_git()
    if not repo:
        raise SystemExit("Cannot detect repository; pass --repo owner/name.")

    gh = GitHub(gh_token, repo)
    tasks = load_tasks()
    user = gh.get("/user")
    print(f"repo={repo} as={user['login']} tasks={len(tasks)} apply={args.apply}")

    labels_created = ensure_labels(gh, args.apply)
    milestones = ensure_milestones(gh, args.apply)
    created, skipped = ensure_issues(gh, tasks, milestones, args.apply)

    if args.apply:
        print(f"done: labels_created={labels_created} issues_created={created} issues_skipped={skipped}")
    else:
        print("dry run only; pass --apply to create")


if __name__ == "__main__":
    main()
