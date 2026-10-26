import csv
from collections import Counter
from datetime import date
from pathlib import Path

BACKLOG = Path(__file__).resolve().parents[1] / "docs" / "issues_backlog.csv"
EXPECTED_GATES = {"A": 8, "B": 8, "C": 9, "D": 8, "E": 7}
OWNERS = {"MinhToan1001", "khangcules10"}


def load():
    with BACKLOG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_backlog_has_40_tasks_split_across_gates():
    rows = load()
    assert len(rows) == 40
    assert Counter(row["gate"] for row in rows) == Counter(EXPECTED_GATES)


def test_task_ids_are_unique_and_match_gate():
    rows = load()
    ids = [row["task_id"] for row in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["task_id"].startswith(row["gate"])
        assert row["task_id"][1:].isdigit()


def test_every_task_has_owner_reviewer_labels_and_due_date():
    for row in load():
        assert row["owner"] in OWNERS
        assert row["reviewer"] in OWNERS
        assert row["owner"] != row["reviewer"]
        assert len([label for label in row["labels"].split(";") if label.strip()]) >= 2
        assert date.fromisoformat(row["due"])


def test_every_task_has_context_acceptance_and_artifacts():
    for row in load():
        assert len(row["summary"]) >= 20
        assert len([part for part in row["acceptance"].split("|") if part.strip()]) >= 2
        assert row["artifacts"].strip()
