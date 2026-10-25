from pathlib import Path


ACCEPTANCE_IDS = {f"AC-{number:03d}" for number in range(1, 22)}


def test_all_srs_acceptance_criteria_are_mapped():
    report = Path(__file__).resolve().parents[1] / "results" / "acceptance_report.md"
    text = report.read_text(encoding="utf-8")
    assert ACCEPTANCE_IDS <= set(text[i:i + 6] for i in range(len(text)) if text[i:i + 3] == "AC-")

