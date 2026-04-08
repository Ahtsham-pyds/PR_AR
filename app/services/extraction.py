import re
from datetime import date

from app.schemas import FormSuggestion


def _extract_budget(text: str) -> float | None:
    match = re.search(r"(?:budget|cost)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", text, re.I)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _extract_due_date(text: str) -> str | None:
    match = re.search(r"(?:due|deadline|target date)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})", text, re.I)
    if not match:
        return None
    try:
        date.fromisoformat(match.group(1))
        return match.group(1)
    except ValueError:
        return None


def _extract_labeled(text: str, label: str) -> str | None:
    pattern = rf"{label}\s*[:\-]\s*(.+)"
    match = re.search(pattern, text, re.I)
    if match:
        return match.group(1).strip()
    return None


def suggest_from_text(text: str) -> FormSuggestion:
    notes: list[str] = []

    client_name = _extract_labeled(text, "client")
    project_title = _extract_labeled(text, "project")
    budget = _extract_budget(text)
    due_date = _extract_due_date(text)

    summary = text.strip()
    if len(summary) > 500:
        summary = f"{summary[:497]}..."
        notes.append("Long input truncated to 500 chars for preview.")

    filled_fields = sum(
        1
        for item in [client_name, project_title, summary if summary else None, budget, due_date]
        if item is not None
    )
    confidence = round(min(1.0, 0.15 + (filled_fields * 0.17)), 2)

    if not client_name:
        notes.append("Client name not found. Use 'Client: <name>' for better extraction.")
    if not project_title:
        notes.append("Project title not found. Use 'Project: <title>' for better extraction.")
    if not budget:
        notes.append("Budget not found. Use 'Budget: 25000'.")

    return FormSuggestion(
        client_name=client_name,
        project_title=project_title,
        requirements_summary=summary if summary else None,
        budget=budget,
        due_date=due_date,
        confidence=confidence,
        notes=notes,
    )
