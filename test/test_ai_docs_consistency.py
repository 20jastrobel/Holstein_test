from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

AGENTS_MD = REPO_ROOT / "AGENTS.md"
README_MD = REPO_ROOT / "README.md"
RUN_GUIDE_MD = REPO_ROOT / "pipelines" / "run_guide.md"
LLM_CONTEXT_MD = REPO_ROOT / "docs" / "LLM_RESEARCH_CONTEXT.md"
CHANGELOG_MD = REPO_ROOT / "docs" / "AGENT_CONTRACT_CHANGELOG.md"

CANONICAL_DOCS = (AGENTS_MD, README_MD, RUN_GUIDE_MD)
RUNBOOK_DOCS = (README_MD, RUN_GUIDE_MD)

STALE_SCRIPT_PATTERNS = (
    r"run_L_drive_accurate\.sh",
    r"run_scaling_preset_L2_L6\.sh",
)
STALE_RUN_GUIDE_ALIAS_PATTERN = r"PIPELINE_RUN_GUIDE\.md"
STRICT_GATE_PATTERN = r"(?<!\d)1e-7(?!\d)"


"""legacy_alias_section := text block between Appendix A heading and next heading."""
def _legacy_alias_section_bounds(text: str) -> tuple[int, int]:
    start_token = "### Appendix A. Legacy alias map (non-canonical; back-compat only)"
    start = text.find(start_token)
    assert start >= 0, f"Missing Appendix A heading in {LLM_CONTEXT_MD}"

    next_heading = "\n### Detailed data-flow by active pipeline"
    end = text.find(next_heading, start)
    assert end >= 0, f"Missing end marker for Appendix A block in {LLM_CONTEXT_MD}"
    return start, end


def _find_line_numbers(text: str, pattern: str) -> list[int]:
    return [text.count("\n", 0, m.start()) + 1 for m in re.finditer(pattern, text)]


def _line_at(text: str, line_no: int) -> str:
    lines = text.splitlines()
    return lines[line_no - 1] if 1 <= line_no <= len(lines) else ""


def test_no_stale_script_aliases_in_canonical_docs() -> None:
    for doc in CANONICAL_DOCS:
        text = doc.read_text(encoding="utf-8")
        for pattern in STALE_SCRIPT_PATTERNS:
            assert not re.search(pattern, text), f"Found stale alias '{pattern}' in canonical doc {doc}"

    llm_text = LLM_CONTEXT_MD.read_text(encoding="utf-8")
    start, end = _legacy_alias_section_bounds(llm_text)
    for pattern in STALE_SCRIPT_PATTERNS:
        for m in re.finditer(pattern, llm_text):
            assert start <= m.start() < end, (
                f"Stale alias '{pattern}' found outside Appendix A in {LLM_CONTEXT_MD}"
            )


def test_no_stale_pipeline_run_guide_alias() -> None:
    for doc in CANONICAL_DOCS:
        text = doc.read_text(encoding="utf-8")
        assert not re.search(STALE_RUN_GUIDE_ALIAS_PATTERN, text), (
            f"Found stale run-guide alias in canonical doc {doc}"
        )

    llm_text = LLM_CONTEXT_MD.read_text(encoding="utf-8")
    start, end = _legacy_alias_section_bounds(llm_text)
    line_numbers = _find_line_numbers(llm_text, STALE_RUN_GUIDE_ALIAS_PATTERN)
    for line_no in line_numbers:
        char_index = sum(len(line) + 1 for line in llm_text.splitlines()[: line_no - 1])
        assert start <= char_index < end, (
            f"PIPELINE_RUN_GUIDE.md alias found outside Appendix A at line {line_no}"
        )


def test_run_guide_single_top_title() -> None:
    text = RUN_GUIDE_MD.read_text(encoding="utf-8")
    matches = re.findall(r"^# Hubbard Pipeline Run Guide\s*$", text, flags=re.MULTILINE)
    assert len(matches) == 1, "run_guide.md must contain exactly one top-level title"


def test_no_smoke_or_experimental_command_blocks_in_canonical_runbooks() -> None:
    disallowed_block_markers = (
        r"CFQM smoke commands",
        r"# NOTE \(Hubbard smoke\)",
        r"experimental",
    )
    for doc in RUNBOOK_DOCS:
        text = doc.read_text(encoding="utf-8")
        for marker in disallowed_block_markers:
            assert not re.search(marker, text, flags=re.IGNORECASE), (
                f"Found non-production command-block marker '{marker}' in {doc}"
            )


def test_default_hard_gate_documented_for_final_conventional_vqe() -> None:
    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    runguide_text = RUN_GUIDE_MD.read_text(encoding="utf-8")
    assert "Default Hard Gate" in agents_text and "< 1e-4" in agents_text, (
        "AGENTS.md must explicitly declare default hard gate '< 1e-4'"
    )
    assert "Default Hard Gate" in runguide_text and "< 1e-4" in runguide_text, (
        "run_guide.md must explicitly declare default hard gate '< 1e-4'"
    )


def test_1e7_mentions_are_strict_mode_scoped_in_canonical_docs() -> None:
    for doc in CANONICAL_DOCS:
        text = doc.read_text(encoding="utf-8")
        for line_no in _find_line_numbers(text, STRICT_GATE_PATTERN):
            neighborhood = [
                _line_at(text, max(1, line_no - 1)),
                _line_at(text, line_no),
                _line_at(text, line_no + 1),
            ]
            context = " ".join(neighborhood).lower()
            assert "strict" in context, (
                f"Unscoped 1e-7 mention at {doc}:{line_no}; strict mode context required"
            )


def test_conflict_stop_rule_exists_in_agents() -> None:
    text = AGENTS_MD.read_text(encoding="utf-8").lower()
    assert "stop and ask the user before proceeding" in text, (
        "AGENTS.md must include conflict-stop rule text for policy-vs-code mismatch"
    )


def test_changelog_exists_and_is_linked_from_canonical_docs() -> None:
    assert CHANGELOG_MD.exists(), "Missing docs/AGENT_CONTRACT_CHANGELOG.md"
    for doc in CANONICAL_DOCS:
        text = doc.read_text(encoding="utf-8")
        assert "AGENT_CONTRACT_CHANGELOG.md" in text, (
            f"{doc} must link to docs/AGENT_CONTRACT_CHANGELOG.md"
        )
