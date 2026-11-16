"""Check whether a saved EDA verification report still matches the current project."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from eda.verify_notebooks import changed_inputs, input_snapshot, package_versions, sha256

ROOT = Path(__file__).resolve().parents[1]


def check_report(report: dict, root: Path, *, allow_partial: bool = False) -> list[str]:
    """Return discrepancies; an empty list means the recorded state still matches."""
    root = root.resolve()
    required = {"input_files", "notebooks", "uv_lock_sha256", "pyproject_sha256", "verifier_sha256", "packages", "python"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"Report lacks provenance fields {sorted(missing)}; run verification again")
    if not isinstance(report["input_files"], dict) or not isinstance(report["notebooks"], list) or not report["notebooks"]:
        raise ValueError("Report has an invalid input or notebook list")

    problems: list[str] = []
    changes = changed_inputs(report["input_files"], input_snapshot(root / "input"))
    for kind, names in changes.items():
        if names:
            problems.append(f"input {kind}: {', '.join(names)}")

    for label, path, field in [
        ("pyproject.toml", root / "pyproject.toml", "pyproject_sha256"),
        ("uv.lock", root / "uv.lock", "uv_lock_sha256"),
        ("verifier", root / "eda" / "verify_notebooks.py", "verifier_sha256"),
    ]:
        actual = sha256(path) if path.is_file() else None
        if actual != report[field]:
            problems.append(f"{label} changed or is missing")

    recorded: set[Path] = set()
    for item in report["notebooks"]:
        if not isinstance(item, dict) or not isinstance(item.get("notebook"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("Report has an invalid notebook entry")
        name = Path(item["notebook"])
        path = (root / name).resolve() if not name.is_absolute() else name.resolve()
        if path in recorded:
            problems.append(f"duplicate notebook in report: {name}")
        recorded.add(path)
        if not path.is_file() or sha256(path) != item["sha256"]:
            problems.append(f"notebook changed or is missing: {name}")
    if not allow_partial:
        current = {p.resolve() for p in (root / "eda").glob("*.ipynb")}
        unverified = sorted(str(p.relative_to(root)) for p in current - recorded)
        if unverified:
            problems.append(f"notebooks not covered by report: {', '.join(unverified)}")

    if report["python"] != sys.version.split()[0]:
        problems.append(f"Python version changed: {report['python']} -> {sys.version.split()[0]}")
    if not isinstance(report["packages"], dict):
        raise ValueError("Report has an invalid package list")
    current_packages = package_versions(tuple(report["packages"]))
    for name, old in report["packages"].items():
        if current_packages[name] != old:
            problems.append(f"package {name} changed: {old} -> {current_packages[name]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=ROOT / "output/eda/notebook_verification.json")
    parser.add_argument("--allow-partial", action="store_true", help="Allow a report covering only selected notebooks")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        problems = check_report(report, ROOT, allow_partial=args.allow_partial)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Cannot check report: {exc}\n")
    print(f"Report: {args.report}; completed at: {report.get('completed_at', 'unknown')}")
    print(f"Notebook entries: {len(report['notebooks'])}; input files: {len(report['input_files'])}")
    if report.get("platform") != platform.platform():
        print(f"Note: platform differs: {report.get('platform', 'unknown')} -> {platform.platform()}")
    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}")
        parser.exit(1, "Verification report does not match the current project. Run verification again.\n")
    print("OK: recorded inputs, notebooks, project configuration and Python package versions match.")
    if args.allow_partial:
        print("Note: --allow-partial permits notebooks not included in the report.")


if __name__ == "__main__":
    main()
