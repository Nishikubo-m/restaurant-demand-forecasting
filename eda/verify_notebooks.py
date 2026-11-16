"""Execute EDA notebooks independently with this Python environment's Jupyter kernel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_snapshot(directory: Path) -> dict[str, dict[str, str | int]]:
    """Fingerprint top-level input files, including ZIP archives, without extracting them."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    result: dict[str, dict[str, str | int]] = {}
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed while hashing: {path}")
        result[path.name] = {"bytes": after.st_size, "sha256": digest}
    return result


def changed_inputs(before: dict, after: dict) -> dict[str, list[str]]:
    """Identify top-level input files added, removed or changed during execution."""
    old, new = set(before), set(after)
    return {
        "added": sorted(new - old),
        "removed": sorted(old - new),
        "changed": sorted(name for name in old & new if before[name] != after[name]),
    }


def verify_notebook_hashes(paths: list[Path], results: list[dict]) -> None:
    """Recheck every notebook, including earlier ones, before producing a report."""
    if len(paths) != len(results):
        raise RuntimeError("Not all notebooks were verified")
    for path, result in zip(paths, results):
        if not path.is_file() or sha256(path) != result["sha256"]:
            raise RuntimeError(f"Notebook changed after verification: {path}")


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_report(path: Path, report: dict) -> None:
    """Replace a report only after its complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebook paths; defaults to all eda/*.ipynb")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds per code cell")
    parser.add_argument("--report", type=Path, help="Optional JSON report (written only after all notebooks pass)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    notebooks = [p.resolve() for p in args.notebooks] if args.notebooks else sorted((ROOT / "eda").glob("*.ipynb"))
    if not notebooks:
        parser.error("No notebooks found")
    for path in notebooks:
        if not path.is_file() or path.suffix != ".ipynb":
            parser.error(f"Notebook not found: {path}")

    input_dir = ROOT / "input"
    if args.report:
        report_path = args.report.resolve()
        protected = {p.resolve() for p in notebooks} | {
            Path(__file__).resolve(), (ROOT / "pyproject.toml").resolve(), (ROOT / "uv.lock").resolve()
        }
        if report_path.is_relative_to(input_dir.resolve()) or report_path in protected:
            parser.error("--report cannot overwrite an input, notebook, or project configuration")

    results = []
    started = time.monotonic()
    inputs_before = input_snapshot(input_dir)
    config_paths = {"pyproject.toml": ROOT / "pyproject.toml", "uv.lock": ROOT / "uv.lock",
                    "verifier": Path(__file__).resolve()}
    config_before = {name: sha256(path) if path.is_file() else None for name, path in config_paths.items()}
    # Put an ephemeral kernel spec first so a user's globally installed Python kernel
    # cannot silently replace the Python environment used to invoke this script.
    original_jupyter_path = os.environ.get("JUPYTER_PATH")
    with tempfile.TemporaryDirectory(prefix="restaurant-eda-kernel-") as temp:
        kernel_dir = Path(temp) / "kernels" / "restaurant-eda"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                        "display_name": "Restaurant EDA", "language": "python"}),
            encoding="utf-8",
        )
        os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, [temp, original_jupyter_path]))
        try:
            for index, path in enumerate(notebooks, 1):
                content = path.read_bytes()
                before = hashlib.sha256(content).hexdigest()
                notebook = nbformat.reads(content.decode("utf-8"), as_version=4)
                nbformat.validate(notebook)
                # Execute a copy in memory; never replace the notebook's saved outputs.
                NotebookClient(
                    notebook,
                    timeout=args.timeout,
                    kernel_name="restaurant-eda",
                    resources={"metadata": {"path": str(ROOT)}},
                ).execute()
                cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
                errors = [out for cell in cells for out in cell.outputs if out.output_type == "error"]
                if errors or any(cell.execution_count is None for cell in cells):
                    raise RuntimeError(f"Notebook has failed or unexecuted code cells: {path}")
                if sha256(path) != before:
                    raise RuntimeError(f"Notebook changed during verification: {path}")
                try:
                    label = str(path.relative_to(ROOT))
                except ValueError:
                    label = str(path)
                results.append({"notebook": label, "sha256": before, "code_cells": len(cells)})
                print(f"[{index}/{len(notebooks)}] OK {label} ({len(cells)} code cells)", flush=True)
        finally:
            if original_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = original_jupyter_path

    verify_notebook_hashes(notebooks, results)
    changes = changed_inputs(inputs_before, input_snapshot(input_dir))
    if any(changes.values()):
        raise RuntimeError(f"Input files changed during verification: {changes}")
    config_after = {name: sha256(path) if path.is_file() else None for name, path in config_paths.items()}
    if config_after != config_before:
        raise RuntimeError("Project configuration or verifier changed during verification")

    elapsed = time.monotonic() - started
    print(f"All {len(results)} notebooks passed in {elapsed:.1f}s.", flush=True)
    if args.report:
        report = {
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(("numpy", "pandas", "scipy", "pyarrow", "matplotlib",
                                           "seaborn", "nbclient", "nbformat", "ipykernel")),
            "kernel": "temporary kernelspec using the invoking Python environment",
            "elapsed_seconds": round(elapsed, 2),
            "uv_lock_sha256": config_before["uv.lock"],
            "pyproject_sha256": config_before["pyproject.toml"],
            "verifier_sha256": config_before["verifier"],
            "input_files": inputs_before,
            "notebooks": results,
        }
        write_report(args.report, report)
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
