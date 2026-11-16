"""配布ZIPの公式スクリプトを使う提出ファイル変換・検証CLI。"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from src.config import INPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="日次予測。--validate時は変換後ファイル")
    parser.add_argument("--output", type=Path, help="変換後の出力先")
    parser.add_argument("--validate", action="store_true", help="公式スクリプトで提出形式を検証")
    args = parser.parse_args()
    if args.validate and args.output:
        parser.error("--validateと--outputは同時に指定できません")
    if not args.validate and not args.output:
        parser.error("変換時は--outputが必要です")
    if not args.predictions.is_file():
        parser.error(f"ファイルがありません: {args.predictions}")
    if args.output and args.output.resolve() == args.predictions.resolve():
        parser.error("入力と出力には別のファイルを指定してください")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT_DIR / "submission_scripts.zip") as archive, tempfile.TemporaryDirectory(prefix="submission-") as temp:
        script = Path(temp) / "convert_submission.py"
        script.write_bytes(archive.read("submission_scripts/convert_submission.py"))
        command = [sys.executable, str(script), "--predictions", str(args.predictions)]
        command += ["--validate"] if args.validate else ["--output", str(args.output)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
