from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soci_ai.dataset_audit import DatasetAuditError, audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="只读审计 split/class/file 数据集的跨划分泄漏")
    parser.add_argument("dataset", type=Path, help="数据集根目录")
    parser.add_argument("--json", type=Path, required=True, dest="json_path")
    parser.add_argument("--markdown", type=Path, required=True, dest="markdown_path")
    args = parser.parse_args()
    try:
        report = audit_dataset(args.dataset)
    except DatasetAuditError as exc:
        parser.error(str(exc))
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    args.markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    verdict = "VALID" if report.independent_evaluation_valid else "INVALID: CROSS-SPLIT DUPLICATES"
    print(f"Audited {report.total_files} files — {verdict}")
    print(f"JSON: {args.json_path.resolve()}")
    print(f"Markdown: {args.markdown_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
