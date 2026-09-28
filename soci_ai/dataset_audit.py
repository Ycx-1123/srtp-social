from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatasetAuditError(ValueError):
    """Raised when a dataset cannot be audited safely."""


class DuplicateGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    sha256: str
    splits: list[str]
    paths: list[str]


class DatasetAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: str
    generated_at: str
    algorithm: str = "sha256"
    total_files: int = Field(ge=0)
    split_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    empty_file_count: int = Field(ge=0)
    empty_files: list[str]
    same_relative_path_collisions: int = Field(ge=0)
    duplicate_hash_groups: int = Field(ge=0)
    cross_split_exact_duplicates: int = Field(ge=0)
    duplicate_groups: list[DuplicateGroup]
    independent_evaluation_valid: bool
    conclusion: str

    def to_markdown(self) -> str:
        validity = "有效" if self.independent_evaluation_valid else "无效（存在跨划分完全重复内容）"
        lines = [
            "# SOCI-AI 数据集完整性审计",
            "",
            f"- 生成时间：`{self.generated_at}`",
            f"- 数据根目录：`{self.root}`",
            f"- 指纹算法：`{self.algorithm}`（流式读取）",
            f"- 文件总数：**{self.total_files}**",
            f"- 独立评估结论：**{validity}**",
            "",
            "## 划分统计",
            "",
            "| 划分 | 文件数 | 类别数 |",
            "|---|---:|---:|",
        ]
        for split, count in self.split_counts.items():
            lines.append(f"| {split} | {count} | {len(self.class_counts.get(split, {}))} |")
        lines.extend(
            [
                "",
                "## 完整性检查",
                "",
                f"- 跨划分完全重复文件：**{self.cross_split_exact_duplicates}**",
                f"- 跨划分重复哈希组：**{self.duplicate_hash_groups}**",
                f"- 同相对路径跨划分碰撞：**{self.same_relative_path_collisions}**",
                f"- 空文件：**{self.empty_file_count}**",
                "",
                "## 解释",
                "",
                self.conclusion,
                "",
                "> 本报告仅陈述文件级证据，不删除、移动或重命名任何原始数据。",
            ]
        )
        return "\n".join(lines) + "\n"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root: Path) -> DatasetAudit:
    root = root.resolve()
    if not root.exists():
        raise DatasetAuditError(f"dataset root does not exist: {root}")
    if not root.is_dir():
        raise DatasetAuditError(f"dataset root is not a directory: {root}")

    split_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not split_dirs:
        raise DatasetAuditError(f"dataset root contains no split directories: {root}")

    split_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    empty_files: list[str] = []
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    relative_splits: dict[str, set[str]] = defaultdict(set)

    for split_dir in split_dirs:
        split = split_dir.name
        files = sorted(path for path in split_dir.rglob("*") if path.is_file())
        split_counts[split] = len(files)
        counts: dict[str, int] = defaultdict(int)
        for path in files:
            relative = path.relative_to(split_dir).as_posix()
            class_name = relative.split("/", 1)[0]
            counts[class_name] += 1
            if path.stat().st_size == 0:
                empty_files.append(path.relative_to(root).as_posix())
            hashes[_sha256(path)].append((split, path.relative_to(root).as_posix()))
            relative_splits[relative].add(split)
        class_counts[split] = dict(sorted(counts.items()))

    duplicate_groups = []
    for digest, items in sorted(hashes.items()):
        splits = sorted({split for split, _ in items})
        if len(splits) > 1:
            duplicate_groups.append(
                DuplicateGroup(
                    sha256=digest,
                    splits=splits,
                    paths=sorted(path for _, path in items),
                )
            )

    collisions = sum(1 for splits in relative_splits.values() if len(splits) > 1)
    leaked_files = 0
    for group in duplicate_groups:
        counts = [sum(path.startswith(f"{split}/") for path in group.paths) for split in group.splits]
        leaked_files += len(group.paths) - max(counts)
    valid = not duplicate_groups
    conclusion = (
        "未发现训练集与验证/测试集共享完全相同的文件内容；文件级独立评估条件成立。"
        if valid
        else "训练集与验证/测试集存在 SHA-256 完全相同的内容，当前验证结果不能作为独立泛化性能证据；建议按内容哈希重新划分后再训练与评估。"
    )
    return DatasetAudit(
        root=str(root),
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_files=sum(split_counts.values()),
        split_counts=split_counts,
        class_counts=class_counts,
        empty_file_count=len(empty_files),
        empty_files=empty_files,
        same_relative_path_collisions=collisions,
        duplicate_hash_groups=len(duplicate_groups),
        cross_split_exact_duplicates=leaked_files,
        duplicate_groups=duplicate_groups,
        independent_evaluation_valid=valid,
        conclusion=conclusion,
    )
