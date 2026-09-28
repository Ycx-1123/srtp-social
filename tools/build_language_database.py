"""Build the bundled, indexed SQLite scenario library from its reviewable JSON.

The JSON is the only authored source. Running this script does not download data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "text", "label", "category", "context", "explanation", "rewrite",
          "weight", "provenance", "source_role")


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def build_database(source, output):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    corpus = json.loads(raw.decode("utf-8"))
    rows = corpus["examples"]
    counts = Counter(row["label"] for row in rows)
    if counts != {"bias": 300, "control": 60}:
        raise ValueError("Expected 300 original bias examples and 60 controls")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Example IDs must be unique")
    if len({normalize(row["text"]) for row in rows}) != len(rows):
        raise ValueError("Normalized example texts must be unique")
    for row in rows:
        if any(key not in row for key in FIELDS):
            raise ValueError("Every example must contain the complete schema")
        if row["provenance"] != "synthetic_authored" or row["source_role"] != "original_sample":
            raise ValueError("This library accepts only original authored scenario examples")
        if not 0 <= row["weight"] <= 1 or (row["label"] == "control" and row["weight"] != 0):
            raise ValueError("Invalid demonstration weight")
    metadata = dict(corpus["metadata"], source_sha256=hashlib.sha256(raw).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".language-", suffix=".sqlite", dir=output.parent,
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE examples (
                    id TEXT PRIMARY KEY, text TEXT NOT NULL, label TEXT NOT NULL
                        CHECK(label IN ('bias', 'control')),
                    category TEXT NOT NULL, context TEXT NOT NULL,
                    explanation TEXT NOT NULL, rewrite TEXT NOT NULL,
                    weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
                    provenance TEXT NOT NULL, source_role TEXT NOT NULL,
                    normalized_text TEXT NOT NULL UNIQUE
                );
                CREATE INDEX examples_label_category ON examples(label, category);
                PRAGMA user_version = 1;
            """)
            connection.executemany("INSERT INTO metadata VALUES (?, ?)",
                                   [(key, json.dumps(value, ensure_ascii=False))
                                    for key, value in metadata.items()])
            connection.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   [tuple(row[key] for key in FIELDS) + (normalize(row["text"]),)
                                    for row in rows])
        # Close before replacement so building also works on Windows.
        connection.close()
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "resources/language/social_bias_examples.json")
    parser.add_argument("--output", type=Path, default=ROOT / "resources/language/social_bias_examples.sqlite")
    args = parser.parse_args()
    counts = build_database(args.source, args.output)
    print(f"Built {args.output}: {counts['bias']} bias examples, {counts['control']} controls")


if __name__ == "__main__":
    main()
