from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .api import create_app
from .config import Settings


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(root)
    uvicorn.run(
        create_app(settings),
        host=os.environ.get("SOCI_HOST", "127.0.0.1"),
        port=int(os.environ.get("SOCI_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
