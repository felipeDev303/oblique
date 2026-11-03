"""Assemble the folder that is uploaded to the Hugging Face Space: only what the app needs.

    uv run python deploy/space/build.py dist/space

The folder holds the package, the deck, the product page, the results page, the Space config, the
lock file and the Dockerfile. No `.env`, no data, no runs and no cache: the key is a Space secret.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["pyproject.toml", "uv.lock", "LICENSE", "configs/space.yaml", "deck/LICENSE"]
TREES = {"src/oblique": "*.py", "deck/cards": "*.yaml", "web/product": "*", "web/results": "*"}
NEVER = (".env", ".db", ".jsonl")


def build(out: Path, root: Path = Path(".")) -> list[Path]:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    copied = []
    sources = [root / f for f in FILES]
    for tree, pattern in TREES.items():
        if (root / tree).is_dir():
            sources += [p for p in sorted((root / tree).rglob(pattern)) if p.is_file()]
    for source in sources:
        relative = source.relative_to(root)
        if "__pycache__" in relative.parts or relative.name.endswith(NEVER):
            continue
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(relative)
    for name in ("Dockerfile", "README.md"):
        shutil.copyfile(HERE / name, out / name)
        copied.append(Path(name))
    return copied


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python deploy/space/build.py <empty output folder>")
    copied = build(Path(argv[0]))
    print(f"{len(copied)} files in {argv[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
