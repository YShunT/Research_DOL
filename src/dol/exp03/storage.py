"""原子的なJSON保存、入力同一性、条件単位の再開、実行ロック。"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
import subprocess
import tempfile


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False makes missing/undefined diagnostics explicitly null.
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity() -> dict:
    paths = sorted(Path("src").rglob("*.py"))
    paths = [p for p in paths if "tests" not in p.parts and "__pycache__" not in p.parts]
    paths += [Path("pyproject.toml"), Path("uv.lock"), Path("experiments/exp03/strategy.md")]
    hashes = {str(p): sha256(p) for p in paths}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src",
                                    "pyproject.toml", "uv.lock", "experiments/exp03/strategy.md"],
                                   text=True).strip()
    return {"commit": commit, "dirty": bool(dirty), "sha256": hashes}


@contextlib.contextmanager
def suite_lock(root: Path):
    directory = root / "arrays"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another exp03 process is running in {root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def logger(root: Path) -> logging.Logger:
    log = logging.getLogger("dol.exp03")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    root.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.FileHandler(root / "run.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


class Records:
    """One split's completed cases. Atomic checkpoints allow case-level resume."""

    def __init__(self, path: Path):
        self.path = path
        self.data = read_json(path) if path.exists() else {"status": "running", "cases": {}}

    def contains(self, key: str) -> bool:
        return key in self.data["cases"]

    def save(self, key: str, record: dict) -> None:
        self.data["cases"][key] = record
        self.data["status"] = "running"
        write_json(self.path, self.data)

    def complete(self) -> None:
        self.data["status"] = "completed"
        write_json(self.path, self.data)
