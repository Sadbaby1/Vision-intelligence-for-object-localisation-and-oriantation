"""
Project folders, resolved independently of the current working directory, so every
command works from any folder on any machine.

The project root is, in order of priority:
1. the ``VISION_INTELLIGENCE_ROOT`` environment variable, if set;
2. the nearest folder at or above the current directory containing this project's ``pyproject.toml``;
3. the repository the package is installed from (``uv sync`` installs it in editable mode);
4. the current directory.

Paths stored in files (e.g. ``index.csv``) are relative to the root, so they survive
moving or cloning the repository.
"""

import os
from pathlib import Path

ENV_VAR = "VISION_INTELLIGENCE_ROOT"
_PROJECT_MARKER = 'name = "vision-intelligence"'


def _is_project_root(path):
    pyproject = path / "pyproject.toml"
    return pyproject.is_file() and _PROJECT_MARKER in pyproject.read_text(encoding="utf-8", errors="ignore")


def find_project_root():
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _is_project_root(candidate):
            return candidate
    installed_from = Path(__file__).resolve().parents[2]  # src/vision_intelligence/paths.py -> repo
    if _is_project_root(installed_from):
        return installed_from
    return cwd


ROOT = find_project_root()
DATA = ROOT / "data"
CAPTURES = DATA / "captures"
ROBOFLOW = DATA / "roboflow"
DATASETS = DATA / "datasets"
CONFIG = ROOT / "config"  # small settings files that are tracked in git
PART_DIMENSIONS = CONFIG / "part_dimensions.json"
MODEL = ROOT / "model"
OUTPUTS = ROOT / "outputs"


def from_root(path):
    """Absolute path for a path stored relative to the project root (absolute paths are kept)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def to_root_relative(path):
    """POSIX path relative to the project root when inside it, otherwise the absolute POSIX path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
