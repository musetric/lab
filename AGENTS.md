# Musetric Lab

## Context

- This is a Python project managed with `uv`: Python comes from `requires-python` and uv from `required-version` in `pyproject.toml`.
- Prefer existing `uv run` scripts and tools; use free-form commands only when no script exists.
- The repository checks of `@musetric/checks` (English letters, notices, links) come from npm: `npm ci` installs them and `scripts/check.py` runs them; a file where another language is part of its meaning goes to `checks.english.except` of `package.json`.

## Commits & PR Titles

- Name commits and PR titles per `.github/prTitle.config.yml`.

## Repository Map

- `musetric_toolkit/separate_audio`: audio source separation (BSRoformer, MelBandRoformer, MDX-Net, FFmpeg).
- `musetric_toolkit/transcribe_audio`: speech transcription via WhisperX.
- `musetric_toolkit/rhythm_audio`: tempo, beat, and rhythm analysis.
- `musetric_toolkit/key_audio`: musical key detection.
- `musetric_toolkit/chords_audio`: chord detection and recognition.
- `musetric_toolkit/pitch_audio`: the reference pitch track the Musetric pitch bench scores against (RMVPE, CREPE, SwiftF0 and FCPE combined).
- `musetric_toolkit/pitch_zoo`: pitch models, ground truth and the metrics of the bench that check that reference (`musetric-pitch-zoo`).
- `musetric_toolkit/common`: shared utilities (logging, paths, env, model files).
- `scripts/`: dev tooling (lint checks, ONNX export helpers).

## Ground Rules

**Syntax**

- Follow the existing ruff, black, and isort config. Do not fight it.
- Use `snake_case` for modules, functions, variables, and directories.
- Use `PascalCase` for classes.
- Do not use relative imports (`ruff` bans them — `ban-relative-imports = "all"`).

**Files**

- Keep files domain-specific and self-contained: do not create catch-all files like `types.py` or `utils.py`; keep an entity's types and functions together.
- If a file contains several small domain entities, do not interleave them; keep each entity grouped, and place the more foundational entity before the next one.

**Imports And Exports**

- Use absolute imports only (e.g. `from musetric_toolkit.common.logger import ...`).
- Prefer explicit imports over `import *`.
- Group imports: stdlib, third-party, local — separated by a blank line (isort `black` profile).

**Type Hints**

- Add type hints to function signatures.
- Use `from __future__ import annotations` when needed for forward references.
- Prefer built-in generics (`list`, `dict`) over `typing.List`, `typing.Dict`.

**CLI Entry Points**

- Each audio module exposes a `main_cli.py` with a `main_cli()` function registered via `[project.scripts]` in `pyproject.toml`.
- Use `argparse` for CLI argument parsing.
- Accept `--log-level` as `debug|info|warn|error` (default: `info`).

## Runtime Boundaries

- All source code lives under `musetric_toolkit/`.
- `musetric_toolkit/chords_audio/chordmini/` is excluded from black, isort, and ruff formatting.
- ONNX-related export scripts live in `scripts/onnx/`, one directory per model
  family: `roformer/` (vocal separation), `whisper/` (ASR), `chordmini/` (chord
  recognition), `skey/` (key detection), `beat_this/` (rhythm).

## Before Finishing

If code was changed, run the relevant checks before finishing.

- `uv run python scripts/check.py --fix`: auto-fix linting issues.
