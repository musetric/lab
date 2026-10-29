# Musetric Lab

## Context

- This is a Python project managed with `uv`: Python comes from `requires-python` and uv from `required-version` in `pyproject.toml`.
- Prefer existing `uv run` scripts and tools; use free-form commands only when no script exists.
- The repository checks of `@musetric/checks` (English letters, notices, links) come from npm: `npm ci` installs them and `scripts/check.py` runs them; a file where another language is part of its meaning goes to `checks.english.except` of `package.json`.
- The rules of the organization apply here and are not repeated: the [code](https://github.com/musetric/headquarters/blob/main/practices/code.md) of every repository, [what may be said about audio](https://github.com/musetric/headquarters/blob/main/practices/public-texts.md#material) and the other [practices](https://github.com/musetric/headquarters/blob/main/AGENTS.md#map). The rules below are this repository's own.

## Repository Map

- `musetric_toolkit/separate_audio`: audio source separation (MelBandRoformer, MDX-Net, FFmpeg).
- `musetric_toolkit/transcribe_audio`: speech transcription via WhisperX.
- `musetric_toolkit/rhythm_audio`: tempo, beat, and rhythm analysis.
- `musetric_toolkit/rhythm_bench`: the beats the metronome would click, their counts of musetric/musetric#1108 and listening renders (`musetric-rhythm-bench`).
- `musetric_toolkit/distill`: the recipe that distilled the vocals student from Duality: corpus, labels, fits, training, release and its evaluation (`musetric-distill`).
- `musetric_toolkit/key_audio`: musical key detection.
- `musetric_toolkit/chords_audio`: chord detection and recognition.
- `musetric_toolkit/pitch_audio`: the reference pitch track the Musetric pitch bench scores against (RMVPE, CREPE, SwiftF0 and FCPE combined).
- `musetric_toolkit/pitch_zoo`: pitch models, ground truth and the metrics of the bench that check that reference (`musetric-pitch-zoo`).
- `musetric_toolkit/common`: shared utilities (logging, paths, env, model files).
- `scripts/`: dev tooling (lint checks); `scripts/onnx/`: the ONNX export of each model, one directory with its readme per model.

## Ground Rules

**Syntax and imports**

- Follow the existing ruff, black, and isort config. Do not fight it.
- Use `snake_case` for modules, functions, variables, and directories, and `PascalCase` for classes.
- Use absolute, explicit imports (e.g. `from musetric_toolkit.common.logger import ...`): no relative imports (ruff bans them) and no `import *`.

**Type Hints**

- Add type hints to function signatures.
- Use `from __future__ import annotations` when needed for forward references.
- Prefer built-in generics (`list`, `dict`) over `typing.List`, `typing.Dict`.

**CLI Entry Points**

- Each audio module exposes a `main_cli.py` with a `main_cli()` function registered via `[project.scripts]` in `pyproject.toml`.
- Use `argparse`, and accept `--log-level` as `debug|info|warn|error` (default: `info`).

## Runtime Boundaries

- All source code lives under `musetric_toolkit/`.
- The vendored code that the black, isort and ruff excludes of `pyproject.toml` list, such as `musetric_toolkit/chords_audio/chordmini/`, is not reformatted.

## Before Finishing

If code was changed, run the relevant checks before finishing; `uv run python scripts/check.py --fix` auto-fixes linting issues.
