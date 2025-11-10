# GitHub Copilot Instructions for Titli

## General Guidelines
- Prefer clear, modular Python code following PEP8 style.
- Use type hints and docstrings for all public functions and classes.
- When generating code, prefer using the existing `titli.fe` and `titli.ids` modules for feature extraction and IDS models, respectively.
- For new feature extractors, subclass from the base classes in `titli.fe`.
- For new IDS models, subclass from the base classes in `titli.ids`.
- Use utility functions from `titli.utils` where possible.
- When writing examples, place them in the `examples/` directory and use relative imports.
- For tests, use clear, minimal examples and avoid duplicating logic from the main codebase.

## Naming and Structure
- Use descriptive, lowercase file and folder names with underscores (e.g., `after_image.py`).
- Place images and documentation assets in `assets/images/`.
- Keep all user-facing documentation in `README.md` and `INSTRUCTIONS.md`.

## Documentation
- Always update the pipeline diagram in `assets/images/` and reference it in the `README.md` when the architecture changes.
- Add docstrings to all new classes, methods, and functions.
- When adding new features, update `INSTRUCTIONS.md` with usage details.

## Dependencies
- Use only the dependencies listed in `pyproject.toml` and `setup.py` unless absolutely necessary.
- For deep learning, prefer PyTorch (`torch`, `torchvision`).

## Security and Data
- Do not include sensitive data or credentials in code or documentation.
- Use sample or dummy data in examples.

## Contribution
- Follow the structure and conventions of the existing codebase.
- Add new scripts to `examples/` and new modules to the appropriate subpackage.
- Update documentation and add usage examples for new features.

## Copilot Behavior
- When suggesting code, prefer using and extending the existing Titli framework rather than writing from scratch.
- Avoid generating duplicate code; reuse and reference existing modules.
- When in doubt, prompt the user to clarify requirements or point to relevant files.

---

_This file guides GitHub Copilot to generate code and documentation that is consistent with the Titli project’s architecture and standards._
