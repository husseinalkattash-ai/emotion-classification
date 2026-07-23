"""Ensures the project root is importable as `src` during tests.

The presence of this file at the repo root puts the root on ``sys.path`` for
pytest, so ``import src`` works without installing the package.
"""
