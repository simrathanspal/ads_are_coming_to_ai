"""Tests that verify import logging is present in all src modules."""

import ast
from pathlib import Path


SRC_DIR = Path(__file__).parent.parent / "src"


def _has_logging_import(filepath: Path) -> bool:
    tree = ast.parse(filepath.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "logging" for alias in node.names):
                return True
    return False


def test_constrained_decoder_has_logging():
    assert _has_logging_import(SRC_DIR / "constrained_decoder.py")


def test_index_builder_has_logging():
    assert _has_logging_import(SRC_DIR / "index_builder.py")


def test_trie_has_logging():
    assert _has_logging_import(SRC_DIR / "trie.py")
