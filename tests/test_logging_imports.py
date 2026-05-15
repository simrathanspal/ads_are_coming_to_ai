"""Tests that verify logging is importable from all src modules."""

import importlib
import logging
import types


def _get_module_globals(module_name: str) -> dict:
    mod = importlib.import_module(module_name)
    return vars(mod)


def test_trie_has_logging_import():
    import src.trie as trie_mod
    assert hasattr(trie_mod, "logging"), "src.trie must import logging"
    assert trie_mod.logging is logging


def test_index_builder_has_logging_import():
    import src.index_builder as ib_mod
    assert hasattr(ib_mod, "logging"), "src.index_builder must import logging"
    assert ib_mod.logging is logging


def test_constrained_decoder_has_logging_import():
    # constrained_decoder imports torch/transformers; just check the module source
    import ast
    import pathlib

    src_file = pathlib.Path(__file__).parents[1] / "src" / "constrained_decoder.py"
    tree = ast.parse(src_file.read_text())
    imports = [
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    ]
    assert "logging" in imports, "src/constrained_decoder.py must contain 'import logging'"
