"""
Tests that each src module emits an INFO log with the filename and timestamp
when the module is loaded (compiled/imported).
"""

import importlib
import sys
import unittest


class TestTrieLogging(unittest.TestCase):
    def test_logger_attribute_exists(self):
        import src.trie as module
        self.assertTrue(hasattr(module, "logger"))
        self.assertEqual(module.logger.name, "src.trie")

    def test_logs_on_import(self):
        import src.trie as module
        with self.assertLogs("src.trie", level="INFO") as ctx:
            importlib.reload(module)
        messages = [r.getMessage() for r in ctx.records]
        self.assertTrue(
            any("compiled" in m.lower() for m in messages),
            f"Expected 'compiled' in log messages, got: {messages}",
        )
        self.assertTrue(
            any("trie" in m.lower() for m in messages),
            f"Expected filename in log messages, got: {messages}",
        )

    def test_log_contains_timestamp(self):
        import src.trie as module
        with self.assertLogs("src.trie", level="INFO") as ctx:
            importlib.reload(module)
        messages = [r.getMessage() for r in ctx.records]
        # Timestamp format: YYYY-MM-DD HH:MM:SS
        import re
        timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertTrue(
            any(timestamp_pattern.search(m) for m in messages),
            f"Expected timestamp in log messages, got: {messages}",
        )


class TestIndexBuilderLogging(unittest.TestCase):
    def test_logger_attribute_exists(self):
        import src.index_builder as module
        self.assertTrue(hasattr(module, "logger"))
        self.assertEqual(module.logger.name, "src.index_builder")

    def test_logs_on_import(self):
        import src.index_builder as module
        with self.assertLogs("src.index_builder", level="INFO") as ctx:
            importlib.reload(module)
        messages = [r.getMessage() for r in ctx.records]
        self.assertTrue(
            any("compiled" in m.lower() for m in messages),
            f"Expected 'compiled' in log messages, got: {messages}",
        )
        self.assertTrue(
            any("index_builder" in m.lower() for m in messages),
            f"Expected filename in log messages, got: {messages}",
        )

    def test_log_contains_timestamp(self):
        import src.index_builder as module
        with self.assertLogs("src.index_builder", level="INFO") as ctx:
            importlib.reload(module)
        messages = [r.getMessage() for r in ctx.records]
        import re
        timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertTrue(
            any(timestamp_pattern.search(m) for m in messages),
            f"Expected timestamp in log messages, got: {messages}",
        )


_HAS_TORCH = False
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    pass


@unittest.skipUnless(_HAS_TORCH, "torch not installed — skipping constrained_decoder tests")
class TestConstrainedDecoderLogging(unittest.TestCase):
    def test_logger_attribute_exists(self):
        import src.constrained_decoder as module
        self.assertTrue(hasattr(module, "logger"))
        self.assertEqual(module.logger.name, "src.constrained_decoder")

    def test_logs_on_import(self):
        import src.constrained_decoder as module
        with self.assertLogs("src.constrained_decoder", level="INFO") as ctx:
            importlib.reload(module)
        messages = [r.getMessage() for r in ctx.records]
        self.assertTrue(
            any("compiled" in m.lower() for m in messages),
            f"Expected 'compiled' in log messages, got: {messages}",
        )
        self.assertTrue(
            any("constrained_decoder" in m.lower() for m in messages),
            f"Expected filename in log messages, got: {messages}",
        )

    def test_log_contains_timestamp(self):
        import src.constrained_decoder as module
        with self.assertLogs("src.constrained_decoder", level="INFO") as ctx:
            importlib.reload(module)
        messages = [r.getMessage() for r in ctx.records]
        import re
        timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertTrue(
            any(timestamp_pattern.search(m) for m in messages),
            f"Expected timestamp in log messages, got: {messages}",
        )


if __name__ == "__main__":
    unittest.main()
