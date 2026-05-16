"""Tests for src/response_generator.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from unittest.mock import MagicMock

from src.response_generator import generate_response


def _make_mocks(prompt_len: int = 5, total_output_len: int = 10):
    """Build minimal model/tokenizer mocks for generate_response tests."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<prompt>"
    tokenizer.eos_token_id = 0

    # encode().to(device) → input_ids mock with .shape[1] = prompt_len
    input_ids = MagicMock()
    input_ids.shape = [1, prompt_len]
    encoded = MagicMock()
    encoded.to.return_value = input_ids
    tokenizer.encode.return_value = encoded

    # generate() returns a real tensor so slicing works
    output_tensor = torch.zeros(1, total_output_len, dtype=torch.long)
    model = MagicMock()
    model.generate.return_value = output_tensor

    tokenizer.decode.return_value = "  test response  "

    device = MagicMock()
    return model, tokenizer, device


class TestGenerateResponse:
    def test_returns_stripped_string(self):
        model, tokenizer, device = _make_mocks()
        result = generate_response(model, tokenizer, device, "what shoes should I buy?")
        assert result == "test response"

    def test_calls_model_generate(self):
        model, tokenizer, device = _make_mocks()
        generate_response(model, tokenizer, device, "query")
        assert model.generate.called

    def test_passes_max_new_tokens(self):
        model, tokenizer, device = _make_mocks()
        generate_response(model, tokenizer, device, "query", max_new_tokens=256)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 256

    def test_passes_temperature(self):
        model, tokenizer, device = _make_mocks()
        generate_response(model, tokenizer, device, "query", temperature=0.3)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    def test_uses_do_sample(self):
        model, tokenizer, device = _make_mocks()
        generate_response(model, tokenizer, device, "query")
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True

    def test_strips_prompt_tokens(self):
        """Only newly generated tokens (after prompt_len) are decoded."""
        prompt_len = 5
        model, tokenizer, device = _make_mocks(prompt_len=prompt_len, total_output_len=10)

        captured = {}

        def fake_decode(tokens, **kwargs):
            captured["tokens"] = tokens
            return "decoded"

        tokenizer.decode.side_effect = fake_decode
        generate_response(model, tokenizer, device, "query")

        # Decoded tokens should be a slice of length (10 - 5) = 5
        assert len(captured["tokens"]) == 5

    def test_apply_chat_template_called(self):
        model, tokenizer, device = _make_mocks()
        generate_response(model, tokenizer, device, "my query")
        assert tokenizer.apply_chat_template.called

    def test_empty_query_handled(self):
        model, tokenizer, device = _make_mocks()
        result = generate_response(model, tokenizer, device, "")
        assert isinstance(result, str)
