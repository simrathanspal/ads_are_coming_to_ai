"""Unit tests for src/response_generator.py"""

import torch
from unittest.mock import MagicMock

from src.response_generator import generate_response


def _make_mocks(response_text: str = "Test response."):
    """Build lightweight model + tokenizer mocks for generate_response."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = "<prompt>"
    # encode() must return a tensor so .to(device) and .shape work
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer.decode.return_value = response_text

    model = MagicMock()
    # generate() returns prompt tokens + new tokens
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    return model, tokenizer


def test_returns_string():
    model, tokenizer = _make_mocks("Hello, world!")
    result = generate_response(model, tokenizer, "cpu", "What is hiking?")
    assert isinstance(result, str)


def test_calls_model_generate():
    model, tokenizer = _make_mocks()
    generate_response(model, tokenizer, "cpu", "query")
    model.generate.assert_called_once()


def test_strips_whitespace():
    model, tokenizer = _make_mocks("  padded response  ")
    result = generate_response(model, tokenizer, "cpu", "query")
    assert result == "padded response"


def test_uses_chat_template():
    model, tokenizer = _make_mocks()
    generate_response(model, tokenizer, "cpu", "my query")
    tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = tokenizer.apply_chat_template.call_args
    assert call_kwargs.kwargs.get("tokenize") is False
    assert call_kwargs.kwargs.get("add_generation_prompt") is True


def test_decodes_only_new_tokens():
    """Verifies that only tokens after the prompt are decoded."""
    model, tokenizer = _make_mocks("new content")
    # prompt length = 5, total output = 10 → 5 new tokens
    prompt_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    full_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    tokenizer.encode.return_value = prompt_tensor
    model.generate.return_value = full_output

    generate_response(model, tokenizer, "cpu", "query")

    decoded_input = tokenizer.decode.call_args[0][0]
    # Should only contain tokens 6–10, not the prompt tokens 1–5
    assert decoded_input.tolist() == [6, 7, 8, 9, 10]
