"""
Vanilla LLM Response Generator

Generates an unconstrained response to a user query using the already-loaded
Qwen model. This is Part 1 of the two-part pipeline: the full, helpful answer
the user receives before any ad injection.

The model and tokenizer are passed in from the ConstrainedBeamSearchDecoder
so there is no second model load.
"""

import torch


def generate_response(
    model,
    tokenizer,
    device: torch.device,
    query: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Generate a vanilla (unconstrained) LLM response for a user query.

    Args:
        model: A loaded AutoModelForCausalLM (Qwen2.5 or compatible).
        tokenizer: The corresponding AutoTokenizer.
        device: The torch device the model lives on.
        query: The user's question or search query.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (higher = more varied output).

    Returns:
        The generated response text, decoded and stripped of the prompt.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question clearly "
                "and informatively in 2-4 paragraphs."
            ),
        },
        {"role": "user", "content": query},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
