"""
Vanilla LLM Response Generator (Part 1 of the AD recommender pipeline)

Generates a plain LLM answer for a user query using the already-loaded
Qwen model, with no output constraints.
"""

import torch


def generate_response(
    model,
    tokenizer,
    device,
    query: str,
    max_new_tokens: int = 400,
) -> str:
    """
    Generate a vanilla LLM response for the user query.

    Args:
        model:          Loaded AutoModelForCausalLM
        tokenizer:      Matching AutoTokenizer
        device:         torch.device (cpu / cuda / mps)
        query:          The user's natural-language question
        max_new_tokens: Maximum tokens to generate

    Returns:
        The generated response string (decoded, special tokens stripped)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the user's question clearly and concisely."
            ),
        },
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
