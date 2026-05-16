"""
Vanilla LLM Response Generator

Part 1 of the AD recommender pipeline: generate an unconstrained LLM response
for a user query, reusing the already-loaded model from the constrained decoder.
"""

import torch


def generate_response(
    model,
    tokenizer,
    device,
    query: str,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a vanilla (unconstrained) LLM response for a user query.

    Reuses the model already loaded by ConstrainedBeamSearchDecoder, so no
    second model load is needed.

    Args:
        model: Loaded HuggingFace CausalLM model (from decoder.model)
        tokenizer: Matching tokenizer (from decoder.tokenizer)
        device: Device the model is on (from decoder.device)
        query: User's natural-language question
        max_new_tokens: Maximum new tokens to generate

    Returns:
        Decoded response text, stripped of leading/trailing whitespace
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question clearly and concisely.",
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
