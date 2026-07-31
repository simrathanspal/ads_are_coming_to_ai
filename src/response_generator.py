"""Vanilla LLM response generation (Part 1 of the AD recommender pipeline)."""

import torch


def generate_response(model, tokenizer, device, query: str, max_new_tokens: int = 256) -> str:
    """
    Generate a vanilla LLM response for the given query.

    Reuses the same model loaded by ConstrainedBeamSearchDecoder to avoid
    loading a second model into memory.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question clearly and concisely.",
        },
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
