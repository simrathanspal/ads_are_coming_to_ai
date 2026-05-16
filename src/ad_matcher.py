"""
Semantic Ad Matcher

Part 2 of the AD recommender pipeline: given the LLM response and a list of
Commercial Intents (CIs) from constrained beam search, find relevant ads by
comparing cosine similarity between the response embedding and each CI embedding.

Uses mean-pooled last hidden states of the shared Qwen model for embedding —
no separate sentence encoder needed.
"""

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class MatchedAd:
    """An ad matched to the LLM response, with its semantic similarity score."""
    product_id: str
    title: str
    brand: str
    category: str
    description: str
    url: str
    price: Optional[float]
    ci_text: str       # The CI that triggered this match
    match_score: float  # Cosine similarity between CI and response (0–1)


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool last hidden states, masking out padding tokens.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]  (1 = real token, 0 = padding)

    Returns:
        [batch, hidden_dim] pooled representation
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


def _embed(model, tokenizer, device, texts: list[str]) -> torch.Tensor:
    """
    Embed a batch of texts using mean-pooled last hidden states, L2-normalized.

    Args:
        model: Loaded HuggingFace CausalLM (output_hidden_states=True)
        tokenizer: Matching tokenizer
        device: Device the model is on
        texts: List of strings to embed

    Returns:
        [len(texts), hidden_dim] float tensor, each row L2-normalized
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    last_hidden = outputs.hidden_states[-1].float()  # [batch, seq_len, hidden]
    pooled = _mean_pool(last_hidden, attention_mask)  # [batch, hidden]
    return F.normalize(pooled, p=2, dim=-1)


def match_ads(
    model,
    tokenizer,
    device,
    response: str,
    ci_results,         # list[CIResult] from constrained_decoder
    ci_ad_index_path: str,
    threshold: float = 0.70,
    max_ads: int = 2,
) -> list[MatchedAd]:
    """
    Match Commercial Intents to ads based on cosine similarity to the LLM response.

    Embeds the response and each CI together (one forward pass per batch),
    filters by threshold, deduplicates by product_id, and caps at max_ads.

    Args:
        model: Loaded HuggingFace CausalLM
        tokenizer: Matching tokenizer
        device: Device the model is on
        response: The vanilla LLM response text (Part 1 output)
        ci_results: List of CIResult from constrained decoder (Part 2 input)
        ci_ad_index_path: Path to ci_ad_index.json
        threshold: Minimum cosine similarity to include an ad (default 0.70)
        max_ads: Maximum number of ads to return (default 2)

    Returns:
        List of MatchedAd objects sorted by match_score descending
    """
    if not ci_results or not response.strip():
        return []

    with open(ci_ad_index_path) as f:
        ci_ad_index = json.load(f)

    ci_texts = [r.ci_text for r in ci_results]

    # Embed response + all CIs in one forward pass
    all_texts = [response] + ci_texts
    embeddings = _embed(model, tokenizer, device, all_texts)

    response_emb = embeddings[0:1]    # [1, hidden]
    ci_embeddings = embeddings[1:]    # [n_cis, hidden]

    # Cosine similarities (vectors are L2-normalized, so dot product = cosine)
    similarities = (ci_embeddings @ response_emb.T).squeeze(1)  # [n_cis]

    # Sort CIs by similarity (highest first)
    scored = sorted(
        zip(ci_texts, similarities.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    matched: list[MatchedAd] = []
    seen_product_ids: set[str] = set()

    for ci_text, sim_score in scored:
        if sim_score < threshold:
            continue

        for ad in ci_ad_index.get(ci_text, []):
            pid = ad["product_id"]
            if pid in seen_product_ids:
                continue
            seen_product_ids.add(pid)
            matched.append(MatchedAd(
                product_id=pid,
                title=ad["title"],
                brand=ad["brand"],
                category=ad["category"],
                description=ad["description"],
                url=ad["url"],
                price=ad.get("price"),
                ci_text=ci_text,
                match_score=sim_score,
            ))
            if len(matched) >= max_ads:
                return matched

    return matched
